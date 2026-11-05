"""Host-only socket transport and Gym interface for the independent RL experiment."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np

from integrations.dynamics_rl.features import OBS_DIM, sample_scene
from orchestrator.resources import sim_env
from orchestrator.session_client import receive, send

ROOT = Path(__file__).resolve().parents[2]


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class SimulatorClient:
    """Own one persistent simulator process; never retry a simulator request."""

    def __init__(
        self, run_dir, *, parameters, gpu=0, mode="state", simulator_python=None,
        startup_timeout=1200.0, step_timeout=120.0,
        worker_module="orchestrator.dynamics_rl_worker",
    ):
        self.run_dir = Path(run_dir).resolve()
        if not re.match(r"^\d{12}(?:_|$)", self.run_dir.name):
            raise ValueError("Simulator --run-dir must start with YYYYMMDDHHMM")
        if mode not in {"state", "render"}:
            raise ValueError("mode must be state or render")
        self.parameters = dict(parameters)
        self.worker_module = worker_module
        for name, value in (("startup_timeout", startup_timeout), ("step_timeout", step_timeout)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.gpu, self.mode = int(gpu), mode
        if self.gpu < 0:
            raise ValueError("gpu must be nonnegative")
        self.interpreter = Path(simulator_python or ROOT / ".venv-robodojo/bin/python").resolve()
        self.startup_timeout, self.step_timeout = float(startup_timeout), float(step_timeout)
        self.resources = contextlib.ExitStack()
        self.process = self.connection = self.stream = None
        self.started = self.closed = self.failed = False
        self.failure_count = 0
        self.latest_response = None

    def start(self):
        if self.closed or self.failed:
            raise RuntimeError("This simulator client cannot be restarted")
        if self.started:
            return self.latest_response
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive log creation protects failed attempts even when the caller
        # already made the directory for its configuration.
        self.stdout = self.resources.enter_context(
            (self.run_dir / "simulator.stdout.log").open("xb")
        )
        try:
            self.stderr = self.resources.enter_context(
                (self.run_dir / "simulator.stderr.log").open("xb")
            )
            folder = self.resources.enter_context(
                tempfile.TemporaryDirectory(prefix="embodiedevo-rl-")
            )
            self.socket_path = Path(folder) / "worker.sock"
            listener = self.resources.enter_context(
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            )
            listener.bind(str(self.socket_path))
            listener.listen(1)
            listener.settimeout(0.2)
            command = [
                str(self.interpreter), "-m", self.worker_module,
                "--run-dir", str(self.run_dir), "--socket", str(self.socket_path),
                "--gpu", str(self.gpu), "--mode", self.mode,
                "--parameters", json.dumps(self.parameters),
            ]
            _write_json(self.run_dir / "transport-config.json", {
                "command": command, "mode": self.mode,
                "parameters": self.parameters,
                "startup_timeout": self.startup_timeout, "step_timeout": self.step_timeout,
            })
            self.process = subprocess.Popen(
                command, cwd=ROOT, env=sim_env(str(self.gpu), "robodojo"),
                stdout=self.stdout, stderr=self.stderr, start_new_session=True,
            )
            deadline = time.monotonic() + self.startup_timeout
            while True:
                if self.process.poll() is not None:
                    raise ConnectionError(
                        f"Simulator exited before connecting ({self.process.returncode}); "
                        f"inspect {self.run_dir}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError("Simulator startup timed out; no automatic retry")
                try:
                    connection, _ = listener.accept()
                    break
                except TimeoutError:
                    continue
            listener.close()
            self.connection = self.resources.enter_context(connection)
            self.connection.settimeout(max(0.001, deadline - time.monotonic()))
            self.stream = self.resources.enter_context(self.connection.makefile("rwb"))
            answer = receive(self.stream)
            if not answer.get("ok") or not answer.get("ready"):
                raise RuntimeError(f"Simulator failed to initialize: {answer}")
            self.connection.settimeout(self.step_timeout)
            self.started, self.latest_response = True, answer
            return answer
        except BaseException as exc:
            self._record_failure("startup", exc)
            self.close()
            raise

    def _record_failure(self, operation, exc):
        self.failed = True
        self.failure_count += 1
        if self.run_dir.is_dir():
            _write_json(self.run_dir / f"transport-failure-{self.failure_count:06d}.json", {
                "operation": operation, "classification": "unclassified",
                "type": type(exc).__name__, "error": str(exc),
                "time": time.time(), "automatic_retry": False,
            })

    def request(self, payload, *, timeout=None):
        if not self.started or self.closed or self.failed:
            raise RuntimeError("Simulator is not available; requests are never retried")
        try:
            self.connection.settimeout(self.step_timeout if timeout is None else timeout)
            send(self.stream, payload)
            answer = receive(self.stream)
            if not answer.get("ok"):
                raise RuntimeError(f"Simulator {payload['op']} failed: {answer}")
            self.latest_response = answer
            return answer
        except (OSError, ValueError, RuntimeError) as exc:
            self._record_failure(payload.get("op"), exc)
            raise
        finally:
            if self.connection is not None and not self.closed:
                self.connection.settimeout(self.step_timeout)

    def reset(self, scene, episode_dir):
        episode_dir = Path(episode_dir).resolve()
        episode_dir.relative_to(self.run_dir)
        return self.request({
            "op": "reset", "scene": copy.deepcopy(scene), "episode_dir": str(episode_dir),
        }, timeout=self.startup_timeout)

    def step(self, action):
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("Action must contain seven finite values")
        if np.any(np.abs(values) > 1.000001):
            raise ValueError("Action must lie in [-1, 1]")
        return self.request({"op": "step", "action": values.tolist()})

    def inspect(self):
        return self.request({"op": "inspect"})

    def close(self):
        if self.closed:
            return None if self.process is None else self.process.returncode
        if self.started and not self.failed:
            # request() preserves the failure. Cleanup must not replace an
            # original episode exception with a second transport error.
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                self.request({"op": "close"}, timeout=self.startup_timeout)
        self.closed = True
        self.resources.close()
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)
        code = None if self.process is None else self.process.returncode
        if self.run_dir.is_dir():
            _write_json(self.run_dir / "transport-result.json", {
                "returncode": code, "failed": self.failed, "closed": True,
            })
        return code

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class RemoteEnv(gym.Env):
    """Gymnasium facade, with independent layout RNG and complete host evidence."""

    metadata = {"render_modes": []}

    def __init__(
        self, client, *, seed=0, radius=0.03, scene_seeds: Sequence[int] | None = None,
        episode_root=None, on_transition: Callable | None = None,
        on_episode: Callable | None = None,
    ):
        super().__init__()
        self.client = client
        self.seed_offset, self.radius = int(seed), float(radius)
        self.scene_seeds = None if scene_seeds is None else tuple(int(s) for s in scene_seeds)
        if self.scene_seeds == ():
            raise ValueError("scene_seeds must not be empty")
        self.episode_root = Path(episode_root or client.run_dir / "episodes").resolve()
        self.episode_root.relative_to(Path(client.run_dir).resolve())
        self.on_transition, self.on_episode = on_transition, on_episode
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
        self.episode_index = 0
        self.latest_response = self.current_state = None
        self.pending_scene = None
        self.episode_dir = self.transitions = None
        self.active = False
        self.episode_reward = 0.0
        self.episode_actions = 0
        self.episode_started = 0.0
        self.last_episode = None

    def set_scene(self, scene):
        """Choose the next reset layout explicitly, without advancing simulation."""
        self.pending_scene = copy.deepcopy(scene)

    def _observation(self, response):
        value = np.asarray(response["obs"], dtype=np.float32)
        if value.shape != (OBS_DIM,) or not np.isfinite(value).all():
            raise ValueError(
                f"Simulator returned an invalid observation; expected {OBS_DIM} values"
            )
        return value

    def _finish_episode(self, *, complete, error=None):
        if not self.active:
            return
        self.active = False
        if self.transitions is not None:
            self.transitions.close()
            self.transitions = None
        response = self.latest_response or {}
        info = response.get("info", {})
        record = {
            "episode": self.episode_index, "complete": bool(complete),
            "success": bool(complete and info.get("success", False)),
            "terminated": bool(response.get("terminated", False)),
            "truncated": bool(response.get("truncated", False)),
            "actions": self.episode_actions, "reward": self.episode_reward,
            "elapsed_seconds": time.monotonic() - self.episode_started,
            "classification": None if complete else "incomplete",
            "error": error, "episode_dir": str(self.episode_dir),
        }
        _write_json(self.episode_dir / "learner-result.json", record)
        self.last_episode = record
        if self.on_episode is not None:
            self.on_episode(record)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.active:
            self._finish_episode(complete=False, error="Explicit reset before episode termination")
        if seed is not None:
            self.seed_offset = int(seed)
        options = {} if options is None else options
        explicit = options.get("scene", self.pending_scene)
        self.pending_scene = None
        if explicit is not None:
            scene = copy.deepcopy(explicit)
            layout_seed = None
        else:
            layout_seed = (
                self.scene_seeds[self.episode_index % len(self.scene_seeds)]
                if self.scene_seeds is not None else self.seed_offset + self.episode_index
            )
            scene = sample_scene(layout_seed, radius=self.radius)
        self.episode_index += 1
        self.episode_dir = self.episode_root / f"{self.episode_index:06d}"
        # The worker exclusively creates the episode directory before acting.
        response = self.client.reset(scene, self.episode_dir)
        self.latest_response, self.current_state = response, response["state"]
        self.episode_reward, self.episode_actions = 0.0, 0
        self.episode_started = time.monotonic()
        self.active = True
        self.transitions = (self.episode_dir / "learner-transitions.jsonl").open("x")
        _write_json(self.episode_dir / "learner-layout.json", {
            "layout_seed": layout_seed, "scene": scene,
        })
        return self._observation(response), dict(response.get("info", {}))

    def step(self, action):
        if not self.active:
            raise RuntimeError("Reset before stepping; an ended episode cannot continue")
        action = np.asarray(action, dtype=np.float32)
        previous = self.latest_response
        try:
            response = self.client.step(action)
            observation = self._observation(response)
            reward = float(response["reward"])
            if not np.isfinite(reward):
                raise ValueError("Simulator returned a non-finite reward")
            self.latest_response, self.current_state = response, response["state"]
            self.episode_actions += 1
            self.episode_reward += reward
            terminated, truncated = bool(response["terminated"]), bool(response["truncated"])
            transition = {
                "step": self.episode_actions, "obs": previous["obs"],
                "action": action.tolist(), "next_obs": response["obs"],
                "reward": reward, "native_reward": response.get("native_reward", 0.0),
                "terminated": terminated, "truncated": truncated,
                "success": bool(response.get("info", {}).get("success", False)),
                "applied_action": response.get("applied_action"),
            }
            self.transitions.write(json.dumps(transition, allow_nan=False) + "\n")
            self.transitions.flush()
            if self.on_transition is not None:
                self.on_transition(previous, action.copy(), response)
            info = dict(response.get("info", {}))
            info["native_reward"] = response.get("native_reward", 0.0)
            if terminated or truncated:
                self._finish_episode(complete=True)
                info["episode"] = {"r": self.episode_reward, "l": self.episode_actions,
                                   "t": time.monotonic() - self.episode_started}
            return observation, reward, terminated, truncated, info
        except (OSError, ValueError, RuntimeError) as exc:
            self._finish_episode(complete=False, error=f"{type(exc).__name__}: {exc}")
            raise

    def close(self):
        self._finish_episode(complete=False, error="Environment closed before episode termination")
        return self.client.close()
