"""Private simulator transport for DynamicsAdapt experiments."""

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
from pathlib import Path

import numpy as np

from orchestrator.resources import sim_env
from orchestrator.transport import receive, send
from orchestrator.runtime import ROOT


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class SimulatorClient:
    """Own one persistent simulator process; never retry a simulator request."""

    def __init__(
        self, run_dir, *, parameters, gpu=0, mode="state", simulator_python=None,
        startup_timeout=1200.0, step_timeout=120.0,
        worker_module="integrations.friction_rl.worker",
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
                tempfile.TemporaryDirectory(prefix="dynamicsadapt-simulator-")
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
