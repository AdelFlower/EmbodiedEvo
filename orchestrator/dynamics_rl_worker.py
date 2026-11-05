"""Private simulator worker for privileged-state RL; no learner dependencies."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import socket
import traceback
from pathlib import Path

import numpy as np

from orchestrator.files import write_json
from orchestrator.session_client import receive, send
from simulators.dynamics_rl import DynamicsRLTask, json_value


class RLWorker:
    def __init__(self, env, run_dir):
        self.env = env
        self.run_dir = Path(run_dir).resolve()
        self.episode_dir = None
        self.state = None
        self.previous_action = np.zeros(7, dtype=np.float32)
        self.transitions = self.commands = None
        self.finished = False
        self.episode_initialized = False
        self.failed = False

    def _record(self, stream, value):
        stream.write(json.dumps(json_value(value), allow_nan=False) + "\n")
        stream.flush()

    def finish_episode(self, reason):
        if self.episode_dir is None or self.finished:
            return
        self.env.close_videos()
        if self.episode_initialized:
            write_json(self.episode_dir / "final-state.json", self.env.host_snapshot())
        write_json(self.episode_dir / "result.json", {
            "success": bool(self.env.success) and self.episode_initialized and not self.failed,
            "complete": self.episode_initialized and not self.failed and bool(self.env.done),
            "terminated": self.episode_initialized and bool(self.env.terminated),
            "truncated": self.episode_initialized and bool(self.env.truncated),
            "steps": self.env.episode_steps if self.episode_initialized else 0,
            "reason": reason,
            "videos": dict(self.env.video_paths),
        })
        self.finished = True

    def _close_records(self):
        for stream in (self.transitions, self.commands):
            if stream is not None:
                stream.close()
        self.transitions = self.commands = None

    def reset(self, command):
        from integrations.dynamics_rl.features import encode_observation

        self.finish_episode("host_reset")
        self._close_records()
        episode = Path(command["episode_dir"]).resolve()
        if not episode.is_relative_to(self.run_dir) or episode == self.run_dir:
            raise ValueError("Episode evidence must live beneath the explicit run directory")
        episode.mkdir(parents=True, exist_ok=False)
        self.episode_dir = episode
        self.finished = False
        self.failed = False
        self.episode_initialized = False
        write_json(episode / "scene.json", command["scene"])
        self.commands = (episode / "commands.jsonl").open("w")
        self.transitions = (episode / "transitions.jsonl").open("w")
        self._record(self.commands, command)
        self.env.reset_episode(command["scene"], episode)
        self.episode_initialized = True
        self.state = self.env.rl_state()
        self.previous_action = np.zeros(7, dtype=np.float32)
        write_json(episode / "initial-state.json", self.env.host_snapshot())
        obs = encode_observation(self.state, self.previous_action, self.env.episode_steps)
        result = self.reply(obs, reward=0.0, native_reward=0.0)
        self._record(self.transitions, {"reset": True, **result})
        return result

    def reply(self, obs, *, reward, native_reward):
        return {
            "ok": True,
            "obs": np.asarray(obs, dtype=np.float32).tolist(),
            "reward": float(reward),
            "native_reward": float(native_reward),
            "terminated": bool(self.env.terminated),
            "truncated": bool(self.env.truncated),
            "info": {
                "success": bool(self.env.success),
                "steps": self.env.episode_steps,
                "videos": dict(self.env.video_paths),
            },
            "state": self.state,
        }

    def step(self, command):
        from integrations.dynamics_rl.features import (
            decode_action,
            encode_observation,
            shaped_reward,
        )

        if self.state is None:
            raise RuntimeError("Reset before stepping the RL simulator")
        if self.finished or self.env.done:
            raise RuntimeError("The episode has ended; no additional physics action is allowed")
        self._record(self.commands, command)
        action = np.asarray(command["action"], dtype=np.float32)
        native_action = decode_action(action, self.env.left_pose, self.env.left_grip)
        _, native_reward, terminated, truncated, _ = self.env.control_step(native_action)
        next_state = self.env.rl_state()
        reward = shaped_reward(
            self.state, next_state, terminated=terminated, truncated=truncated,
            native_reward=native_reward,
        )
        self.state = next_state
        self.previous_action = action
        obs = encode_observation(self.state, action, self.env.episode_steps)
        result = self.reply(obs, reward=reward, native_reward=native_reward)
        result["applied_action"] = json_value(native_action)
        self._record(self.transitions, {"action": action.tolist(), **result})
        if terminated or truncated:
            self.finish_episode("native_episode_end")
        return result

    def inspect(self):
        if self.state is None:
            return {"ok": True, "ready": True, "state": None}
        return {"ok": True, "state": self.state, "snapshot": self.env.host_snapshot()}

    def close(self):
        try:
            self.finish_episode("host_close")
        finally:
            self._close_records()


def run_worker(stream, run_dir, *, parameters, mode="state", builder=None, worker_class=RLWorker):
    env = worker = None
    run_dir = Path(run_dir)
    try:
        factory = DynamicsRLTask if builder is None else builder
        env = factory(mode=mode, settings=parameters)
        worker = worker_class(env, run_dir)
        send(stream, {"ok": True, "ready": True, "mode": mode,
                      "parameters": parameters})
        while True:
            command = receive(stream)
            operation = command.get("op")
            if operation == "close":
                worker.close()
                send(stream, {"ok": True, "closed": True})
                return 0
            if operation not in {"reset", "step", "inspect"}:
                raise ValueError(f"Unknown RL worker operation: {operation}")
            result = (
                worker.inspect() if operation == "inspect" else getattr(worker, operation)(command)
            )
            send(stream, json_value(result))
    except BaseException as exc:
        error = {
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(), "classification": "unclassified",
        }
        write_json(run_dir / "worker-error.json", error)
        if worker is not None and worker.episode_dir is not None:
            worker.failed = True
            write_json(worker.episode_dir / "worker-error.json", error)
        with contextlib.suppress(OSError, ValueError):
            send(stream, {"ok": False, "fatal": True, "error": error})
        return 1
    finally:
        if worker is not None:
            try:
                worker.close()
            except Exception as exc:
                write_json(run_dir / "evidence-close-error.json", {
                    "type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(),
                })
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--mode", choices=("state", "render"), default="state")
    parser.add_argument("--parameters", type=json.loads, required=True)
    args = parser.parse_args(argv)
    if (not isinstance(args.parameters, dict) or
            set(args.parameters) != {"friction_scale", "mass_scale", "restitution"} or
            any(type(value) not in (int, float) for value in args.parameters.values())):
        parser.error("--parameters requires a JSON object with the three numeric parameters")
    if not re.match(r"^\d{12}", args.run_dir.name):
        raise ValueError("--run-dir must start with YYYYMMDDHHMM")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(args.socket)
        with connection.makefile("rwb") as stream:
            return run_worker(stream, args.run_dir, mode=args.mode,
                              parameters=args.parameters)


if __name__ == "__main__":
    raise SystemExit(main())
