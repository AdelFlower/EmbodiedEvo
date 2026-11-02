"""One private simulator process for a dynamics-repair trial.

The private Unix socket carries protocol messages independently of Isaac Sim's stdout.
Only the four existing robot primitives enter the submitted Python namespace.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import socket
import traceback
from pathlib import Path

import numpy as np

from environments import execute
from environments.base import SourceValidationError
from orchestrator.files import write_json
from orchestrator.session_client import receive, send
from simulators.observations import export_observation
from simulators.robodojo import InvalidControlAction
from simulators.video import write_videos

OUTPUT_LIMIT = 256 * 1024


class ExecutionTimeout(BaseException):
    pass


@contextlib.contextmanager
def deadline(seconds):
    expired = False

    def timeout(signum, frame):
        nonlocal expired
        expired = True
        raise ExecutionTimeout("Program time budget exhausted")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
        if expired:
            raise ExecutionTimeout("Program time budget exhausted")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class TrialWorker:
    """Execute programs and request global budget grants from the supervisor."""

    def __init__(self, env, config, event):
        self.env, self.config, self.event = env, config, event
        self.workspace = Path(config["run"]) / "workspace"
        self.private = Path(config["trial_dir"])
        self.active_execution = None
        self.last_frames = {}
        self.latest = config["initial_observation"]
        self.env.execution_phase = "playground" if config["allow_reset"] else "test"
        self.env.reset(seed=config["settings"].get("seed", 0))
        original_control = self.env.control_step
        original_reset = self.env.reset_from_program

        def control(action):
            if self.env.done:
                return original_control(action)
            self.grant("action")
            before = self.env.control_steps
            completed = False
            try:
                result = original_control(action)
                completed = True
                return result
            except InvalidControlAction:
                # The adapter guarantees that input rejection preceded physics.
                completed = True
                raise
            finally:
                consumed = self.env.control_steps - before
                # A native failure can happen after physics advanced but before
                # the adapter updated its counter. Never refund that reservation.
                self.event({
                    "event": "action_completed",
                    "consumed": consumed if completed else max(1, consumed),
                    "consumption_unknown": not completed,
                })

        def reset():
            if not config["allow_reset"]:
                raise PermissionError("reset() is forbidden during frozen policy evaluation")
            self.grant("episode", previous=self.status())
            return original_reset()

        self.env.control_step = control
        self.env.reset_from_program = reset
        export_observation(env, self.workspace / self.latest, workspace=self.workspace)

    def grant(self, resource, **details):
        answer = self.event({"event": "reserve", "resource": resource, **details})
        if not answer["ok"]:
            raise RuntimeError(answer["error"])

    def status(self):
        return {
            "success": bool(self.env.success),
            "terminated": bool(self.env.terminated),
            "truncated": bool(self.env.truncated),
            "native_action_limit": self.env.native_step_limit,
            "native_steps_remaining": max(0, self.env.max_steps - self.env.control_steps),
            "observation": self.latest,
        }

    def execute(self, code, execution):
        folder = self.workspace / "observations" / f"{execution:06d}"
        self.active_execution = execution
        # Keep the collection alive if the adapter raises after CodeTask hands
        # it off and replaces _frames, before execute can return its result.
        self.last_frames = self.env._frames
        try:
            with deadline(self.config["execution_seconds"]):
                result = execute(self.env, code)
        except SourceValidationError as exc:
            # Rejected model source is a model failure, not a simulator fault.
            (folder / "stdout.txt").write_text("")
            (folder / "stderr.txt").write_text(str(exc))
            self.active_execution = None
            return {**self.status(), "ok": False, "stdout": "", "stderr": str(exc),
                    "execution": execution, "classification": "model_code_error"}
        classification = None if result.ok else "model_code_error"
        self.last_frames = result.frames
        # Save the captured code feedback before optional image/video encoding.
        (folder / "stdout.txt").write_text(result.stdout)
        (folder / "stderr.txt").write_text(result.stderr)
        write_json(self.private / f"execution-{execution:06d}.json", {
            "reward": result.reward,
            "success": result.success,
            "ok": result.ok,
            "classification": classification,
            "snapshot": self.env.host_snapshot(),
        })
        export_observation(self.env, folder, workspace=self.workspace)
        videos = write_videos(result.frames, folder, stem="execution", fps=self.config["fps"])
        self.latest = str(folder.relative_to(self.workspace))
        self.active_execution = None
        return {
            **self.status(),
            "ok": result.ok,
            "classification": classification,
            "stdout": result.stdout[-OUTPUT_LIMIT:],
            "stderr": result.stderr[-OUTPUT_LIMIT:],
            "execution": execution,
            "videos": {
                key: str(Path(value).relative_to(self.workspace)) for key, value in videos.items()
            },
        }

    def preserve_partial_output(self):
        """Retain model output and collected frames even when execution times out."""
        folder = self.workspace / "observations" / f"{self.active_execution:06d}"
        stdout, stderr = (stream.getvalue() for stream in self.env._program_output)
        (folder / "stdout.txt").write_text(stdout)
        (folder / "stderr.txt").write_text(stderr)
        frames = self.last_frames or self.env._frames
        if frames:
            np.savez_compressed(folder / "partial-frames.npz", **{
                camera: np.stack(images) for camera, images in frames.items() if images
            })
        return {"stdout": stdout[-OUTPUT_LIMIT:], "stderr": stderr[-OUTPUT_LIMIT:]}


def run_worker(stream, config, *, builder=None):
    if builder is None:
        from simulators.dynamics_task import build_environment

        builder = build_environment
    env = worker = None
    private = Path(config["trial_dir"])

    def event(message):
        send(stream, message)
        return receive(stream)

    try:
        env = builder(config["settings"])
        worker = TrialWorker(env, config, event)
        write_json(private / "initial-state.json", env.host_snapshot())
        send(stream, {"ok": True, **worker.status()})
        while True:
            command = receive(stream)
            if command["op"] == "close":
                write_json(private / "final-state.json", env.host_snapshot())
                send(stream, {"ok": True, **worker.status()})
                return
            if command["op"] == "exec":
                response = worker.execute(command["code"], command["execution"])
            else:
                raise ValueError("Unknown worker command")
            send(stream, response)
    except BaseException as exc:
        write_json(private / "worker-error.json", {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "classification": "unclassified",
        })
        partial = {}
        if worker is not None and worker.active_execution is not None:
            try:
                partial = worker.preserve_partial_output()
            except Exception as capture_error:
                write_json(private / "partial-output-error.json", {
                    "type": type(capture_error).__name__, "message": str(capture_error),
                })
        with contextlib.suppress(OSError):
            send(stream, {
                "ok": False,
                "fatal": True,
                "classification": "unclassified",
                "error": "Simulator execution failed; the original attempt was preserved.",
                **partial,
            })
    finally:
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--socket", required=True)
    args = parser.parse_args(argv)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(args.socket)
        with connection.makefile("rwb") as stream:
            config = receive(stream)
            config.update(run=str(args.workspace.parent), trial_dir=str(args.run_dir))
            run_worker(stream, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
