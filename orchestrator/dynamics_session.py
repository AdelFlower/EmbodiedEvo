"""Host-owned controller for isolated dynamics-repair simulator trials."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from orchestrator.files import append_jsonl, write_json
from orchestrator.resources import sim_env
from orchestrator.session_client import receive, send

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = {"friction_scale": (0.05, 2.0), "mass_scale": (0.25, 4.0),
              "restitution": (0.0, 1.0)}


class SubprocessWorker:
    """A simulator lifetime, with all native stdout/stderr kept on the host."""

    def __init__(self, config, interpreter, gpu):
        self.config = config
        self.resources = contextlib.ExitStack()
        # Startup covers launch, connection, scene construction and the initial reset;
        # later replies are bounded by the program deadline plus export time.
        self.startup_timeout = config["startup_seconds"]
        self.timeout = config["execution_seconds"] + 60
        try:
            # A short private path works across Docker mounts as well as local Python.
            # This directory is never mounted into the coding-agent container.
            folder = self.resources.enter_context(
                tempfile.TemporaryDirectory(prefix="embodiedevo-worker-")
            )
            self.socket_path = Path(folder) / "worker.sock"
            self.listener = self.resources.enter_context(
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            )
            self.listener.bind(str(self.socket_path))
            self.listener.listen(1)
            self.listener.settimeout(0.2)
            trial = Path(config["trial_dir"]).resolve()
            workspace = (Path(config["run"]) / "workspace").resolve()
            self.stdout = self.resources.enter_context((trial / "simulator.stdout.log").open("wb"))
            self.stderr = self.resources.enter_context((trial / "simulator.stderr.log").open("wb"))
            self.process = subprocess.Popen(
                [str(interpreter), "-m", "orchestrator.dynamics_worker",
                 "--run-dir", str(trial), "--workspace", str(workspace),
                 "--socket", str(self.socket_path)],
                cwd=ROOT, env=sim_env(gpu, "robodojo"),
                stdout=self.stdout, stderr=self.stderr, start_new_session=True,
            )
        except BaseException:
            self.resources.close()
            raise

    def response(self, event):
        while True:
            answer = receive(self.stream)
            if "event" not in answer:
                return answer
            send(self.stream, event(answer))

    def initialize(self, event):
        deadline = time.monotonic() + self.startup_timeout
        while True:
            if self.process.poll() is not None:
                raise ConnectionError(
                    f"Simulator exited before connecting (status {self.process.returncode}); "
                    "inspect this trial's simulator logs"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("Simulator connection timed out; no automatic retry")
            try:
                connection, _ = self.listener.accept()
                break
            except TimeoutError:
                continue
        self.listener.close()
        self.connection = self.resources.enter_context(connection)
        self.connection.settimeout(max(0.001, deadline - time.monotonic()))
        self.stream = self.resources.enter_context(self.connection.makefile("rwb"))
        send(self.stream, self.config)
        answer = self.response(event)
        self.connection.settimeout(self.timeout)
        return answer

    def request(self, payload, event):
        send(self.stream, payload)
        return self.response(event)

    def close(self):
        # Closing is cleanup, never an implicit execution retry.
        self.resources.close()
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        return self.process.returncode

    def terminate(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self.close()


class DynamicsService:
    """Keep global budgets while each trial owns a fresh simulator process."""

    def __init__(
        self, run, settings, interpreter, mode, *, budget=100, episode_budget=20,
        target_budget=8, action_budget=10000, execution_seconds=1200, startup_seconds=1200,
        fps=25, gpu=None, allow_reset=True, worker_factory=SubprocessWorker,
    ):
        if mode not in {"source", "intervention", "control"}:
            raise ValueError("Unknown dynamics experiment mode")
        self.run, self.settings = Path(run), copy.deepcopy(settings)
        self.workspace = self.run / "workspace"
        self.interpreter, self.mode = interpreter, mode
        self.budget, self.episode_budget = budget, episode_budget
        self.target_budget, self.action_budget = target_budget, action_budget
        self.execution_seconds, self.startup_seconds = execution_seconds, startup_seconds
        self.fps, self.gpu = fps, gpu
        self.allow_reset, self.worker_factory = allow_reset, worker_factory
        self.used = self.episodes = self.targets = self.actions = 0
        self.worker = self.current = None
        self.trials = []
        # Trials whose simulator failed; the cause stays unclassified.
        self.failed_trials = []
        self.latest = None
        self.last_state = {}
        self.closed, self.valid, self.reason = False, True, "active"
        (self.workspace / "observations").mkdir(parents=True, exist_ok=True)
        self.save()

    def status(self):
        return {
            "ok": True,
            "phase": "playground" if self.allow_reset else "frozen_evaluation",
            "mode": self.mode,
            "closed": self.closed,
            "reason": self.reason,
            "trial": self.current["trial"] if self.current else None,
            "kind": self.current["kind"] if self.current else None,
            "execution_budget": self.budget,
            "executions_used": self.used,
            "executions_remaining": max(0, self.budget - self.used),
            "episode_budget": self.episode_budget,
            "episodes_used": self.episodes,
            "episodes_remaining": max(0, self.episode_budget - self.episodes),
            "target_budget": self.target_budget,
            "target_episodes_used": self.targets,
            "target_episodes_remaining": max(0, self.target_budget - self.targets),
            "action_budget": self.action_budget,
            "native_steps_used": self.actions,
            "total_actions_remaining": max(0, self.action_budget - self.actions),
            "execution_timeout_sec": self.execution_seconds,
            "observation": self.latest,
            **self.last_state,
        }

    def save(self):
        write_json(self.run / "session-result.json", {
            **self.status(), "protocol": "workspace-session-v1",
            "experiment": "dynamics-repair-v4", "valid": self.valid,
            "failed_trials": self.failed_trials,
            "task_completed": bool(self.last_state.get("success", False)),
        })

    def record(self, event, **data):
        append_jsonl(self.run / "ledger.jsonl", {
            "event": event, "time": time.time(),
            "trial": self.current["trial"] if self.current else None, **data,
        })

    def reserve_episode(self, kind):
        if self.episodes >= self.episode_budget:
            raise ValueError("Global episode budget exhausted")
        if kind == "target" and self.targets >= self.target_budget:
            raise ValueError("Target episode budget exhausted")
        self.episodes += 1
        self.targets += int(kind == "target")
        self.save()

    def event(self, event):
        try:
            if event["event"] == "reserve":
                if event["resource"] == "episode":
                    self.reserve_episode(self.current["kind"])
                else:
                    if self.actions >= self.action_budget:
                        raise ValueError("Global action budget exhausted")
                    self.actions += 1
            elif event["event"] == "action_completed":
                self.actions += event["consumed"] - 1
            self.record("worker_event", detail=event)
            self.save()
            return {"ok": True}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    def _update_state(self, answer):
        self.latest = answer.get("observation", self.latest)
        self.last_state = {key: answer[key] for key in (
            "success", "terminated", "truncated", "native_action_limit",
            "native_steps_remaining",
        ) if key in answer}

    def _failure(self, exc=None):
        self.valid = False
        self.reason = "unclassified_worker_failure"
        self.failed_trials.append(self.current["trial"])
        details = {"classification": "unclassified"}
        if exc is not None:
            details.update(error_type=type(exc).__name__, error=str(exc))
        else:
            report = self.run / "trials" / self.current["trial"] / "worker-error.json"
            if report.is_file():
                diagnostic = json.loads(report.read_text(encoding="utf-8"))
                details.update(classification=diagnostic["classification"],
                               error_type=diagnostic["type"], error=diagnostic["message"])
        # Native diagnostics belong in the host ledger, never in the agent reply.
        self.record("worker_failure", **details)
        self.save()
        return {"ok": False, "fatal": True, "classification": "unclassified",
                "error": "Simulator execution failed; the original attempt was preserved."}

    def _request(self, payload=None):
        try:
            answer = (self.worker.initialize(self.event) if payload is None
                      else self.worker.request(payload, self.event))
        except (OSError, ConnectionError, subprocess.SubprocessError) as exc:
            answer = self._failure(exc)
        else:
            if answer.get("fatal"):
                self._failure()
        if answer.get("fatal"):
            self.current["failure"] = answer
            self.worker.close()
            self.worker = None
        else:
            self._update_state(answer)
        return answer

    def start(self, kind, parameters=None, hypothesis="", prediction=""):
        if self.closed:
            raise ValueError("Session is closed")
        allowed = {"source"} if self.mode == "source" else {"target", "diagnostic"}
        if not isinstance(kind, str) or kind not in allowed:
            raise ValueError(f"Trial kind {kind!r} is unavailable in this condition")
        parameters = {} if parameters is None else parameters
        if not isinstance(parameters, dict):
            raise ValueError("Diagnostic parameters must be a JSON object")
        if parameters and (kind != "diagnostic" or self.mode == "control"):
            raise PermissionError("Parameter changes are only allowed in intervention diagnostics")
        for name, value in parameters.items():
            if name not in PARAMETERS:
                raise ValueError(f"Unsupported diagnostic parameter: {name}")
            low, high = PARAMETERS[name]
            if (type(value) not in {int, float} or not math.isfinite(value)
                    or not low <= value <= high):
                raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
        if kind == "diagnostic" and (
            not isinstance(hypothesis, str) or not isinstance(prediction, str)
            or not hypothesis.strip() or not prediction.strip()
        ):
            raise ValueError("Diagnostic trials require a hypothesis and prediction")
        # Check the next grant before disturbing an existing trial.
        if self.episodes >= self.episode_budget:
            raise ValueError("Global episode budget exhausted")
        if kind == "target" and self.targets >= self.target_budget:
            raise ValueError("Target episode budget exhausted")
        self.end()
        self.reserve_episode(kind)
        trial_id = f"{len(self.trials) + 1:06d}"
        trial_dir = self.run / "trials" / trial_id
        trial_dir.mkdir(parents=True)
        self.current = {
            "trial": trial_id, "kind": kind, "parameters": parameters,
            "hypothesis": hypothesis, "prediction": prediction,
            "started_at": time.time(),
        }
        self.trials.append(self.current)
        actual = copy.deepcopy(self.settings["target" if kind == "target" else "source"])
        actual.update(parameters)
        write_json(trial_dir / "settings.json", actual)
        write_json(trial_dir / "trial.json", self.current)
        self.record("trial_started", **{key: value for key, value in self.current.items()
                                       if key != "trial"})
        self.last_state = {}
        config = {
            "run": str(self.run), "trial_dir": str(trial_dir), "settings": actual,
            "initial_observation": f"observations/trial-{trial_id}-initial",
            "allow_reset": self.allow_reset, "execution_seconds": self.execution_seconds,
            "startup_seconds": self.startup_seconds, "fps": self.fps,
        }
        try:
            self.worker = self.worker_factory(config, self.interpreter, self.gpu)
            answer = self._request()
        except (OSError, subprocess.SubprocessError) as exc:
            answer = self._failure(exc)
            self.current["failure"] = answer
        self.save()
        return {**self.status(), **answer}

    def end(self, conclusion=""):
        if self.current is None:
            return self.status()
        if self.worker is not None:
            self._request({"op": "close"})
            if self.worker is not None:
                returncode = self.worker.close()
                self.worker = None
                if returncode:
                    self.current["failure"] = self._failure(
                        RuntimeError(f"Simulator exited during cleanup with status {returncode}"))
        self.current.update(ended_at=time.time(), conclusion=conclusion,
                            final_state=dict(self.last_state))
        write_json(self.run / "trials" / self.current["trial"] / "trial.json", self.current)
        self.record("trial_ended", conclusion=conclusion, final_state=self.last_state)
        self.current = None
        self.save()
        return self.status()

    def handle(self, request):
        op = request.get("op")
        if not isinstance(op, str):
            raise ValueError("Operation must be a string")
        if op in {"status", "dynamics_status"}:
            return self.status()
        if self.closed:
            raise ValueError("Session is closed")
        if op == "finish":
            self.end()
            self.closed, self.reason = True, "agent_finished"
            self.save()
            return self.status()
        if op == "dynamics_start":
            return self.start(request["kind"], request.get("parameters"),
                              request.get("hypothesis", ""), request.get("prediction", ""))
        if op == "dynamics_end":
            return self.end(request.get("conclusion", ""))
        if op not in {"exec", "observe", "instruction", "reset"}:
            raise ValueError("Unknown session operation")
        if self.closed or self.worker is None:
            raise ValueError("Start a trial before submitting robot code")
        if op == "reset" and not self.allow_reset:
            raise PermissionError("reset() is forbidden during frozen policy evaluation")
        if self.used >= self.budget:
            raise ValueError("Execution budget exhausted")
        fixed = {"observe": "get_observation()", "instruction": "print(get_instruction())",
                 "reset": "reset()"}
        files = (request.get("files") if op == "exec"
                 else [{"path": f"{op}.py", "code": fixed[op]}])
        if not isinstance(files, list) or not files or any(
            not isinstance(item, dict) or not isinstance(item.get("code"), str)
            or not isinstance(item.get("path"), str) for item in files
        ):
            raise ValueError("Expected source files with path and code")
        code = "\n\n".join(item["code"] for item in files)
        self.used += 1
        folder = self.workspace / "observations" / f"{self.used:06d}"
        folder.mkdir()
        (folder / "code.py").write_text(code)
        write_json(folder / "sources.json", files)
        self.record("execution_started", execution=self.used,
                    sources=str(folder.relative_to(self.run) / "sources.json"))
        self.save()
        answer = self._request({"op": "exec", "code": code, "execution": self.used})
        result = {**self.status(), **answer, "execution": self.used}
        write_json(folder / "result.json", result)
        self.record("execution_completed", execution=self.used,
                    ok=result["ok"], success=result.get("success", False),
                    classification=result.get("classification"))
        self.save()
        return result

    def close(self):
        self.end()
        self.closed = True
        self.save()

    def interrupt(self):
        """Stop the child before a supervisor can kill this broker process."""
        if self.worker is not None:
            self.worker.terminate()
            self.worker = None
            self.current["failure"] = self._failure(RuntimeError("Supervisor interrupted trial"))
        self.reason = "interrupted"
        self.save()


def serve(service, socket_path, *, ready_fd=None):
    socket_path = Path(socket_path)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(socket_path))
            listener.listen(1)
            if ready_fd is not None:
                os.write(ready_fd, b'{"event":"ready"}\n')
            while not service.closed:
                connection, _ = listener.accept()
                with connection, contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    connection.settimeout(10)
                    with connection.makefile("rwb") as stream:
                        try:
                            request = receive(stream)
                            answer = service.handle(request)
                        except (
                            ValueError, PermissionError, KeyError, TimeoutError, ConnectionError,
                        ) as exc:
                            answer = {"ok": False, "error": str(exc), "status": service.status()}
                        send(stream, answer)
    finally:
        service.close()
        socket_path.unlink(missing_ok=True)
        if ready_fd is not None:
            os.close(ready_fd)
    # The exit status reports the broker itself. A failed trial is recorded in the
    # ledger, failed_trials and valid=False; the agent may then request a new trial
    # within the remaining budgets, and that failure is never cleared.
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--interpreter", required=True)
    parser.add_argument("--mode", choices=("source", "intervention", "control"), required=True)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--episode-budget", type=int, default=20)
    parser.add_argument("--target-budget", type=int, default=8)
    parser.add_argument("--action-budget", type=int, default=10000)
    parser.add_argument("--execution-seconds", type=float, default=1200)
    parser.add_argument("--startup-seconds", type=float, default=1200)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--gpu")
    parser.add_argument("--ready-fd", type=int)
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args(argv)
    service = DynamicsService(
        args.run_dir, json.loads(args.settings.read_text()), args.interpreter, args.mode,
        budget=args.budget, episode_budget=args.episode_budget, target_budget=args.target_budget,
        action_budget=args.action_budget, execution_seconds=args.execution_seconds,
        startup_seconds=args.startup_seconds, fps=args.fps, gpu=args.gpu,
        allow_reset=not args.no_reset,
    )

    def terminate(signum, frame):
        service.interrupt()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    return serve(service, args.socket, ready_fd=args.ready_fd)


if __name__ == "__main__":
    raise SystemExit(main())
