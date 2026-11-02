"""Independent dynamics diagnosis experiment; never a RoboDojo Test campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from orchestrator import launch_agent
from orchestrator.config import CODE_ROOT, REPO, load_task
from orchestrator.files import codex_token_usage, run_lock, write_json
from orchestrator.resources import gpu_lease
from orchestrator.session_client import request, source_snapshot
from orchestrator.session_runner import stop_process
from simulators.dynamics_native import NOMINAL_DYNAMIC_FRICTION, NOMINAL_STATIC_FRICTION
from simulators.dynamics_scoring import Criteria, describe_criteria

# v2: target qualification requires 0/3 complete physical failures (v1 accepted 1/3), and
# the cube material is applied before PhysX creates its collision shape.
# v3: nominal cube friction 1.0/1.0 (0.5/0.5 against the default fingers and table);
# v2 used 0.8/0.6.
# v4: the resting cube no longer drifts before contact; workspaces state the success
# criteria and the actual nominal material; 20 episodes, 8 target episodes, 10000 actions.
EXPERIMENT = "dynamics-repair-v4"
REPETITIONS = 3
CALIBRATION_SCALES = (0.75, 0.5, 0.25, 0.125)
TEMPLATE = CODE_ROOT / "tasks/DynamicsRepair/Workspace"
DEFAULT_BUDGETS = {"executions": 100, "episodes": 20, "target_episodes": 8, "actions": 10000}
# Scene-default material of the fingers, table and plate (see docs/DYNAMICS_REPAIR.md).
SURFACE_FRICTION = 0.5
NOMINAL_CUBE_MASS = 0.35


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path):
    """Resolve a small host-only configuration, with no target truth in templates."""
    supplied = read_json(path)
    unknown = set(supplied) - {
        "agent_task",
        "agent",
        "simulator_python",
        "seed",
        "target_friction_scale",
        "action_limit",
        "budgets",
        "execution_timeout_sec",
        "agent_timeout_sec",
        "startup_timeout_sec",
        "feedback_fps",
    }
    if unknown:
        raise ValueError(f"Unknown experiment settings: {sorted(unknown)}")
    cfg = load_task(supplied.get("agent_task", "RoboDojo"))
    agent = replace(cfg.agent, **supplied.get("agent", {}))
    if agent.isolation != "docker":
        raise ValueError("The research agent requires Docker isolation")
    budgets = {**DEFAULT_BUDGETS, **supplied.get("budgets", {})}
    if set(budgets) != set(DEFAULT_BUDGETS) or any(
        type(value) is not int or value <= 0 for value in budgets.values()
    ):
        raise ValueError("budgets must contain positive integer execution/episode/action limits")
    simulator_python = Path(supplied.get("simulator_python", str(cfg.python)))
    if not simulator_python.is_absolute():
        simulator_python = REPO / simulator_python
    settings = {
        "experiment": EXPERIMENT,
        "agent": asdict(agent),
        "simulator_python": str(simulator_python.absolute()),
        "seed": supplied.get("seed", 0),
        "target_friction_scale": supplied.get("target_friction_scale", 0.25),
        "action_limit": supplied.get("action_limit", 500),
        "budgets": budgets,
        "execution_timeout_sec": supplied.get("execution_timeout_sec", 1200),
        "agent_timeout_sec": supplied.get("agent_timeout_sec", agent.timeout_sec or 0),
        "startup_timeout_sec": supplied.get("startup_timeout_sec", 1200),
        "feedback_fps": supplied.get("feedback_fps", 25),
        "scene": read_json(CODE_ROOT / "tasks/DynamicsRepair/scene.json"),
        "criteria": asdict(Criteria()),
    }
    if type(settings["seed"]) is not int or settings["seed"] != 0:
        raise ValueError("The first experiment uses the fixed scene seed 0")
    scale = settings["target_friction_scale"]
    if type(scale) not in (int, float) or not 0 < scale < 1:
        raise ValueError("target_friction_scale must be a number strictly between 0 and 1")
    if type(settings["action_limit"]) is not int or settings["action_limit"] != 500:
        raise ValueError("The first experiment fixes the custom task action_limit at 500")
    for field in (
        "execution_timeout_sec", "startup_timeout_sec", "feedback_fps", "agent_timeout_sec"
    ):
        value = settings[field]
        allow_zero = field == "agent_timeout_sec"
        if (
            type(value) not in (int, float) or not math.isfinite(value)
            or (value < 0 if allow_zero else value <= 0)
        ):
            bound = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{field} must be a finite {bound} number")
    fps = settings["feedback_fps"]
    if fps != int(fps):
        raise ValueError("feedback_fps must be a whole number")
    settings["feedback_fps"] = int(fps)
    return settings


def environment_settings(config, alpha=1.0):
    source = {
        "task_name": "red_cube_to_plate",
        "seed": config["seed"],
        "action_limit": config["action_limit"],
        "friction_scale": 1.0,
        "mass_scale": 1.0,
        "restitution": 0.0,
        "scene": config["scene"],
        "criteria": config["criteria"],
    }
    return {"source": source, "target": {**source, "friction_scale": alpha}}


def start_run(run, command, config):
    """Claim a fresh directory; resuming never means repeating a physical attempt."""
    run = Path(run).absolute()
    if not re.match(r"^\d{12}(?:_|$)", run.name):
        raise ValueError("--run-dir must start with YYYYMMDDHHMM followed by '_' or end")
    datetime.strptime(run.name[:12], "%Y%m%d%H%M")
    run.mkdir(parents=True, exist_ok=True)
    if any(path.name != ".protocol.lock" for path in run.iterdir()):
        raise FileExistsError("Run directory already contains evidence; use summarize or a new run")
    write_json(run / "config.json", config)
    write_json(
        run / "result.json",
        {
            "experiment": EXPERIMENT,
            "command": command,
            "status": "running",
            "started_at": time.time(),
            "failure_classification": None,
        },
    )
    shutil.copytree(TEMPLATE, run / "templates")
    for module in ("session_client", "dynamics_client"):
        shutil.copyfile(
            CODE_ROOT / "orchestrator" / f"{module}.py", run / "templates/scripts" / f"{module}.py"
        )
    return run


class PolicySubmissionError(ValueError):
    """The submitted manifest or its declared source files cannot form a policy."""


def archive_policy(manifest, destination, *, workspace=None, version):
    """Freeze the declared complete program, not a concatenation of execution history."""
    manifest = Path(manifest)
    workspace = Path(workspace) if workspace is not None else manifest.parent
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        raw = manifest.read_bytes()
    except FileNotFoundError as exc:
        raise PolicySubmissionError("Policy manifest was not submitted") from exc
    submitted = destination / "submitted-policy.json"
    submitted.write_bytes(raw)
    submitted.chmod(0o444)
    try:
        declaration = json.loads(raw)
    except ValueError as exc:
        raise PolicySubmissionError(f"Invalid policy manifest JSON: {exc}") from exc
    if not isinstance(declaration, dict):
        raise PolicySubmissionError("Policy manifest must be a JSON object")
    entry, includes = declaration.get("entry"), declaration.get("includes", [])
    if not isinstance(entry, str) or not entry:
        raise PolicySubmissionError("Policy entry must be a nonempty string")
    if not isinstance(includes, list) or any(
        not isinstance(name, str) or not name for name in includes
    ):
        raise PolicySubmissionError("Policy includes must be a list of nonempty strings")
    try:
        files = source_snapshot(workspace, entry, includes)
    except (
        ValueError, IndexError, FileNotFoundError, NotADirectoryError, IsADirectoryError
    ) as exc:
        raise PolicySubmissionError(f"Invalid policy source: {exc}") from exc
    bundle = {
        "version": version,
        # source_snapshot resolves links; the manifest must name the frozen files.
        "entry": files[-1]["path"],
        "includes": [item["path"] for item in files[:-1]],
        "files": files,
    }
    write_json(destination / "policy.json", bundle)
    for item in files:
        path = destination / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item["code"], encoding="utf-8")
        path.chmod(0o444)
    (destination / "policy.json").chmod(0o444)
    return bundle


def _copy_public_tree(source, destination):
    """Materialize links only within the original agent-visible source tree."""
    source = Path(source)
    # Resolve the host-owned parent, not the agent-controlled root entry itself.
    allowed = source.parent.resolve() / source.name

    def copy(path, target, ancestors):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(allowed):
            raise ValueError("Agent experience link escapes its permitted source tree")
        if resolved.is_dir():
            if resolved in ancestors:
                raise ValueError("Agent experience contains a directory link cycle")
            target.mkdir(parents=True)
            for child in resolved.iterdir():
                copy(child, target / child.name, ancestors | {resolved})
            shutil.copystat(resolved, target)
        else:
            shutil.copy2(resolved, target)

    if not source.exists() and not source.is_symlink():
        return
    copy(source, Path(destination), set())


def prepare_workspace(config, run, phase, *, templates, experience=None, policy=None):
    workspace = run / "workspace"
    shutil.copytree(templates, workspace, ignore=shutil.ignore_patterns("source.md", "repair.md"))
    for name in ("observations", "scratch", "skills", "lessons", "submission"):
        (workspace / name).mkdir(exist_ok=True)
    if experience is None:
        (workspace / "source_experience").mkdir()
    else:
        _copy_public_tree(experience, workspace / "source_experience")
        for component in ("skills", "lessons"):
            inherited = experience / "source" / component
            if inherited.is_dir():
                shutil.rmtree(workspace / component)
                _copy_public_tree(inherited, workspace / component)
    if policy is not None:
        for item in policy["files"]:
            path = workspace / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(item["code"], encoding="utf-8")
        write_json(
            workspace / "submission/policy.json",
            {
                "entry": policy["entry"],
                "includes": policy["includes"],
            },
        )
    template_name = "source.md" if phase == "source" else "repair.md"
    instruction = (templates / template_name).read_text(encoding="utf-8")
    values = {
        "EXECUTION_BUDGET": config["budgets"]["executions"],
        "EPISODE_BUDGET": config["budgets"]["episodes"],
        "TARGET_EPISODE_BUDGET": config["budgets"]["target_episodes"],
        "TOTAL_ACTION_BUDGET": config["budgets"]["actions"],
        "NATIVE_ACTION_LIMIT": config["action_limit"],
        "SUCCESS_CRITERIA": describe_criteria(Criteria(**config["criteria"])),
    }
    for name, value in values.items():
        instruction = instruction.replace("{{" + name + "}}", str(value))
    instruction += (
        "\nThis branch may modify the diagnostic copy's three public parameters.\n"
        if phase == "intervention"
        else "\nThis branch cannot edit diagnostic parameters; copies use public nominal values.\n"
        if phase == "control"
        else "\nThis phase uses only the source environment.\n"
    )
    (workspace / "instruction.md").write_text(instruction, encoding="utf-8")
    write_json(workspace / "nominal.json", nominal_parameters())
    return workspace


def nominal_parameters():
    """Public source values of the manipulated cube; never the target's."""
    return {
        "static_friction": NOMINAL_STATIC_FRICTION,
        "dynamic_friction": NOMINAL_DYNAMIC_FRICTION,
        "restitution": 0.0,
        "mass_kg": NOMINAL_CUBE_MASS,
        "friction_combine_mode": "multiply",
        "contact_surface_friction": SURFACE_FRICTION,
        "effective_contact_friction": {
            "static": NOMINAL_STATIC_FRICTION * SURFACE_FRICTION,
            "dynamic": NOMINAL_DYNAMIC_FRICTION * SURFACE_FRICTION,
        },
        "parameters": {"friction_scale": 1.0, "mass_scale": 1.0, "restitution": 0.0},
    }


def _agent_command(config, run, socket_dir):
    agent = config["agent"]
    launch_agent.require_image(agent["image"])
    if agent["network"] != "none":
        launch_agent.ensure_network()
    name = f"embodiedevo-dynamics-{os.getpid()}-{time.time_ns()}"
    argv = launch_agent.docker_argv(
        run / "workspace",
        launch_agent.stage_home(socket_dir / "agent-home"),
        launch_agent.adapt_for_docker(
            launch_agent.agent_argv(
                agent["preset"],
                "Read instruction.md and work autonomously. Start trials with scripts/dynamics.py; "
                "execute control code with scripts/env.py. Publish the complete policy using "
                "submission/policy.json. Preserve failed attempts and finish when done.",
                model=agent["model"],
                effort=agent["reasoning_effort"],
            ),
            preset=agent["preset"],
        ),
        image=agent["image"],
        network=agent["network"],
        cpus=agent["cpus"],
        memory_mb=agent["memory_mb"],
        container_name=name,
    )
    mounts = [
        "--env",
        "EMBODIEDEVO_SOCKET=/run/embodiedevo/session.sock",
        "--mount",
        f"type=bind,src={socket_dir},dst=/run/embodiedevo,readonly",
    ]
    for component in (
        "instruction.md",
        "primitives",
        "scripts",
        "observations",
        "source_experience",
        "nominal.json",
    ):
        mounts += [
            "--mount",
            f"type=bind,src={run / 'workspace' / component},dst=/workspace/{component},readonly",
        ]
    index = argv.index(agent["image"])
    argv[index:index] = mounts
    return name, argv


def _await_ready(process, descriptor, timeout):
    deadline = time.monotonic() + timeout
    data = b""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Dynamics service exited before readiness; inspect host service log")
        ready, _, _ = select.select(
            [descriptor], [], [], min(0.5, max(0, deadline - time.monotonic()))
        )
        if ready:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                raise RuntimeError("Dynamics service closed its readiness pipe")
            data += chunk
            while b"\n" in data:
                line, data = data.split(b"\n", 1)
                if json.loads(line).get("event") == "ready":
                    return
    raise TimeoutError("Dynamics service startup timed out; no automatic retry")


def run_agent_phase(config, run, mode, settings):
    """One fresh container and one broker; all native trials remain serial subprocesses."""
    write_json(run / "settings.json", settings)
    started = time.monotonic()
    result = {"status": "running", "mode": mode, "agent_returncode": None}
    write_json(run / "phase-result.json", result)
    service = agent_process = None
    container = None
    with tempfile.TemporaryDirectory(prefix="embodiedevo-dynamics-") as temporary:
        socket_dir = Path(temporary)
        socket_dir.chmod(0o755)
        socket_path = socket_dir / "session.sock"
        ready_read, ready_write = os.pipe()
        try:
            budget = config["budgets"]
            command = [
                sys.executable,
                "-m",
                "orchestrator.dynamics_session",
                "--run-dir",
                str(run),
                "--socket",
                str(socket_path),
                "--settings",
                str(run / "settings.json"),
                "--interpreter",
                config["simulator_python"],
                "--mode",
                mode,
                "--budget",
                str(budget["executions"]),
                "--episode-budget",
                str(budget["episodes"]),
                "--target-budget",
                str(budget["target_episodes"]),
                "--action-budget",
                str(budget["actions"]),
                "--execution-seconds",
                str(config["execution_timeout_sec"]),
                "--startup-seconds",
                str(config["startup_timeout_sec"]),
                "--fps",
                str(config["feedback_fps"]),
                "--ready-fd",
                str(ready_write),
            ]
            if config.get("gpu") is not None:
                command += ["--gpu", str(config["gpu"])]
            with (run / "service.log").open("w") as service_log:
                service = subprocess.Popen(
                    command,
                    cwd=CODE_ROOT,
                    pass_fds=(ready_write,),
                    stdout=service_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                os.close(ready_write)
                ready_write = None
                _await_ready(service, ready_read, config["startup_timeout_sec"])
                container, argv = _agent_command(config, run, socket_dir)
                with (run / "agent.log").open("w") as agent_log:
                    agent_process = subprocess.Popen(
                        argv,
                        cwd=CODE_ROOT,
                        stdout=agent_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    agent_started = time.monotonic()
                    limit = config["agent_timeout_sec"]
                    while agent_process.poll() is None:
                        if limit and time.monotonic() - agent_started >= limit:
                            raise TimeoutError("Agent wall-clock limit reached; no automatic retry")
                        if service.poll() is not None and service.returncode != 0:
                            raise RuntimeError("Dynamics service failed; inspect host records")
                        time.sleep(0.2)
                result["agent_returncode"] = agent_process.returncode
                if service.poll() is None:
                    request({"op": "finish"}, socket_path=str(socket_path))
                    service.wait(timeout=30)
                result["service_returncode"] = service.returncode
                # Failed trials do not end the phase; they remain in its record.
                session = read_json(run / "session-result.json")
                result.update(
                    status=(
                        "complete"
                        if agent_process.returncode == service.returncode == 0
                        else "agent_failed"
                        if agent_process.returncode
                        else "incomplete"
                    ),
                    failed_trials=session["failed_trials"],
                    session_valid=session["valid"],
                )
                if service.returncode:
                    result["failure_classification"] = "unclassified"
        except (Exception, KeyboardInterrupt) as exc:
            result.update(
                status="incomplete", error=str(exc), failure_classification="unclassified"
            )
            if isinstance(exc, KeyboardInterrupt):
                raise
        finally:
            cleanup_errors = []
            if container:
                try:
                    subprocess.run(
                        ["docker", "rm", "-f", container],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    cleanup_errors.append(str(exc))
            for process in (agent_process, service):
                try:
                    stop_process(process)
                except (OSError, subprocess.SubprocessError) as exc:
                    cleanup_errors.append(str(exc))
            os.close(ready_read)
            if ready_write is not None:
                os.close(ready_write)
            agent_home = socket_dir / "agent-home"
            result["model_usage"] = codex_token_usage(agent_home)
            sessions = agent_home / ".codex/sessions"
            if sessions.is_dir():
                shutil.copytree(sessions, run / "agent_sessions")
            if cleanup_errors:
                result.update(
                    status="incomplete",
                    cleanup_errors=cleanup_errors,
                    failure_classification="unclassified",
                )
            result["elapsed_sec"] = round(time.monotonic() - started, 3)
            write_json(run / "phase-result.json", result)
    return result


def replay_batch(config, run, policy, settings, *, kind):
    """All scheduled replays remain in the denominator, including incomplete ones."""
    from orchestrator.dynamics_session import DynamicsService

    run.mkdir(parents=True, exist_ok=False)
    results = []

    def save_batch():
        summary = {
            "policy": policy["version"],
            "kind": kind,
            "attempts": REPETITIONS,
            "executed_attempts": len(results),
            "pending": REPETITIONS - len(results),
            "successes": sum(result["success"] for result in results),
            "incomplete": REPETITIONS - sum(result["status"] == "complete" for result in results),
            "results": results,
        }
        write_json(run / "result.json", summary)
        return summary

    save_batch()
    for number in range(1, REPETITIONS + 1):
        attempt = run / f"{number:03d}"
        (attempt / "workspace/observations").mkdir(parents=True)
        result = {"repeat": number, "ok": False, "success": False, "status": "running"}
        write_json(attempt / "result.json", result)
        results.append(result)
        save_batch()
        service = None
        try:
            service = DynamicsService(
                attempt,
                settings=settings,
                interpreter=config["simulator_python"],
                mode="source" if kind == "source" else "intervention",
                budget=1,
                episode_budget=1,
                target_budget=1,
                action_budget=config["action_limit"],
                execution_seconds=config["execution_timeout_sec"],
                startup_seconds=config["startup_timeout_sec"],
                fps=config["feedback_fps"],
                gpu=config.get("gpu"),
                allow_reset=False,
            )
            response = service.handle({"op": "dynamics_start", "kind": kind})
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Replay failed to start"))
            response = service.handle({"op": "exec", "files": policy["files"]})
            complete = bool(response.get("ok")) and not response.get("fatal")
            result.update(
                status="complete" if complete else "incomplete",
                ok=bool(response.get("ok")),
                success=bool(response.get("success")) and complete,
                feedback=response,
            )
            if not complete:
                result["failure_classification"] = response.get("classification") or "unclassified"
        except (Exception, KeyboardInterrupt) as exc:
            result.update(
                status="incomplete", error=str(exc), failure_classification="unclassified"
            )
            if isinstance(exc, KeyboardInterrupt):
                write_json(attempt / "result.json", result)
                raise
        finally:
            if service is not None:
                try:
                    service.close()
                    if not service.valid:
                        result.update(
                            status="incomplete",
                            success=False,
                            failure_classification="unclassified",
                        )
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    result.update(
                        status="incomplete",
                        success=False,
                        cleanup_error=str(exc),
                        failure_classification="unclassified",
                    )
            write_json(attempt / "result.json", result)
            save_batch()
    return save_batch()


@contextmanager
def record_replay(run, result, directory, destination, key):
    """Publish the saved batch even when a replay exits through an exception."""
    result["stage"] = str(directory.relative_to(run))
    write_json(run / "result.json", result)
    try:
        yield
    finally:
        record = directory / "result.json"
        if record.is_file():
            destination[key] = read_json(record)
        write_json(run / "result.json", result)


def calibrate(config, run, baseline_manifest, reference_manifest, scales=CALIBRATION_SCALES):
    # Candidates are fixed before any replay and tried from the mildest perturbation down.
    if list(scales) != sorted(set(scales), reverse=True) or not all(0 < s < 1 for s in scales):
        raise ValueError("Calibration scales must be distinct, descending and within (0, 1)")
    baseline = archive_policy(
        baseline_manifest, run / "policies/baseline", version="development_baseline"
    )
    reference = archive_policy(
        reference_manifest, run / "policies/reference", version="development_reference"
    )
    result = {
        "experiment": EXPERIMENT,
        "command": "calibrate",
        "status": "running",
        "stage": "calibration/source",
        "scales": list(scales),
        "candidates": [],
        "selected_alpha": None,
        "environment": environment_settings(config)["source"],
    }
    source_directory = run / "calibration/source"
    with record_replay(run, result, source_directory, result, "source"):
        source = result["source"] = replay_batch(
            config, source_directory, baseline, environment_settings(config), kind="source"
        )
    if source["successes"] == REPETITIONS:
        for index, alpha in enumerate(scales, 1):
            settings = environment_settings(config, alpha)
            candidate = run / "calibration" / f"candidate_{index:02d}"
            outcome = {"alpha": alpha}
            result["candidates"].append(outcome)
            with record_replay(run, result, candidate / "baseline", outcome, "baseline"):
                original = outcome["baseline"] = replay_batch(
                    config, candidate / "baseline", baseline, settings, kind="target"
                )
            with record_replay(run, result, candidate / "reference", outcome, "reference"):
                repaired = outcome["reference"] = replay_batch(
                    config, candidate / "reference", reference, settings, kind="target"
                )
            # The same thresholds as the experiment gates: the baseline fails all three
            # complete target executions, and the reference repairs all three.
            if (
                original["incomplete"] == 0
                and repaired["incomplete"] == 0
                and original["successes"] == 0
                and repaired["successes"] == REPETITIONS
            ):
                result.update(status="calibrated", selected_alpha=alpha)
                break
    if result["selected_alpha"] is None:
        result["status"] = "not_calibrated"
    result["stage"] = "finished"
    write_json(run / "result.json", result)
    return result


def freeze_experience(run, source_workspace, p0, qualification, transfer):
    """Both branches receive the same evidence, with no host configuration or truth."""
    destination = run / "experience"
    destination.mkdir()
    for component in ("skills", "lessons", "observations", "scratch"):
        _copy_public_tree(source_workspace / component, destination / "source" / component)
    write_json(destination / "P0.json", p0)
    for kind, batch in (("source_replays", qualification), ("target_replays", transfer)):
        for number in range(1, REPETITIONS + 1):
            _copy_public_tree(
                batch / f"{number:03d}" / "workspace/observations",
                destination / kind / f"{number:03d}" / "observations",
            )
        public_counts = read_json(batch / "result.json")
        write_json(
            destination / kind / "summary.json",
            {
                "attempts": public_counts["attempts"],
                "successes": public_counts["successes"],
                "incomplete": public_counts["incomplete"],
            },
        )
    return destination


def run_experiment(config, run, calibration_run=None):
    alpha = config["target_friction_scale"]
    target_configuration = {
        "source": "config",
        "friction_scale": alpha,
        "recoverability": "not_prevalidated",
    }
    if calibration_run is not None:
        calibration = read_json(Path(calibration_run) / "result.json")
        if calibration.get("experiment") != EXPERIMENT or calibration.get("status") != "calibrated":
            raise ValueError("The supplied calibration must be completed and qualifying")
        if calibration["environment"] != environment_settings(config)["source"]:
            raise ValueError(
                "Calibration and experiment must use the same frozen scene and action limit"
            )
        alpha = calibration["selected_alpha"]
        # Calibrations record their candidate list; earlier ones used the fixed grid.
        if alpha not in calibration.get("scales", CALIBRATION_SCALES):
            raise ValueError("Calibration did not select a prescribed friction scale")
        write_json(run / "calibration.json", calibration)
        target_configuration.update(
            source="calibration", friction_scale=alpha, recoverability="validated_by_reference"
        )
    settings = environment_settings(config, alpha)
    write_json(run / "settings.json", settings)
    source = run / "source"
    prepare_workspace(config, source, "source", templates=run / "templates")
    result = {
        "experiment": EXPERIMENT,
        "command": "run",
        "status": "running",
        "stage": "source",
        "target_configuration": target_configuration,
        "branches": {},
        "formal_test": False,
    }

    def replay_and_record(directory, policy, kind, destination, key):
        with record_replay(run, result, directory, destination, key):
            destination[key] = replay_batch(config, directory, policy, settings, kind=kind)

    write_json(run / "result.json", result)
    source_phase = run_agent_phase(config, source, "source", settings)
    result["source_phase"] = source_phase
    write_json(run / "result.json", result)
    if source_phase["status"] != "complete":
        result["status"] = "source_not_learned"
        write_json(run / "result.json", result)
        return result
    try:
        p0 = archive_policy(
            source / "workspace/submission/policy.json",
            run / "policies/P0",
            workspace=source / "workspace",
            version="P0",
        )
    except PolicySubmissionError as exc:
        result["status"] = "source_not_learned"
        result["policy_error"] = str(exc)
        result["failure_classification"] = "model_code_error"
        write_json(run / "result.json", result)
        return result
    source_replays = run / "qualification/source"
    target_replays = run / "qualification/target"
    replay_and_record(source_replays, p0, "source", result, "P0_source")
    if result["P0_source"]["successes"] != REPETITIONS:
        result["status"] = "source_not_learned"
        write_json(run / "result.json", result)
        return result
    replay_and_record(target_replays, p0, "target", result, "P0_target")
    if result["P0_target"]["incomplete"]:
        result["status"] = "transfer_incomplete"
        write_json(run / "result.json", result)
        return result
    # Repair starts only after three complete physical failures; 1/3, 2/3 or 3/3 stop here.
    if result["P0_target"]["successes"] != 0:
        result["status"] = "transfer_failure_not_established"
        write_json(run / "result.json", result)
        return result
    experience = freeze_experience(run, source / "workspace", p0, source_replays, target_replays)
    for mode in ("intervention", "control"):
        branch = run / mode
        prepare_workspace(
            config, branch, mode, templates=run / "templates", experience=experience, policy=p0
        )
        outcome = {
            "status": "running",
            "diagnosis": {"assessment": "not_reviewed", "agent_report": None},
        }
        result["branches"][mode] = outcome
        result["stage"] = mode
        write_json(run / "result.json", result)
        phase_result = run_agent_phase(config, branch, mode, settings)
        outcome["phase"] = phase_result
        diagnosis = branch / "workspace/lessons/diagnosis.json"
        if diagnosis.is_file():
            try:
                outcome["diagnosis"]["agent_report"] = read_json(diagnosis)
            except ValueError as exc:
                outcome["diagnosis"]["report_error"] = str(exc)
        if phase_result["status"] != "complete":
            outcome["status"] = "repair_failed"
            write_json(run / "result.json", result)
            continue
        outcome["status"] = "evaluating"
        write_json(run / "result.json", result)
        try:
            p1 = archive_policy(
                branch / "workspace/submission/policy.json",
                run / "policies" / mode,
                workspace=branch / "workspace",
                version="P1",
            )
        except PolicySubmissionError as exc:
            outcome["status"] = "repair_failed"
            outcome["policy_error"] = str(exc)
            outcome["failure_classification"] = "model_code_error"
        else:
            evaluation = run / "evaluation" / mode
            replay_and_record(evaluation / "target", p1, "target", outcome, "target")
            replay_and_record(evaluation / "source", p1, "source", outcome, "source")
            outcome["status"] = (
                "repair_succeeded" if outcome["target"]["successes"] == REPETITIONS
                else "repair_failed"
            )
        write_json(run / "result.json", result)
    result["status"] = result["branches"]["intervention"]["status"]
    result["stage"] = "finished"
    write_json(run / "result.json", result)
    return result


def summarize(run):
    run = Path(run)
    result = read_json(run / "result.json")
    summary = {"experiment": result["experiment"], "status": result["status"]}

    def fields(record, names):
        return {name: record[name] for name in names if name in record}

    failure_fields = ("failure_classification", "error", "policy_error", "cleanup_errors")
    phase_fields = (
        "status", "agent_returncode", "service_returncode", "failed_trials", "session_valid",
        *failure_fields,
    )
    summary.update(fields(result, ("stage", "target_configuration", *failure_fields)))
    if "source_phase" in result:
        summary["source_phase"] = fields(result["source_phase"], phase_fields)

    def counts(batch):
        entry = fields(
            batch, ("successes", "attempts", "incomplete", "executed_attempts", "pending")
        )
        entry["failed_attempts"] = []
        for attempt in batch.get("results", []):
            if attempt["status"] == "complete" and attempt["success"]:
                continue
            failure = fields(attempt, (
                "repeat", "status", "ok", "success", "failure_classification", "error",
                "cleanup_error",
            ))
            if "feedback" in attempt:
                failure["feedback"] = fields(
                    attempt["feedback"], ("classification", "stderr", "error")
                )
            entry["failed_attempts"].append(failure)
        return entry

    if result.get("command") == "calibrate":
        summary.update(
            selected_alpha=result.get("selected_alpha"),
            candidates=len(result.get("candidates", [])),
        )
        if "source" in result:
            summary["source"] = counts(result["source"])
        summary["candidate_results"] = [
            {"alpha": candidate["alpha"], **{
                key: counts(candidate[key]) for key in ("baseline", "reference") if key in candidate
            }}
            for candidate in result.get("candidates", [])
        ]
    for key in ("P0_source", "P0_target"):
        if key in result:
            summary[key] = counts(result[key])
    summary["branches"] = {}
    reviews = (
        read_json(run / "diagnosis-review.json") if (run / "diagnosis-review.json").exists() else {}
    )
    for name, branch in result.get("branches", {}).items():
        entry = {"status": branch["status"], "diagnosis": branch["diagnosis"]["assessment"]}
        entry.update(fields(branch, failure_fields))
        entry.update(fields(branch["diagnosis"], ("agent_report", "report_error")))
        if "phase" in branch:
            entry["phase"] = fields(branch["phase"], phase_fields)
        if name in reviews:
            review = reviews[name]
            if review["assessment"] not in {"supported", "ambiguous", "unsupported"}:
                raise ValueError("Diagnosis review must be supported, ambiguous or unsupported")
            entry["diagnosis"] = review["assessment"]
            entry["host_review"] = review
        for kind in ("source", "target"):
            if kind in branch:
                entry[kind] = counts(branch[kind])
        summary["branches"][name] = entry
    summary["resources"] = {}
    for phase in ("source", *result.get("branches", {})):
        phase_owner = summary if phase == "source" else summary["branches"][phase]
        phase_key = "source_phase" if phase == "source" else "phase"
        phase_record = run / phase / "phase-result.json"
        budget_record = run / phase / "session-result.json"
        if phase_record.exists():
            record = read_json(phase_record)
            phase_owner.setdefault(phase_key, {}).update(fields(record, phase_fields))
            summary["resources"][phase] = {
                "elapsed_sec": record.get("elapsed_sec"),
                "model_usage": record.get("model_usage"),
            }
        ledger = run / phase / "ledger.jsonl"
        if ledger.is_file():
            phase_summary = phase_owner.setdefault(phase_key, {})
            phase_summary.update(execution_failures=[], worker_failures=[])
            with ledger.open(encoding="utf-8") as stream:
                for line in stream:
                    event = json.loads(line)
                    if event["event"] == "execution_completed" and not event["ok"]:
                        phase_summary["execution_failures"].append(fields(event, (
                            "event", "trial", "execution", "classification", "ok", "success",
                        )))
                    elif event["event"] == "worker_failure":
                        phase_summary["worker_failures"].append(fields(event, (
                            "event", "trial", "classification", "error_type", "error",
                        )))
        if budget_record.exists():
            record = read_json(budget_record)
            summary["resources"].setdefault(phase, {})["budgets"] = {
                key: record[key]
                for key in (
                    "execution_budget",
                    "executions_used",
                    "episode_budget",
                    "episodes_used",
                    "target_budget",
                    "target_episodes_used",
                    "action_budget",
                    "native_steps_used",
                )
            }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("calibrate", "run", "summarize"):
        child = sub.add_parser(command)
        child.add_argument("--run-dir", type=Path, required=True)
        if command != "summarize":
            child.add_argument("--config", type=Path, required=True)
            child.add_argument("--gpu", default=None)
        if command == "calibrate":
            child.add_argument("--baseline-policy", type=Path, required=True)
            child.add_argument("--reference-policy", type=Path, required=True)
            child.add_argument(
                "--scales", type=float, nargs="+", default=list(CALIBRATION_SCALES),
                help="Descending candidate scales, fixed before calibration and recorded",
            )
        if command == "run":
            child.add_argument(
                "--calibration", type=Path,
                help="Optional development calibration; otherwise use the configured scale",
            )
    args = parser.parse_args(argv)
    try:
        if args.command == "summarize":
            print(json.dumps(summarize(args.run_dir), ensure_ascii=False, indent=2))
            return 0
        config = load_config(args.config)
        config["gpu"] = args.gpu
        with run_lock(args.run_dir), gpu_lease(args.gpu):
            run = start_run(args.run_dir, args.command, config)
            try:
                if args.command == "calibrate":
                    result = calibrate(
                        config, run, args.baseline_policy, args.reference_policy, args.scales
                    )
                else:
                    result = run_experiment(config, run, args.calibration)
            except (Exception, KeyboardInterrupt) as exc:
                result = read_json(run / "result.json")
                result.update(
                    status="incomplete", error=str(exc), failure_classification="unclassified"
                )
                for outcome in result.get("branches", {}).values():
                    if outcome["status"] in {"running", "evaluating"}:
                        outcome["status"] = "incomplete"
                write_json(run / "result.json", result)
                raise
        print(json.dumps(summarize(args.run_dir), ensure_ascii=False, indent=2))
        return 0 if result["status"] in {"calibrated", "repair_succeeded"} else 1
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
