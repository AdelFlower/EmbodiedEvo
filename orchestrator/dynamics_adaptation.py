"""Source-simulator calibration and policy adaptation."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np

from integrations.dynamics_rl.data import scene_plan
from orchestrator.adaptation_jobs import run_job
from orchestrator.adaptation_export import export_rollout
from orchestrator.adaptation_service import AdaptationService
from orchestrator import launch_agent
from orchestrator.config import load_task
from orchestrator.dynamics_repair import nominal_parameters
from orchestrator.dynamics_rl import ROOT, run_episode, simulator
from orchestrator.files import codex_token_usage, run_lock, write_json
from orchestrator.resources import gpu_lease
from orchestrator.session_runner import stop_process
from simulators.dynamics_scoring import Criteria, describe_criteria


NUMBER = (int, float)
PARAMETERS = dict.fromkeys(("friction_scale", "mass_scale", "restitution"), NUMBER)
CONFIG_TYPES = {
    "agent_task": str, "simulator_python": str, "seed": int, "position_radius": NUMBER,
    "p0": {"policy": str, "training": str, "source_evaluation": str},
    "source_parameters": PARAMETERS, "target_parameters": PARAMETERS,
    "parameter_ranges": {key: [NUMBER, NUMBER] for key in PARAMETERS},
    "development_layouts": int, "evaluation_layouts": int, "required_successes": int,
    "budgets": {"wall_hours": NUMBER, "rounds": int, "actions_per_round": int,
                "actions_total": int, "target_episodes": int, "diagnostic_episodes": int},
    "training": {
        "actor_learning_rate": NUMBER, "value_learning_rate": NUMBER,
        "exploration_log_std": NUMBER, "n_steps": int, "batch_size": int, "n_epochs": int,
        "gamma": NUMBER, "gae_lambda": NUMBER, "clip_range": NUMBER, "target_kl": NUMBER,
        "max_update_kl": NUMBER, "max_grad_norm": NUMBER, "value_warmup_epochs": int,
        "checkpoint_interval": int, "learner_threads": int,
    },
    "trainability_run": (str, type(None)), "startup_timeout": NUMBER, "step_timeout": NUMBER,
}


def load_config(path):
    """Load the complete §9 configuration, rejecting unknown keys and wrong types."""
    def check(value, schema):
        if isinstance(schema, dict):
            if type(value) is not dict or value.keys() != schema.keys():
                raise ValueError("Configuration keys differ from the specification")
            for key in schema:
                check(value[key], schema[key])
        elif isinstance(schema, list):
            if type(value) is not list or len(value) != len(schema):
                raise ValueError("Parameter ranges require two numbers")
            for item, item_schema in zip(value, schema, strict=True):
                check(item, item_schema)
        elif type(value) not in (schema if isinstance(schema, tuple) else (schema,)):
            raise ValueError("Configuration value has the wrong type")

    config = json.loads(Path(path).read_text())
    check(config, CONFIG_TYPES)
    steps = config["training"]["n_steps"]
    if (steps <= 0 or config["training"]["checkpoint_interval"] % steps
            or config["budgets"]["actions_per_round"] % steps):
        raise ValueError("Checkpoint and round action counts must be multiples of n_steps")
    return config


def new_run(path, config, command="preflight", *, agent_kind="model"):
    """Create fresh command evidence with a source snapshot."""
    path = Path(path).resolve()
    if not re.match(r"^\d{12}(?:_|$)", path.name):
        raise ValueError("--run-dir must start with YYYYMMDDHHMM")
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "config.json", config)
    write_json(path / "run.json", {
        "experiment": "dynamics-adaptation-v1", "command": command,
        "started_at": time.time(), "agent_kind": agent_kind,
    })
    sources = [
        *list((ROOT / "integrations/dynamics_rl").glob("*.py")),
        ROOT / "orchestrator/dynamics_adaptation.py", ROOT / "orchestrator/dynamics_rl.py",
        ROOT / "orchestrator/adaptation_jobs.py",
        ROOT / "orchestrator/adaptation_export.py", ROOT / "orchestrator/adaptation_service.py",
        *list((ROOT / "tasks/DynamicsAdaptation").rglob("*.py")),
        ROOT / "tasks/DynamicsAdaptation/Workspace/instruction.md",
        ROOT / "orchestrator/dynamics_rl_worker.py", ROOT / "orchestrator/dynamics_repair.py",
        *list((ROOT / "simulators").glob("dynamics_*.py")), ROOT / "requirements-rl.txt",
    ]
    for source in sources:
        destination = path / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return path


def live_parameters_match(snapshot, parameters):
    """Compare every live PhysX shape material and the cube mass for P1."""
    nominal = nominal_parameters()
    expected = np.asarray([
        nominal["static_friction"] * parameters["friction_scale"],
        nominal["dynamic_friction"] * parameters["friction_scale"],
        parameters["restitution"],
    ], dtype=np.float32)
    materials = snapshot["cube_physx_shape_materials"]
    return bool(materials and np.all(np.asarray(materials) == expected)
                and abs(snapshot["cube_mass"] - nominal["mass_kg"] * parameters["mass_scale"])
                <= 1e-4)


def compare_trajectories(left, right, tolerance):
    """Compare entire cube trajectories, including episode length, for P2/P3."""
    difference = float(np.max(np.abs(left - right))) if len(left) == len(right) else None
    return {"maximum_position_difference": difference,
            "passed": difference is not None and difference <= tolerance}


def preflight(run, config, gpu):
    """Run live-parameter, render/state parity and fresh-process identity checks."""
    import torch

    from integrations.dynamics_rl.learning import load_policy

    result = {"passed": False, "P1": [], "P2": [], "P3": None}
    try:
        torch.set_num_threads(config["training"]["learner_threads"])
        frozen = run / "policy.pt"
        shutil.copyfile(config["p0"]["policy"], frozen)
        policy = load_policy(frozen)
        layouts = scene_plan("adaptation_dev", config["development_layouts"], config["seed"],
                             config["position_radius"])
        write_json(run / "layouts.json", layouts)
        for label, parameters in (
            ("A", nominal_parameters()["parameters"]),
            ("B", {"friction_scale": 0.5, "mass_scale": 2.0, "restitution": 0.5}),
        ):
            episodes, traces = {}, {}
            for mode in ("state", "render"):
                with simulator(run, f"{label}_{mode}", config, gpu, mode,
                               parameters=parameters) as client:
                    for item in layouts[:3]:
                        client.reset(item["scene"], client.run_dir / f"reset_{item['index']}")
                        snapshot = client.inspect()["snapshot"]["current_state"]
                        result["P1"].append({
                            "parameters": parameters, "mode": mode, "layout": item["index"],
                            "cube_physx_shape_materials": snapshot["cube_physx_shape_materials"],
                            "cube_mass": snapshot["cube_mass"],
                            "passed": live_parameters_match(snapshot, parameters),
                        })
                    episodes[mode] = run_episode(client, layouts[0]["scene"], "parity", policy=policy)
                    traces[mode] = np.load(Path(episodes[mode]["episode_dir"]) / "cube-trajectory.npy")
            comparison = compare_trajectories(traces["state"], traces["render"], 1e-4)
            same_success = episodes["state"]["success"] == episodes["render"]["success"]
            result["P2"].append({**comparison, "parameters": parameters,
                                 "same_success": same_success,
                                 "passed": comparison["passed"] and same_success})
        traces = []
        for index in range(2):
            with simulator(run, f"T_{index}", config, gpu,
                           parameters=config["target_parameters"]) as client:
                episode = run_episode(client, layouts[0]["scene"], "identity", policy=policy)
                traces.append(np.load(Path(episode["episode_dir"]) / "cube-trajectory.npy"))
        result["P3"] = compare_trajectories(*traces, 1e-6)
        result["passed"] = (all(row["passed"] for row in result["P1"] + result["P2"])
                            and result["P3"]["passed"])
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
        raise
    finally:
        write_json(run / "preflight.json", result)
    return result


def select(candidates):
    """Select the most development successes, breaking ties by creation order."""
    return max(enumerate(candidates), key=lambda c: (c[1][1], -c[0]))[1][0]


def selection(networks):
    """Include P0 and deployed networks only, retaining creation order."""
    candidates = [(row["id"], row["target_dev_successes"]) for row in networks
                  if row["target_dev_successes"] is not None]
    return {"candidates": candidates, "winner": select(candidates)}


def load_parameters(path, config):
    """Validate the CLI parameter file at its trust boundary."""
    parameters = json.loads(Path(path).read_text())
    if type(parameters) is not dict or parameters.keys() != PARAMETERS.keys():
        raise ValueError("--parameters requires all three parameter keys")
    for key, value in parameters.items():
        low, high = config["parameter_ranges"][key]
        if type(value) not in NUMBER or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid parameter: {key}")
    return parameters


def trainability(run, config, gpu, parameters):
    """Train an oracle chain and gate its selected network on independent layouts."""
    started = time.time()
    freeze(run, config)
    p0 = run / "networks/N000"
    layouts = json.loads((run / "layouts.json").read_text())
    common = {key: config[key] for key in
              ("simulator_python", "startup_timeout", "step_timeout")}
    common.update(gpu=gpu, parameters=parameters)

    def rollout(directory, network, plan, fresh=False):
        return run_job(directory, {
            **common, "kind": "rollout", "network": str(network / "policy.pt"),
            "layouts": plan, "fresh_simulator_per_episode": fresh,
        })

    baseline = rollout(run / "baseline/target", p0, layouts)
    networks = [{"id": "N000", "target_dev_successes": baseline["successes"]}]
    parent, rounds = p0, []
    for round_index in range(1, config["budgets"]["rounds"] + 1):
        if networks[-1]["target_dev_successes"] == len(layouts):
            break
        network = run / "networks" / f"N{round_index:03d}"
        directory = run / "rounds" / f"R{round_index}"
        trained = run_job(directory / "train", {
            **common, "kind": "train", "parent": str(parent / "training.zip"),
            "round": round_index, "actions": config["budgets"]["actions_per_round"],
            "seed": config["seed"], "training": config["training"], "output": str(network),
        })
        deployed = rollout(directory / "deploy", network, layouts)
        networks.append({"id": network.name, "target_dev_successes": deployed["successes"]})
        rounds.append({"round": round_index, "network": network.name,
                       "training": trained, "deployment": deployed})
        parent = network
    selected = selection(networks)
    write_json(run / "selection.json", selected)
    network = run / "networks" / selected["winner"]
    evaluation_layouts = scene_plan("trainability_eval", config["evaluation_layouts"],
                                    config["seed"], config["position_radius"])
    evaluation = rollout(run / "evaluation", network, evaluation_layouts, fresh=True)
    result = {
        "parameters": parameters, "selected_network": network.name,
        "selected_network_dir": str(network), "baseline": baseline, "rounds": rounds,
        "evaluation": evaluation, "successes": evaluation["successes"],
        "incomplete": sum(not row["complete"] for row in evaluation["episodes"]),
        "passed": evaluation["successes"] >= config["required_successes"],
        "elapsed_seconds": time.time() - started,
    }
    write_json(run / "trainability.json", result)
    return result


STAGES = ("freeze", "baseline", "calibration", "control")


def requested_stages(run, stage):
    """Validate an explicit stage request without reopening any attempted stage."""
    if not re.match(r"^\d{12}(?:_|$)", run.name):
        raise ValueError("--run-dir must start with YYYYMMDDHHMM")
    if stage is None:
        if run.exists():
            raise FileExistsError(run)
        return STAGES
    if stage not in STAGES or not run.is_dir():
        raise ValueError("--stage requires an existing run and a named stage")
    if (run / stage).exists():
        raise FileExistsError(run / stage)
    for earlier in STAGES[:STAGES.index(stage)]:
        path = run / earlier / "result.json"
        if not path.is_file() or json.loads(path.read_text())["status"] != "complete":
            raise ValueError("An earlier stage has not completed")
    return (stage,)


def freeze(run, config):
    """Copy P0, convert its learner to PPO and persist the shared development layouts."""
    from integrations.dynamics_rl.adaptation import convert_p0
    from integrations.dynamics_rl.learning import load_policy

    network = run / "networks/N000"
    network.mkdir(parents=True)
    for key, name in (("policy", "policy.pt"), ("training", "sac-training.zip")):
        shutil.copyfile(config["p0"][key], network / name)
    convert_p0(network / "sac-training.zip", config["training"], config["seed"], network / "training.zip")
    policy = load_policy(network / "policy.pt")
    write_json(network / "network.json", {
        "id": "N000", "parent": None, "round": 0, "parameters": config["source_parameters"],
        "round_actions": 0, "online_steps": policy.online_steps,
        "policy_updates": 0, "created_at": time.time(),
    })
    write_json(run / "layouts.json", scene_plan("adaptation_dev", config["development_layouts"],
                                               config["seed"], config["position_radius"]))
    return {"status": "complete"}


def baseline(run, config, gpu):
    """Run P0 on the same development layouts in source and target."""
    layouts = json.loads((run / "layouts.json").read_text())
    results = {}
    for label in ("source", "target"):
        results[label] = run_job(run / "baseline" / label, {
            **{key: config[key] for key in ("simulator_python", "startup_timeout", "step_timeout")},
            "gpu": gpu, "kind": "rollout", "network": str(run / "networks/N000/policy.pt"),
            "parameters": config[f"{label}_parameters"], "layouts": layouts,
            "fresh_simulator_per_episode": False,
        })
    return {"status": "complete", **results}


def prepare_workspace(run, config):
    """Render the public template and export the shared baseline."""
    workspace = run / "calibration/workspace"
    shutil.copytree(run / "source/tasks/DynamicsAdaptation/Workspace", workspace)
    for name in ("notes", "submission", "observations"):
        (workspace / name).mkdir()
    instruction = (workspace / "instruction.md").read_text()
    values = {**config["budgets"], "development_layouts": config["development_layouts"],
              "success_criteria": describe_criteria(Criteria()), "n_steps": config["training"]["n_steps"]}
    for key, value in values.items():
        instruction = instruction.replace("{{" + key.upper() + "}}", str(value))
    (workspace / "instruction.md").write_text(instruction)
    write_json(workspace / "nominal.json", {**nominal_parameters(), "ranges": config["parameter_ranges"]})
    for label in ("source", "target"):
        result = json.loads((run / "baseline" / label / "result.json").read_text())
        export_rollout(result, workspace / "observations/baseline" / label)
    return workspace


def agent_command(config, workspace, socket_dir):
    """Use the thin agent image and mount only public experiment data."""
    agent = config["agent"]
    launch_agent.require_image(agent["image"])
    if agent["network"] != "none":
        launch_agent.ensure_network()
    name = f"embodiedevo-adaptation-{os.getpid()}-{time.time_ns()}"
    argv = launch_agent.docker_argv(
        workspace, launch_agent.stage_home(socket_dir.parent / "agent-home"),
        launch_agent.adapt_for_docker(launch_agent.agent_argv(
            agent["preset"], "Read instruction.md and work autonomously. Use scripts/adapt.py for all experiments. "
            "Write analysis code under notes/. Finish by submitting submission/diagnosis.json with scripts/adapt.py submit.",
            model=agent["model"], effort=agent["reasoning_effort"]), preset=agent["preset"]),
        image=agent["image"], network=agent["network"], cpus=agent["cpus"],
        memory_mb=agent["memory_mb"], container_name=name)
    mounts = ["--env", "EMBODIEDEVO_SOCKET=/run/embodiedevo/session.sock", "--mount",
              f"type=bind,src={socket_dir},dst=/run/embodiedevo,readonly"]
    for component in ("instruction.md", "nominal.json", "scripts", "observations"):
        mounts += ["--mount", f"type=bind,src={workspace / component},dst=/workspace/{component},readonly"]
    argv[argv.index(agent["image"]):argv.index(agent["image"])] = mounts
    return name, argv


def calibration(run, config, gpu, scripted_agent):
    """Own the agent deadline, service thread, job teardown and host selection."""
    branch = run / "calibration"
    workspace = prepare_workspace(run, config)
    layouts = json.loads((run / "layouts.json").read_text())
    with ExitStack() as stack:
        temporary = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="adaptation-")))
        temporary.chmod(0o755)
        socket_dir = temporary / "socket"
        socket_dir.mkdir(mode=0o755)
        listener = stack.enter_context(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))
        address = socket_dir / "session.sock"
        listener.bind(str(address))
        listener.listen(1)
        address.chmod(0o666)
        service = AdaptationService(branch, {**config, "gpu": gpu}, layouts, float("inf"))
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        future = executor.submit(service.serve, listener)
        stack.callback(service.stop, "stopped_by_agent_exit")
        container, command = (None, [sys.executable, str(scripted_agent)]) if scripted_agent else agent_command(config, workspace, socket_dir)
        started = time.time()
        service.deadline = started + config["budgets"]["wall_hours"] * 3600
        write_json(branch / "agent-start.json", {"started_at": started, "deadline": service.deadline})
        log = stack.enter_context((branch / "agent.log").open("x"))
        process = subprocess.Popen(command, cwd=workspace, env={**os.environ, "EMBODIEDEVO_SOCKET": str(address)},
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        agent_resources = stack.enter_context(ExitStack())
        agent_resources.callback(stop_process, process)
        if container:
            agent_resources.callback(subprocess.run, ["docker", "rm", "-f", container], check=False,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        while True:
            if future.done():
                future.result()
                raise RuntimeError("Adaptation service ended before the agent")
            service.poll()
            if service.submitted:
                reason = "submitted"
                break
            if time.time() >= service.deadline:
                reason = "deadline"
                break
            if process.poll() is not None:
                reason = "agent_exited_without_submission"
                break
            time.sleep(0.1)
        if reason == "submitted":
            until = min(time.time() + 120, service.deadline)
            while process.poll() is None and time.time() < until:
                time.sleep(0.1)
        service.stop("stopped_by_deadline" if reason == "deadline" else "stopped_by_agent_exit")
        future.result()
        agent_resources.close()
        home = temporary / "agent-home"
        sessions = home / ".codex/sessions"
        if sessions.is_dir():
            shutil.copytree(sessions, branch / "agent_sessions")
        write_json(branch / "selection.json", selection(service.networks))
        return {"status": "complete", "end_reason": reason, "budgets": service.used,
                "jobs": service.jobs, "networks": service.networks, "started_at": started,
                "deadline": service.deadline, "elapsed_seconds": time.time() - started,
                "agent_returncode": process.returncode, "model_usage": codex_token_usage(home)}


def control(run, config, gpu):
    """Repeat completed calibration training at nominal dynamics on an independent chain."""
    branch = run / "control"
    layouts = json.loads((run / "layouts.json").read_text())
    calibration_result = json.loads((run / "calibration/result.json").read_text())
    baseline_result = json.loads((run / "baseline/target/result.json").read_text())
    networks = [{"id": "N000", "target_dev_successes": baseline_result["successes"]}]
    parent, rounds, skipped = run / "networks/N000", [], []
    for job in calibration_result["jobs"]:
        if job["kind"] != "train":
            continue
        record = read_record(run / "calibration/jobs" / job["id"] / "train/result.json")
        spec = job["spec"]
        if record is None or record["status"] != "complete":
            skipped.append({"job": job["id"], "round": spec["round"], "status": job["status"]})
            continue
        network = branch / "networks" / f"N{spec['round']:03d}"
        directory = branch / "rounds" / f"R{spec['round']}"
        trained = run_job(directory / "train", {
            **spec, "gpu": gpu, "parameters": nominal_parameters()["parameters"],
            "parent": str(parent / "training.zip"), "output": str(network),
        })
        deployed = None
        if (run / "calibration/jobs" / job["id"] / "deploy").exists():
            deployed = run_job(directory / "deploy", {
                **{key: config[key] for key in ("simulator_python", "startup_timeout", "step_timeout")},
                "gpu": gpu, "kind": "rollout", "network": str(network / "policy.pt"),
                "parameters": config["target_parameters"], "layouts": layouts,
                "fresh_simulator_per_episode": False,
            })
        networks.append({"id": network.name, "target_dev_successes": deployed["successes"] if deployed else None})
        rounds.append({"round": spec["round"], "calibration_job": job["id"],
                       "network": network.name, "training": trained, "deployment": deployed})
        parent = network
    write_json(branch / "selection.json", selection(networks))
    return {"status": "complete", "rounds": rounds, "skipped": skipped}


def read_record(path):
    """Return absent optional or unfinished evidence as null; propagate malformed records."""
    return json.loads(path.read_text()) if path.exists() else None


def evaluate(run, config, gpu, oracle=None):
    """Evaluate selected policies once each, retaining the entire planned denominator."""
    for branch in ("calibration", "control"):
        result = read_record(run / branch / "result.json")
        if result is None or result["status"] != "complete":
            raise ValueError("Evaluation requires completed calibration and control")
    policies = {"P0": run / "networks/N000"}
    for branch in ("calibration", "control"):
        winner = json.loads((run / branch / "selection.json").read_text())["winner"]
        policies[branch] = policies["P0"] if winner == "N000" else run / branch / "networks" / winner
    if oracle is not None:
        result = json.loads((oracle / "trainability.json").read_text())
        policies["oracle"] = (policies["P0"] if result["selected_network"] == "N000" else
                              Path(result["selected_network_dir"]))
    directory = run / "evaluation"
    directory.mkdir()
    started = time.time()
    layouts = scene_plan("adaptation_eval", config["evaluation_layouts"], config["seed"],
                         config["position_radius"])
    owners = {}
    for arm, network in policies.items():
        owners.setdefault(network, arm)
    try:
        for network, arm in owners.items():
            run_job(directory / arm, {
                **{key: config[key] for key in ("simulator_python", "startup_timeout", "step_timeout")},
                "gpu": gpu, "kind": "rollout", "network": str(network / "policy.pt"),
                "parameters": config["target_parameters"], "layouts": layouts,
                "fresh_simulator_per_episode": True,
            })
    finally:
        arms = {}
        for arm, network in policies.items():
            owner = owners[network]
            result = read_record(directory / owner / "result.json")
            if result is None:
                result = {"status": "not_started", "successes": 0, "elapsed_seconds": None,
                          "episodes": [{"layout": row["index"], "success": False, "complete": False,
                                        "not_started": True, "actions": 0} for row in layouts]}
            arms[arm] = {**result, "network": str(network), "evaluation_arm": owner,
                         "incomplete": sum(not row["complete"] for row in result["episodes"])}
        results = {"arms": arms, "accepted": arms["calibration"]["successes"] >= config["required_successes"],
                   "comparison_arms": ["P0", "calibration", "control"],
                   "elapsed_seconds": time.time() - started}
        write_json(directory / "results.json", results)
    return results


def summarize(run):
    """Report performance, costs and independent diagnosis review from retained evidence."""
    config = json.loads((run / "config.json").read_text())
    evaluation = read_record(run / "evaluation/results.json")
    stages = {stage: read_record(run / stage / "result.json") for stage in STAGES}
    costs, rounds = {}, {"calibration": [], "control": []}
    roots = {"baseline": run / "baseline", "calibration": run / "calibration/jobs",
             "control": run / "control/rounds", "evaluation": run / "evaluation"}
    for stage, root in roots.items():
        cost = {"gpu_job_seconds": {"train": 0, "rollout": 0}, "actions": 0, "updates": 0,
                "completed_training_actions": 0, "target_episodes": 0, "diagnostic_episodes": 0,
                "unrecorded_jobs": [], "unfinished_training_jobs": []}
        for path in sorted(root.rglob("spec.json")):
            spec = json.loads(path.read_text())
            result = read_record(path.parent / "result.json")
            job = str(path.parent.relative_to(run))
            if result is None:
                cost["unrecorded_jobs"].append(job)
            else:
                cost["gpu_job_seconds"][spec["kind"]] += result["elapsed_seconds"]
            if spec["kind"] == "train":
                cost["actions"] += spec["actions"]
                complete = result is not None and result["status"] == "complete"
                if complete:
                    parent = json.loads((Path(spec["parent"]).parent / "network.json").read_text())
                    cost["updates"] += result["policy_updates"] - parent["policy_updates"]
                    cost["completed_training_actions"] += result["round_actions"]
                else:
                    cost["unfinished_training_jobs"].append(job)
                deployment = read_record(path.parent.parent / "deploy/result.json")
                rounds[stage].append({"round": spec["round"], "parameters": spec["parameters"],
                    "actions": spec["actions"], "status": result["status"] if result else "unrecorded",
                    "online_successes": result["online_successes"] if complete else None,
                    "target_dev_successes": deployment["successes"] if deployment else None})
            elif stage == "calibration" and path.parent.name == "episodes":
                cost["diagnostic_episodes"] += len(spec["layouts"])
            elif stage != "baseline" or path.parent.name == "target":
                cost["target_episodes"] += len(spec["layouts"])
        costs[stage] = cost
    for rows in rounds.values():
        rows.sort(key=lambda row: row["round"])
    review = read_record(run / "diagnosis-review.json")
    trainability = config["trainability_run"]
    summary = {
        "performance": evaluation, "rounds": rounds,
        "costs": {"by_stage": costs,
                  "wall_seconds": {**{key: value["elapsed_seconds"] if value else None
                                      for key, value in stages.items()},
                                   "evaluation": evaluation["elapsed_seconds"] if evaluation else None},
                  "model_usage": stages["calibration"]["model_usage"] if stages["calibration"] else None},
        "diagnosis": {"agent_report": read_record(run / "calibration/diagnosis.json"),
                      "host_review": review["calibration"] if review and "calibration" in review else
                                     {"assessment": "not_reviewed"}},
        "p0_source_evaluation": config["p0"]["source_evaluation"],
        "trainability": {"run": trainability, "result": json.loads(
            (Path(trainability) / "trainability.json").read_text())} if trainability else None,
    }
    write_json(run / "summary.json", summary)
    print("Task performance restored:", evaluation["accepted"] if evaluation else "not_evaluated")
    print("Arm\tSuccesses/planned\tIncomplete\tEvaluation arm")
    if evaluation:
        for arm, row in evaluation["arms"].items():
            print(f"{arm}\t{row['successes']}/{len(row['episodes'])}\t{row['incomplete']}\t{row['evaluation_arm']}")
        print("Comparison arms:", ", ".join(evaluation["comparison_arms"]))
    print("Branch\tRound\tParameters\tActions\tOnline successes\tTarget dev successes\tStatus")
    for branch, rows in rounds.items():
        for row in rows:
            print(branch, *(row[key] for key in ("round", "parameters", "actions", "online_successes",
                                                "target_dev_successes", "status")), sep="\t")
    print("Costs:", json.dumps(summary["costs"]))
    print("Diagnosis supported (host review):", summary["diagnosis"]["host_review"]["assessment"])
    print("Agent diagnosis (self-report):", json.dumps(summary["diagnosis"]["agent_report"]))
    print("P0 source evaluation:", summary["p0_source_evaluation"])
    print("Trainability:", json.dumps(summary["trainability"]))
    return summary


def run_stages(run, config, gpu, stages, scripted_agent=None):
    for stage in stages:
        (run / stage).mkdir()
        started = time.time()
        result = (freeze(run, config) if stage == "freeze" else baseline(run, config, gpu)
                  if stage == "baseline" else calibration(run, config, gpu, scripted_agent)
                  if stage == "calibration" else control(run, config, gpu))
        result["elapsed_seconds"] = time.time() - started
        write_json(run / stage / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "trainability", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--gpu", type=int, required=True)
        if name == "trainability":
            command.add_argument("--parameters", type=Path)
        if name == "run":
            command.add_argument("--preflight", type=Path, required=True)
            command.add_argument("--scripted-agent", type=Path)
            command.add_argument("--stage", choices=STAGES)
    for name in ("evaluate", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--gpu", type=int, required=True)
            command.add_argument("--oracle", type=Path)
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    if args.command in ("evaluate", "summarize") and not re.match(r"^\d{12}(?:_|$)", run.name):
        parser.error("--run-dir must start with YYYYMMDDHHMM")
    if args.command == "summarize":
        summarize(run)
        return 0
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    if args.command == "evaluate":
        config = json.loads((run / "config.json").read_text())
        with run_lock(run), gpu_lease(str(args.gpu)):
            result = evaluate(run, config, args.gpu, args.oracle.resolve() if args.oracle else None)
        print(json.dumps(result))
        return 0 if result["accepted"] else 1
    config = load_config(args.config)
    if args.command == "run":
        run = args.run_dir.resolve()
        stages = requested_stages(run, args.stage)
        if json.loads((args.preflight / "preflight.json").read_text())["passed"] is not True:
            raise ValueError("A passing preflight is required")
        config["agent"] = asdict(load_task(config["agent_task"]).agent)
        if config["agent"]["isolation"] != "docker":
            raise ValueError("The research agent requires Docker isolation")
        script = args.scripted_agent.resolve() if args.scripted_agent else None
        if script is not None and not script.is_file():
            raise ValueError("--scripted-agent must name a Python file")
        kind = "scripted" if script else "model"
        if args.stage:
            if (json.loads((run / "config.json").read_text()) != config or
                    json.loads((run / "run.json").read_text())["agent_kind"] != kind):
                raise ValueError("Stage arguments differ from the frozen run")
        else:
            run = new_run(run, config, "run", agent_kind=kind)
        with run_lock(run), gpu_lease(str(args.gpu)):
            result = run_stages(run, config, args.gpu, stages, script)
        print(json.dumps(result))
        return 0
    if args.command == "trainability":
        parameters = (load_parameters(args.parameters, config) if args.parameters else
                      config["target_parameters"])
    run = new_run(args.run_dir, config, args.command)
    with run_lock(run), gpu_lease(str(args.gpu)):
        result = (preflight(run, config, args.gpu) if args.command == "preflight" else
                  trainability(run, config, args.gpu, parameters))
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
