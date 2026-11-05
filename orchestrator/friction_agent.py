"""Autonomous friction diagnosis, source reconstruction and PPO repair."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

from integrations.friction_rl.reward_sandbox import export_rollout
from orchestrator import launch_agent
from orchestrator.dynamics_adaptation import agent_command as adaptation_agent_command
from orchestrator.files import codex_token_usage, write_json
from orchestrator.friction_agent_jobs import (ROOT, development_episodes, evaluation_episodes,
                                              rollout_spec, run_job)
from orchestrator.friction_agent_service import FrictionAgentService
from orchestrator.session_runner import stop_process

NUMBER = (int, float)
PHYSICS = dict.fromkeys(("friction_scale", "mass_scale", "restitution", "gripper_scale"), NUMBER)
RANGES = {"friction_scale": [.05, 2.], "mass_scale": [.25, 4.], "restitution": [0., 1.]}
NOMINAL = {"friction_scale": 1., "mass_scale": 1., "restitution": 0., "gripper_scale": 1.}
CONFIG_TYPES = {
    "simulator_python": str, "startup_timeout": NUMBER, "step_timeout": NUMBER, "seed": int,
    "num_envs": int, "cube_shift": [NUMBER, NUMBER], "p0": {"policy": str, "demos": str},
    "source_parameters": PHYSICS, "target_parameters": PHYSICS,
    "parameter_ranges": {key: [NUMBER, NUMBER] for key in RANGES},
    "eval_deterministic": int, "eval_stochastic": int, "required_successes": int,
    "budgets": {"wall_hours": NUMBER, "rounds": int, "iterations_per_round": int,
                "iterations_total": int, "target_episodes": int, "diagnostic_episodes": int},
    "training": {"init_std": [NUMBER] * 5, "episodes_per_iter": int, "warmup_iterations": int,
                 "gamma": NUMBER, "gae_lambda": NUMBER, "clip": NUMBER, "epochs": int, "minibatch": int,
                 "actor_lr": NUMBER, "critic_lr": NUMBER, "target_kl": NUMBER, "bc_weight": NUMBER},
    "sandbox": {"image": str, "timeout": NUMBER, "memory_mb": int},
    "agent": {"preset": str, "model": str, "reasoning_effort": str, "service_tier": str,
              "image": str, "network": str, "cpus": int, "memory_mb": int},
}
PROMPT = ("Read instruction.md and work autonomously. Use scripts/adapt.py for all experiments. "
          "Write analysis code under notes/ and reward programs under rewards/. Finish by submitting "
          "submission/diagnosis.json with scripts/adapt.py submit.")


def load_config(path):
    """Validate the exact configuration schema and fixed experiment boundaries."""
    def check(value, schema):
        if type(schema) is dict:
            if type(value) is not dict or value.keys() != schema.keys():
                raise ValueError("Configuration keys differ from the specification")
            for key in schema:
                check(value[key], schema[key])
        elif type(schema) is list:
            if type(value) is not list or len(value) != len(schema):
                raise ValueError("Configuration vector has the wrong length")
            for item, kind in zip(value, schema):
                check(item, kind)
        elif type(value) not in (schema if type(schema) is tuple else (schema,)):
            raise ValueError("Configuration value has the wrong type")
        elif type(value) in NUMBER and not math.isfinite(value):
            raise ValueError("Configuration numbers must be finite")
    cfg = json.loads(Path(path).read_text())
    check(cfg, CONFIG_TYPES)
    if (not 1 <= cfg["num_envs"] <= 2 or cfg["seed"] < 0 or cfg["cube_shift"] != [.03, 0.]
            or cfg["source_parameters"] != NOMINAL or cfg["parameter_ranges"] != RANGES
            or cfg["target_parameters"]["gripper_scale"] != 1.):
        raise ValueError("Invalid fixed scene, physics or num_envs")
    if any(not low <= cfg["target_parameters"][key] <= high for key, (low, high) in RANGES.items()):
        raise ValueError("Target parameters outside permitted physics ranges")
    if (cfg["startup_timeout"] <= 0 or cfg["step_timeout"] <= 0
            or any(value <= 0 for value in cfg["budgets"].values())
            or cfg["budgets"]["iterations_per_round"] < 4 or cfg["budgets"]["iterations_total"] < 4
            or min(cfg["eval_deterministic"], cfg["eval_stochastic"], cfg["required_successes"]) < 0
            or cfg["eval_deterministic"] + cfg["eval_stochastic"] == 0):
        raise ValueError("Invalid experiment counts or budgets")
    training = cfg["training"]
    if (training["warmup_iterations"] != 2 or min(training["init_std"]) <= 0
            or any(training[key] <= 0 for key in ("episodes_per_iter", "epochs", "minibatch", "actor_lr",
                                                  "critic_lr", "target_kl", "clip"))
            or not 0 <= training["gamma"] <= 1 or not 0 <= training["gae_lambda"] <= 1
            or training["bc_weight"] < 0):
        raise ValueError("Invalid training configuration")
    if (cfg["sandbox"]["timeout"] <= 0 or cfg["sandbox"]["memory_mb"] <= 0
            or cfg["agent"]["cpus"] <= 0 or cfg["agent"]["memory_mb"] <= 0
            or cfg["agent"]["preset"] != "codex" or cfg["agent"]["network"] != "filtered"
            or any(not cfg["agent"][key] for key in ("model", "reasoning_effort", "service_tier", "image"))):
        raise ValueError("Invalid sandbox or agent configuration")
    return cfg


def select_network(networks):
    """Rank deployed networks by strict successes, lenient successes and earlier round."""
    return max((row for row in networks if row["strict_outcomes"] is not None),
               key=lambda row: (sum(row["strict_outcomes"]), sum(row["lenient_outcomes"]), -row["round"]))


def freeze(run, cfg):
    """Freeze the resolved configuration, initial policy and source motion prior."""
    write_json(run / "config.json", cfg)
    for source, destination in ((cfg["p0"]["policy"], run / "networks/N000/policy.pt"),
                                (cfg["p0"]["demos"], run / "prior/demos.npz")):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return {"status": "complete"}


def baseline(run, cfg):
    """Run the initial network in source and target on all four development episodes."""
    results = {label: run_job(run / "baseline" / label, rollout_spec(
        cfg, run / "networks/N000/policy.pt", cfg[f"{label}_parameters"], development_episodes(cfg["seed"])))
        for label in ("source", "target")}
    return {"status": "complete" if all(r["status"] == "complete" for r in results.values()) else "failed",
            "arms": results}


def prepare_workspace(run, cfg):
    """Render only public instructions, nominal physics and baseline observations."""
    workspace = run / "calibration/workspace"
    template = ROOT / "tasks/FrictionAgent/Workspace"
    workspace.mkdir(parents=True)
    shutil.copytree(template / "scripts", workspace / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    instruction = (template / "instruction.md").read_text()
    values = {**cfg["budgets"], **cfg["training"], "sandbox_memory_mb": cfg["sandbox"]["memory_mb"],
              "sandbox_timeout": cfg["sandbox"]["timeout"]}
    for key, value in values.items():
        instruction = instruction.replace("{{" + key.upper() + "}}", str(value))
    (workspace / "instruction.md").write_text(instruction)
    write_json(workspace / "nominal.json", {"parameters": cfg["source_parameters"], "ranges": cfg["parameter_ranges"]})
    (workspace / "observations").mkdir()
    for label in ("source", "target"):
        result = json.loads((run / "baseline" / label / "result.json").read_text())
        export_rollout(result, workspace / "observations/baseline" / label)
    return workspace


def agent_command(cfg, workspace, socket_dir):
    """Reuse the adaptation container mounts with the specified Codex command and prompt."""
    name, argv = adaptation_agent_command(cfg, workspace, socket_dir)
    agent = cfg["agent"]
    argv[argv.index(agent["image"]) + 1:] = launch_agent.adapt_for_docker(launch_agent.agent_argv(
        agent["preset"], PROMPT, model=agent["model"], effort=agent["reasoning_effort"],
        service_tier=agent["service_tier"]), preset=agent["preset"])
    return name, argv


def calibration(run, cfg, scripted_agent):
    """Serve the isolated agent until submission, exit or deadline and select its best network."""
    branch, started = run / "calibration", time.time()
    workspace = prepare_workspace(run, cfg)
    with ExitStack() as stack:
        temporary = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="friction-agent-")))
        temporary.chmod(0o755)
        socket_dir = temporary / "socket"
        socket_dir.mkdir(mode=0o755)
        listener = stack.enter_context(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))
        address = socket_dir / "session.sock"
        listener.bind(str(address))
        listener.listen(1)
        address.chmod(0o666)
        service = FrictionAgentService(branch, cfg, float("inf"))
        executor = stack.enter_context(ThreadPoolExecutor(1))
        future = executor.submit(service.serve, listener)
        stack.callback(service.stop, "stopped_by_agent_exit")
        container, command = ((None, [sys.executable, str(scripted_agent)]) if scripted_agent else
                              agent_command(cfg, workspace, socket_dir))
        service.deadline = time.time() + cfg["budgets"]["wall_hours"] * 3600
        log = stack.enter_context((branch / "agent.log").open("x"))
        process = subprocess.Popen(command, cwd=workspace, env={**os.environ, "EMBODIEDEVO_SOCKET": str(address)},
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        resources = stack.enter_context(ExitStack())
        resources.callback(stop_process, process)
        if container:
            resources.callback(subprocess.run, ["docker", "rm", "-f", container], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        while not service.submitted and time.time() < service.deadline and process.poll() is None:
            if future.done():
                future.result()
                raise RuntimeError("Friction service ended before the agent")
            service.poll()
            time.sleep(.1)
        reason = ("submitted" if service.submitted else "deadline" if time.time() >= service.deadline
                  else "agent_exited_without_submission")
        service.stop("stopped_by_deadline" if reason == "deadline" else "stopped_by_agent_exit")
        future.result()
        resources.close()
        home = temporary / "agent-home"
        if (home / ".codex/sessions").is_dir():
            shutil.copytree(home / ".codex/sessions", branch / "agent_sessions")
        diagnosis = json.loads((branch / "diagnosis.json").read_text()) if service.submitted else None
        selected = select_network(service.networks)
        write_json(branch / "selection.json", {"selected": selected["id"], "networks": service.networks,
                                              "preferred_network": diagnosis["preferred_network"] if diagnosis else None})
        return {"status": "complete", "end_reason": reason, "budgets": service.used, "networks": service.networks,
                "diagnosis": diagnosis, "model_usage": codex_token_usage(home), "started_at": started,
                "deadline": service.deadline, "agent_returncode": process.returncode}


def counts(result):
    """Count both outcomes over every planned episode, including incomplete failures."""
    return {"episodes": len(result["episodes"]),
            **{key: sum(row["complete"] and row[field] for row in result["episodes"])
               for key, field in (("strict_successes", "strict_success"), ("lenient_successes", "success_lenient"))}}


def evaluate(run, cfg):
    """Evaluate P0 and the host-selected network once each on held-out target episodes."""
    selected = json.loads((run / "calibration/selection.json").read_text())["selected"]
    arms = {}
    for network in dict.fromkeys(("N000", selected)):
        root = run if network == "N000" else run / "calibration"
        result = run_job(run / "evaluation" / network, rollout_spec(
            cfg, root / "networks" / network / "policy.pt", cfg["target_parameters"], evaluation_episodes(cfg)))
        arms[network] = {"status": result["status"], **counts(result)}
    result = {"selected": selected, "arms": arms,
              "accepted": arms[selected]["strict_successes"] >= cfg["required_successes"]}
    write_json(run / "evaluation/results.json", result)
    return result


def trainability(run, cfg, parameters, reward, iterations):
    """Run a host-only oracle round and evaluate it without exporting to an agent."""
    output = run / "trainability/network"
    trained = run_job(run / "trainability/train", {
        "kind": "train", "config": cfg, "parent": str(run / "networks/N000/policy.pt"),
        "demos": str(run / "prior/demos.npz"), "parameters": parameters, "iterations": iterations,
        "reward": str(reward), "round": 1, "warmup": True, "output": str(output)})
    if trained["status"] != "complete":
        return trained
    result = run_job(run / "trainability/evaluation", rollout_spec(
        cfg, output / "policy.pt", cfg["target_parameters"], evaluation_episodes(cfg)))
    return {"status": result["status"], **counts(result),
            "accepted": counts(result)["strict_successes"] >= cfg["required_successes"]}


def summarize(run):
    """Write and print stage timing, networks, budgets, evaluation, tokens and diagnosis."""
    stages = {path.parent.name: json.loads(path.read_text()) for path in run.glob("*/result.json")}
    calibration_result = stages.get("calibration", {})
    summary = {"stage_seconds": {key: value["elapsed_seconds"] for key, value in stages.items()},
               **{key: calibration_result.get(key) for key in ("networks", "budgets", "model_usage", "diagnosis")},
               "evaluation": stages.get("evaluation"),
               "repeatability": "The simulator is not bit-repeatable across successive episodes on a worker."}
    write_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def stage(run, name, function, *args):
    """Record a stage result and its elapsed wall time."""
    started = time.monotonic()
    result = function(run, *args)
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(run / name / "result.json", result)
    return result


def parse_args(argv=None):
    """Require timestamped new experiment runs or an existing calibrated run."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "evaluate", "trainability", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", required=True, type=Path)
        command.add_argument("--gpu", type=int, default=0)
        if name in ("run", "trainability"):
            command.add_argument("--config", required=True, type=Path)
        if name == "run":
            command.add_argument("--scripted-agent", type=Path)
        if name == "trainability":
            command.add_argument("--parameters", required=True, type=Path)
            command.add_argument("--reward", required=True, type=Path)
            command.add_argument("--iterations", required=True, type=int)
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    if not re.match(r"^\d{12}(?:_|$)", args.run_dir.name) or args.gpu < 0:
        parser.error("--run-dir must start with YYYYMMDDHHMM and --gpu must be nonnegative")
    new = args.command in ("run", "trainability")
    if new == args.run_dir.exists():
        parser.error("run/trainability require a new directory; evaluate/summarize require an existing run")
    for key in ("config", "scripted_agent", "parameters", "reward"):
        path = getattr(args, key, None)
        if path is not None:
            if not path.is_file():
                parser.error(f"--{key.replace('_', '-')} must name an existing file")
            setattr(args, key, path.resolve())
    if args.command == "trainability" and args.iterations <= 0:
        parser.error("--iterations must be positive")
    return args


def main(argv=None):
    """Run the selected friction-agent stage without altering benchmark execution."""
    args = parse_args(argv)
    run = args.run_dir
    if args.command == "summarize":
        return summarize(run)
    if args.command == "evaluate":
        cfg = {**json.loads((run / "config.json").read_text()), "gpu": args.gpu}
        return stage(run, "evaluation", evaluate, cfg)
    cfg = {**load_config(args.config), "gpu": args.gpu}
    parameters = None
    if args.command == "trainability":
        parameters = json.loads(args.parameters.read_text())
        ranges = {**RANGES, "gripper_scale": [1., 1.]}
        if (type(parameters) is not dict or parameters.keys() - ranges.keys()
                or any(type(v) not in NUMBER or not math.isfinite(v) or not ranges[k][0] <= v <= ranges[k][1]
                       for k, v in parameters.items())):
            raise ValueError("Invalid calibrated source parameters")
        parameters = {**NOMINAL, **parameters}
    run.mkdir(parents=True)
    stage(run, "freeze", freeze, cfg)
    if args.command == "trainability":
        return stage(run, "trainability", trainability, cfg, parameters, args.reward, args.iterations)
    result = stage(run, "baseline", baseline, cfg)
    if result["status"] != "complete":
        return result
    return stage(run, "calibration", calibration, cfg, args.scripted_agent)


if __name__ == "__main__":
    main()
