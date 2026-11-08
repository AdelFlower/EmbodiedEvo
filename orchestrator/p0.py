"""Prepare and evaluate the initial neural policy in the nominal source."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from threading import Lock
import time
import traceback

import numpy as np
import torch

from integrations.friction_rl.network import (
    ACT_DIM, OBS_DIM, ScriptController, policy_controller, run_net_episode,
)
from integrations.friction_rl.ppo import ActorCritic, bc_fit, load, save
from integrations.friction_rl.primitive import PARAMETER_NAMES
from integrations.friction_rl.simulation import SOURCE_PARAMETERS, build_scene, start_pool
from orchestrator.files import append_jsonl, write_json

INTEGER_KEYS = ("seed", "num_envs", "demo_episodes", "bc_steps", "bc_batch")
NUMBER_KEYS = ("startup_timeout", "step_timeout", "layout_radius", "script_force",
               "demo_noise_rho", "bc_lr")
VECTOR_SIZES = {"cube_shift": 2, "demo_noise": 3, "init_std": ACT_DIM}
CONFIG_KEYS = set(INTEGER_KEYS + NUMBER_KEYS) | set(VECTOR_SIZES) | {"simulator_python", "script"}


def load_config(path):
    """Validate the source-only preparation configuration at its input boundary."""
    cfg = json.loads(Path(path).read_text())
    if type(cfg) is not dict or set(cfg) != CONFIG_KEYS:
        raise ValueError("Config must contain exactly the P0 preparation keys")
    for key in INTEGER_KEYS:
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid integer for {key}")
    for key in NUMBER_KEYS:
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be finite and numeric")
    for key, length in VECTOR_SIZES.items():
        if (type(cfg[key]) is not list or len(cfg[key]) != length
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in cfg[key])):
            raise ValueError(f"{key} must contain {length} finite numbers")
    if type(cfg["simulator_python"]) is not str or not cfg["simulator_python"]:
        raise ValueError("simulator_python must be a nonempty path")
    script = cfg["script"]
    if type(script) is not dict or set(script) != set(PARAMETER_NAMES):
        raise ValueError("script must contain exactly the source script parameters")
    for key, value in script.items():
        types = (int,) if key.startswith("t_") else (int, float)
        if type(value) not in types or not math.isfinite(value):
            raise ValueError(f"Invalid value for script.{key}")
        if key.startswith("t_") and value < (0 if key == "t_settle" else 1):
            raise ValueError(f"Invalid duration for script.{key}")
    if cfg["num_envs"] > 2 or cfg["layout_radius"] != 0 or cfg["cube_shift"] != [.03, 0.]:
        raise ValueError("P0 requires the fixed source scene and one or two simulators")
    if (min(cfg["startup_timeout"], cfg["step_timeout"], cfg["bc_lr"]) <= 0
            or cfg["script_force"] != 15.0 or not 0 <= cfg["demo_noise_rho"] < 1
            or min(cfg["demo_noise"]) < 0 or min(cfg["init_std"]) <= 0):
        raise ValueError("Invalid source demonstration or BC settings")
    return cfg


def summary(records):
    """Report source successes and motion diagnostics without changing outcomes."""
    return {
        "successes": sum(row["success"] for row in records), "N": len(records),
        "strict_successes": sum(row["strict_success"] for row in records),
        "stop_reasons": dict(Counter(row["stop_reason"] for row in records)),
        "means": {key: float(np.mean([row[key] for row in records]))
                  for key in ("max_clear", "slip", "force_mean_held", "release_dist")},
        **{key: max(row[key] for row in records) for key in ("cmd_peak_speed", "cmd_peak_accel")},
    }


def collect(args, cfg, pool, jobs):
    """Run source episodes and retain each finished record before another episode starts."""
    scene = build_scene(0, cfg)
    records_lock = Lock()

    def episode(client, job):
        name, controller = job
        result = run_net_episode(client, scene, client.run_dir / name, controller)
        with records_lock:
            append_jsonl(args.run_dir / "episodes.jsonl", {"episode": name, **result[0]})
        return result

    return pool.map(episode, jobs)


def demos(args, cfg, pool):
    """Collect the clean source script and all noisy corrective demonstrations."""
    jobs = [(f"demo_e{k:02d}", ScriptController(
        cfg["script"], cfg["script_force"], np.zeros(3) if k == 0 else cfg["demo_noise"],
        cfg["demo_noise_rho"], np.random.default_rng([cfg["seed"], 0, k])))
        for k in range(cfg["demo_episodes"])]
    results = collect(args, cfg, pool, jobs)
    np.savez(args.run_dir / "demos.npz",
             obs=np.concatenate([batch["obs"] for _, _, batch in results]),
             labels=np.concatenate([batch["stored"] for _, _, batch in results]),
             episode=np.concatenate([np.full(len(batch["obs"]), k)
                                     for k, (_, _, batch) in enumerate(results)]))
    return summary([record for record, _, _ in results])


def bc(args, cfg):
    """Fit a source policy from demonstrations without starting simulator clients."""
    with np.load(args.demos) as data:
        model = ActorCritic(OBS_DIM, ACT_DIM)
        records = bc_fit(model, data["obs"], data["labels"], cfg["bc_steps"], cfg["bc_batch"],
                         cfg["bc_lr"], cfg["seed"], cfg["init_std"])
    for record in records:
        append_jsonl(args.run_dir / "bc.jsonl", record)
    save(args.run_dir / "policy.pt", model)
    return records


def evaluate(args, cfg, pool):
    """Evaluate a frozen policy in the nominal source with independent episode seeds."""
    model = load(args.policy)
    jobs = [(f"eval_e{k:02d}", policy_controller(model, [cfg["seed"], 2, k], args.stochastic))
            for k in range(args.episodes)]
    results = collect(args, cfg, pool, jobs)
    result = summary([record for record, _, _ in results])
    write_json(args.run_dir / "summary.json", result)
    return result


def parse_args(argv):
    """Require fresh timestamped outputs and expose only source preparation operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demos", "bc", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", required=True, type=Path)
        command.add_argument("--config", required=True, type=Path)
        if name != "bc":
            command.add_argument("--gpu", type=int, default=0)
        if name == "bc":
            command.add_argument("--demos", required=True, type=Path)
        if name == "evaluate":
            command.add_argument("--policy", required=True, type=Path)
            command.add_argument("--episodes", required=True, type=int)
            command.add_argument("--stochastic", action="store_true")
            command.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    if not re.match(r"^\d{12}(?:_|$)", args.run_dir.name) or args.run_dir.exists():
        parser.error("--run-dir must be new and start with YYYYMMDDHHMM")
    if args.command != "bc" and args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    if args.command == "evaluate" and args.episodes <= 0:
        parser.error("--episodes must be positive")
    for key in ("config", "demos", "policy"):
        if key in vars(args) and not getattr(args, key).is_file():
            parser.error(f"--{key} must name an existing file")
    return args, load_config(args.config)


def main(argv=None):
    """Record one preparation attempt and preserve failures without retries."""
    args, cfg = parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.run_dir / "config.json", cfg)
    write_json(args.run_dir / "run.json", {
        "command": args.command,
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "parameters": SOURCE_PARAMETERS, "started_at": time.time(),
    })
    torch.manual_seed(cfg["seed"])
    started = time.monotonic()
    record = {"status": "failed"}
    try:
        if args.command == "bc":
            result = bc(args, cfg)
        else:
            mode = "render" if args.command == "evaluate" and args.render else "state"
            with start_pool(args.run_dir, cfg, SOURCE_PARAMETERS, mode, args.gpu) as pool:
                result = {"demos": demos, "evaluate": evaluate}[args.command](args, cfg, pool)
        record.update(status="complete", result=result)
        return result
    except BaseException as exc:
        record.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        write_json(args.run_dir / "result.json", {**record, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
