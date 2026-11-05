"""Standalone CMA-ES friction-transfer training and evaluation."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import pickle
from queue import Queue
import re
import time

import cma
import numpy as np

from integrations.dynamics_rl.client import SimulatorClient
from integrations.dynamics_rl.features import sample_scene
from integrations.friction_rl.episode import run_episode
from integrations.friction_rl.primitive import HIGH, LOW, PARAMETER_NAMES, physical
from orchestrator.files import write_json

CONFIG_KEYS = {
    "simulator_python", "startup_timeout", "step_timeout", "seed", "num_envs",
    "layout_radius", "cube_shift", "popsize", "sigma0_source", "sigma0_continue",
    "max_iterations_source", "max_iterations_continue", "dev_every",
    "dev_episodes", "dev_pass", "eval_episodes",
}
COUNTS = ("num_envs", "popsize", "max_iterations_source", "max_iterations_continue",
          "dev_every", "dev_episodes", "dev_pass", "eval_episodes")


def load_config(path):
    """Validate the exact experiment configuration at its input boundary."""
    cfg = json.loads(Path(path).read_text())
    if not isinstance(cfg, dict) or set(cfg) != CONFIG_KEYS:
        raise ValueError("Config must contain exactly the friction-transfer keys")
    for key in COUNTS:
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("sigma0_source", "sigma0_continue", "startup_timeout", "step_timeout"):
        if type(cfg[key]) not in (int, float) or not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (type(cfg["layout_radius"]) not in (int, float)
            or not 0 <= cfg["layout_radius"] <= .03):
        raise ValueError("layout_radius must lie in [0, 0.03]")
    shift = cfg["cube_shift"]
    if (not isinstance(shift, list) or len(shift) != 2
            or any(type(value) not in (int, float) or not np.isfinite(value)
                   or not -.03 <= value <= .03 for value in shift)):
        raise ValueError("cube_shift must contain two finite numbers in [-0.03, 0.03]")
    if cfg["dev_pass"] > cfg["dev_episodes"] or cfg["num_envs"] > 2 or cfg["popsize"] < 2:
        raise ValueError("Require dev_pass <= dev_episodes, num_envs <= 2 and popsize >= 2")
    if not isinstance(cfg["simulator_python"], str) or not cfg["simulator_python"]:
        raise ValueError("simulator_python must be a nonempty path")
    return cfg


def append_record(path, record):
    """Append one JSON record without transforming its evidence."""
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


class SimulatorPool:
    """Run a shared episode queue with one thread per persistent client."""

    def __init__(self, clients):
        self.clients = clients

    def run(self, jobs, grip_force=0.0):
        """Run primitive episodes through the shared ordered map."""
        def episode(client, job):
            params, scene, folder = job
            return run_episode(client, params, scene, client.run_dir / folder,
                               grip_force=grip_force)[0]

        return self.map(episode, jobs)

    def map(self, function, jobs):
        """Apply a client function over the shared queue in job order."""
        queue = Queue()
        for index, job in enumerate(jobs):
            queue.put((index, job))
        for _ in self.clients:
            queue.put(None)
        records = [None] * len(jobs)

        def work(client):
            for index, job in iter(queue.get, None):
                records[index] = function(client, job)

        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            list(executor.map(work, self.clients))
        return records


@contextmanager
def simulator_pool(args, cfg):
    """Preserve the existing CLI physics and rendering choices."""
    mode = "render" if args.command == "evaluate" and args.render else "state"
    with start_pool(args.run_dir, cfg, parameters(args), mode, args.gpu) as pool:
        yield pool


@contextmanager
def start_pool(run_dir, cfg, physics, mode, gpu):
    """Start clients concurrently and close every client on any exit."""
    with ExitStack() as stack:
        clients = []
        for k in range(cfg["num_envs"]):
            client = SimulatorClient(
                run_dir / f"{run_dir.name[:12]}_sim_w{k}",
                parameters=physics, gpu=gpu, mode=mode,
                simulator_python=cfg["simulator_python"],
                startup_timeout=cfg["startup_timeout"], step_timeout=cfg["step_timeout"],
                worker_module="integrations.friction_rl.worker")
            stack.callback(client.close)
            clients.append(client)
        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            list(executor.map(lambda client: client.start(), clients))
        yield SimulatorPool(clients)


def parameters(args):
    """Return the stage's complete physics and gripper settings."""
    return {"friction_scale": args.friction_scale, "mass_scale": 1.0,
            "restitution": 0.0, "gripper_scale": args.gripper_scale}


def build_scene(seed, cfg):
    """Sample a layout and apply the configured cube-only XY shift."""
    scene = sample_scene(seed, cfg["layout_radius"])
    position = scene["Rigid"]["cube"][0]["default_pos"]
    position[:2] = (np.asarray(position[:2]) + cfg["cube_shift"]).tolist()
    return scene


def layout_jobs(params, cfg, layouts, count, prefix=""):
    """Build the shared deterministic dev or evaluation layout set."""
    seed = 2000 if layouts == "dev" else 3000
    return [(params, build_scene(seed + k, cfg), f"{prefix}{layouts}_l{k:02d}")
            for k in range(count)]


def summary(records):
    """Aggregate the host success gates and diagnostic features."""
    successes = sum(record["success"] for record in records)
    rate = successes / len(records)
    return {
        "successes": successes, "N": len(records), "rate": rate, "pass": rate >= .9,
        "successes_strict": sum(record["success_strict"] for record in records),
        **{key: max(record[key] for record in records)
           for key in ("cmd_peak_speed", "cmd_peak_accel", "obs_peak_speed")},
        "settle_before_release": min(record["settle_before_release"] for record in records),
        "per_layout_success": [record["success"] for record in records],
        "stop_reasons": dict(Counter(record["stop_reason"] for record in records)),
        "means": {key: float(np.mean([record[key] for record in records]))
                  for key in ("max_clear", "slip", "grip_force_min_carry", "track_err_carry")},
        "release_dist_failures": [record["release_dist"] for record in records if not record["success"]],
    }


def train(args, cfg, pool):
    """Search policies, check dev layouts and select the latest best dev result."""
    if args.resume_es:
        with args.resume_es.open("rb") as stream:
            es = pickle.load(stream)
        start = args.start_iteration
        if start != es.countiter or not 0 <= start < es.ft_max_iterations:
            raise ValueError("--start-iteration must match the checkpoint and precede the iteration limit")
    else:
        initial = (json.loads(args.init_policy.read_text())["normalized"]
                   if args.init_policy else np.zeros(len(PARAMETER_NAMES)))
        stage = "continue" if args.init_policy else "source"
        es = cma.CMAEvolutionStrategy(initial, cfg[f"sigma0_{stage}"], {
            "bounds": [-1, 1], "popsize": cfg["popsize"], "seed": cfg["seed"] + 1, "verbose": -9})
        es.ft_init_policy = str(args.init_policy) if args.init_policy else None
        es.ft_max_iterations = cfg[f"max_iterations_{stage}"]
        es.ft_dev_counts = []
        es.ft_best_policy = None
        start = 0
    best = es.ft_best_policy
    for i in range(start, es.ft_max_iterations):
        started = time.monotonic()
        candidates = es.ask()
        jobs = [(physical(u), build_scene(100000 + 1000*i + j, cfg),
                 f"it{i:03d}_c{j:02d}") for j, u in enumerate(candidates)]
        records = pool.run(jobs)
        rewards = [record[f"reward_{args.reward}"] for record in records]
        es.tell(candidates, [-reward for reward in rewards])
        for j, (u, record) in enumerate(zip(candidates, records)):
            append_record(args.run_dir / "episodes.jsonl",
                          {**record, "i": i, "candidate": j, "normalized": u.tolist(), "set": "train"})
        append_record(args.run_dir / "iterations.jsonl", {
            "i": i, "xfavorite": es.result.xfavorite.tolist(), "sigma": float(es.sigma),
            "std": es.result.stds.tolist(), "rewards": rewards,
            "successes": [record["success"] for record in records],
            "stop_reasons": [record["stop_reason"] for record in records],
            "wall_time": time.monotonic() - started,
        })
        with (args.run_dir / "es.pkl").open("wb") as stream:
            pickle.dump(es, stream)
        if (i + 1) % cfg["dev_every"] == 0 or i == es.ft_max_iterations - 1:
            u = es.result.xfavorite
            dev_records = pool.run(layout_jobs(
                physical(u), cfg, "dev", cfg["dev_episodes"], prefix=f"it{i:03d}_"))
            dev = summary(dev_records)
            policy = {
                "version": "ft-primitive-v1", "normalized": u.tolist(), "physical": physical(u),
                "iteration": i, "dev": dev, "friction_scale": args.friction_scale,
                "gripper_scale": args.gripper_scale, "reward": args.reward,
                "init_policy": es.ft_init_policy,
            }
            write_json(args.run_dir / f"policy_it{i:03d}.json", policy)
            append_record(args.run_dir / "dev.jsonl", {"i": i, **dev})
            for k, record in enumerate(dev_records):
                append_record(args.run_dir / "episodes.jsonl",
                              {**record, "i": i, "layout": k, "set": "dev"})
            es.ft_dev_counts.append(dev["successes"])
            if best is None or dev["successes"] >= best["dev"]["successes"]:
                best = policy
                es.ft_best_policy = best
            with (args.run_dir / "es.pkl").open("wb") as stream:
                pickle.dump(es, stream)
            if len(es.ft_dev_counts) >= 2 and min(es.ft_dev_counts[-2:]) >= cfg["dev_pass"]:
                break
    write_json(args.run_dir / "policy.json", best)
    return best


def evaluate(args, cfg, pool):
    """Evaluate one deterministic policy on the requested layout prefix."""
    u = json.loads(args.policy.read_text())["normalized"] if args.policy else np.zeros(len(PARAMETER_NAMES))
    params = physical(u)
    params.update(args.override or {})
    count = args.episodes if args.episodes is not None else cfg[f"{args.layouts}_episodes"]
    records = pool.run(layout_jobs(params, cfg, args.layouts, count), grip_force=args.grip_force)
    for k, record in enumerate(records):
        append_record(args.run_dir / "episodes.jsonl", {**record, "layout": k, "set": args.layouts})
    result = summary(records)
    write_json(args.run_dir / "summary.json", result)
    return result


def parse_args(argv):
    """Validate command-line arguments before creating run evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", required=True, type=Path)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--friction-scale", required=True, type=float)
        command.add_argument("--gpu", type=int, default=0)
        command.add_argument("--gripper-scale", type=float, default=1.0)
        if name == "train":
            command.add_argument("--reward", required=True, choices=("v1", "v2"))
            initial = command.add_mutually_exclusive_group()
            initial.add_argument("--init-policy", type=Path)
            initial.add_argument("--resume-es", type=Path)
            command.add_argument("--start-iteration", type=int)
        else:
            command.add_argument("--policy", type=Path)
            command.add_argument("--override", type=json.loads)
            command.add_argument("--layouts", required=True, choices=("dev", "eval"))
            command.add_argument("--episodes", type=int)
            command.add_argument("--render", action="store_true")
            command.add_argument("--grip-force", type=float, default=0.0)
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    if not re.match(r"^\d{12}(?:_|$)", args.run_dir.name) or args.run_dir.exists():
        parser.error("--run-dir must be new and start with YYYYMMDDHHMM")
    if (args.gpu < 0 or not np.isfinite(args.friction_scale) or args.friction_scale < 0
            or not np.isfinite(args.gripper_scale) or args.gripper_scale <= 0):
        parser.error("require nonnegative gpu/friction and positive gripper scale")
    cfg = load_config(args.config)
    if args.command == "train":
        if (args.resume_es is None) != (args.start_iteration is None):
            parser.error("--resume-es and --start-iteration must be supplied together")
        if args.start_iteration is not None and args.start_iteration < 0:
            parser.error("--start-iteration must be nonnegative")
    else:
        if not 0 <= args.grip_force <= 100:
            parser.error("--grip-force must lie in [0, 100] N")
        if args.episodes is not None and not 1 <= args.episodes <= cfg[f"{args.layouts}_episodes"]:
            parser.error("--episodes must select a nonempty prefix of the layout set")
        if args.override is not None:
            if not isinstance(args.override, dict) or not set(args.override) <= set(PARAMETER_NAMES):
                parser.error("--override must be an object of physical parameter values")
            for key, value in args.override.items():
                index = PARAMETER_NAMES.index(key)
                if (type(value) not in (int, float) or not LOW[index] <= value <= HIGH[index]
                        or (key.startswith("t_") and value != int(value))):
                    parser.error(f"invalid physical override: {key}")
                if key.startswith("t_"):
                    args.override[key] = int(value)
    return args, cfg


def main(argv=None):
    """Create a new run and execute its requested standalone stage."""
    args, cfg = parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.run_dir / "config.json", {
        "arguments": resolved, "config": cfg, "parameters": parameters(args)})
    with simulator_pool(args, cfg) as pool:
        return train(args, cfg, pool) if args.command == "train" else evaluate(args, cfg, pool)


if __name__ == "__main__":
    main()
