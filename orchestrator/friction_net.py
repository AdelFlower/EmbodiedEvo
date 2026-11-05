"""Script demonstrations, behaviour cloning and PPO friction transfer."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from integrations.friction_rl.network import ACT_DIM, OBS_DIM, F_MID, F_HALF, ScriptController, run_net_episode
from integrations.friction_rl.ppo import ActorCritic, act, bc_fit, load, optimizers, ppo_update, save
from integrations.friction_rl.primitive import PARAMETER_NAMES
from orchestrator.files import write_json
from orchestrator.friction_rl import append_record, build_scene, simulator_pool

INTEGER_KEYS = ("seed num_envs demo_episodes bc_steps bc_batch episodes_per_iter max_iterations "
                "warmup_iterations epochs minibatch eval_every stop_successes").split()
NUMBER_KEYS = ("startup_timeout step_timeout layout_radius script_force demo_noise_rho bc_lr gamma "
               "gae_lambda clip actor_lr critic_lr target_kl bc_weight").split()
VECTOR_SIZES = {"cube_shift": 2, "demo_noise": 3, "init_std": ACT_DIM}
CONFIG_KEYS = set(INTEGER_KEYS + NUMBER_KEYS) | set(VECTOR_SIZES) | {"simulator_python", "script"}


def load_config(path):
    """Validate exactly the specified configuration keys and value types."""
    cfg = json.loads(Path(path).read_text())
    if type(cfg) is not dict or set(cfg) != CONFIG_KEYS:
        raise ValueError("Config must contain exactly the network keys")
    for key in INTEGER_KEYS:
        if type(cfg[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    for key in NUMBER_KEYS:
        if type(cfg[key]) not in (int, float):
            raise ValueError(f"{key} must be numeric")
    for key, length in VECTOR_SIZES.items():
        if (type(cfg[key]) is not list or len(cfg[key]) != length
                or any(type(value) not in (int, float) for value in cfg[key])):
            raise ValueError(f"{key} must be a list of {length} numbers")
    if type(cfg["simulator_python"]) is not str:
        raise ValueError("simulator_python must be a string")
    script = cfg["script"]
    if type(script) is not dict or set(script) != set(PARAMETER_NAMES):
        raise ValueError("script must contain exactly the primitive parameters")
    for key, value in script.items():
        types = (int,) if key.startswith("t_") else (int, float)
        if type(value) not in types:
            raise ValueError(f"invalid type for script.{key}")
    return cfg


def summary(records):
    """Aggregate success counts, stop reasons and the specified motion diagnostics."""
    return {
        "successes": sum(record["success"] for record in records), "N": len(records),
        "stop_reasons": dict(Counter(record["stop_reason"] for record in records)),
        "means": {key: float(np.mean([record[key] for record in records]))
                  for key in ("max_clear", "slip", "force_mean_held", "release_dist")},
        **{key: max(record[key] for record in records) for key in ("cmd_peak_speed", "cmd_peak_accel")},
    }


def collect(args, cfg, pool, jobs):
    """Run named controllers in job order and append each episode record."""
    scene = build_scene(0, cfg)

    def episode(client, job):
        name, controller = job
        return run_net_episode(client, scene, client.run_dir / name, controller)

    results = pool.map(episode, jobs)
    for (name, _), (record, _, _) in zip(jobs, results):
        append_record(args.run_dir / "episodes.jsonl", {"episode": name, **record})
    return results


def policy_controller(model, seed, stochastic):
    """Give one episode its own reproducible action sampling stream."""
    rng = np.random.default_rng(seed)
    return lambda t, obs, state, decoder: act(model, obs, rng, stochastic)


def demos(args, cfg, pool):
    """Collect the clean script and corrective noisy demonstrations."""
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
    """Fit the demonstration policy without starting simulator clients."""
    with np.load(args.demos) as data:
        model = ActorCritic(OBS_DIM, ACT_DIM)
        records = bc_fit(model, data["obs"], data["labels"], cfg["bc_steps"], cfg["bc_batch"],
                         cfg["bc_lr"], cfg["seed"], cfg["init_std"])
    for record in records:
        append_record(args.run_dir / "bc.jsonl", record)
    save(args.run_dir / "policy.pt", model)
    return records


def evaluate(args, cfg, pool):
    """Evaluate a saved policy with independent episode seeds."""
    model = load(args.policy)
    jobs = [(f"eval_e{k:02d}", policy_controller(model, [cfg["seed"], 2, k], args.stochastic))
            for k in range(args.episodes)]
    results = collect(args, cfg, pool, jobs)
    result = summary([record for record, _, _ in results])
    write_json(args.run_dir / "summary.json", result)
    return result


def train(args, cfg, pool):
    """Continue PPO until the iteration budget or the paired success gate is reached."""
    model = load(args.init)
    with torch.no_grad():
        model.log_std.copy_(torch.as_tensor(cfg["init_std"], dtype=torch.float32).log())
    actor_optimizer, critic_optimizer = optimizers(model, cfg)
    with np.load(args.demos) as data:
        prior = {key: data[key] for key in ("obs", "labels")}
    closed_obs = torch.as_tensor(prior["obs"][prior["labels"][:, 3] > 0], dtype=torch.float32)
    previous_successes = 0
    for i in range(cfg["max_iterations"]):
        started = time.monotonic()
        jobs = [(f"it{i:03d}_e{j:02d}", policy_controller(model, [cfg["seed"], 1, i, j], True))
                for j in range(cfg["episodes_per_iter"])]
        results = collect(args, cfg, pool, jobs)
        records = [record for record, _, _ in results]
        batch = {key: np.concatenate([b[key] for _, _, b in results])
                 for key in ("obs", "stored", "reward", "done")}
        update = ppo_update(model, batch, prior, cfg, actor_optimizer, critic_optimizer,
                            actor_frozen=i < cfg["warmup_iterations"])
        successes = sum(record["success"] for record in records)
        with torch.no_grad():
            mu_force = F_MID + F_HALF * model.actor(closed_obs)[:, 4].clamp(-1, 1)
            mean_force = float(mu_force.mean())
            std = model.log_std.exp().tolist()
        append_record(args.run_dir / "iterations.jsonl", {
            "i": i, "successes": successes,
            "stop_reasons": dict(Counter(record["stop_reason"] for record in records)),
            "means": {key: float(np.mean([record[key] for record in records]))
                      for key in ("return", "max_clear", "slip", "force_mean_held")},
            "mu_force_demo": mean_force, "std": std, "update": update,
            "wall_time": time.monotonic() - started,
        })
        save(args.run_dir / f"policy_it{i:03d}.pt", model)
        paired_success = i > 0 and min(successes, previous_successes) >= cfg["stop_successes"]
        if (i + 1) % cfg["eval_every"] == 0 or paired_success:
            jobs = [(f"it{i:03d}_det{k}", policy_controller(model, [cfg["seed"], 2, k], False))
                    for k in range(cfg["num_envs"])]
            checks = collect(args, cfg, pool, jobs)
            result = summary([record for record, _, _ in checks])
            append_record(args.run_dir / "evals.jsonl", {"i": i, **result})
            if paired_success and result["successes"] == result["N"]:
                break
        previous_successes = successes
    save(args.run_dir / "policy.pt", model)
    return model


def parse_args(argv):
    """Validate the CLI before creating a new timestamped run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demos", "bc", "evaluate", "train"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", required=True, type=Path)
        command.add_argument("--config", required=True, type=Path)
        if name != "bc":
            command.add_argument("--friction-scale", required=True, type=float)
        if name in ("bc", "train"):
            command.add_argument("--demos", required=True, type=Path)
        if name == "train":
            command.add_argument("--init", required=True, type=Path)
        if name == "evaluate":
            command.add_argument("--policy", required=True, type=Path)
            command.add_argument("--episodes", required=True, type=int)
            command.add_argument("--stochastic", action="store_true")
            command.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    if not re.match(r"^\d{12}(?:_|$)", args.run_dir.name) or args.run_dir.exists():
        parser.error("--run-dir must be new and start with YYYYMMDDHHMM")
    if args.command != "bc" and (not np.isfinite(args.friction_scale) or args.friction_scale < 0):
        parser.error("--friction-scale must be finite and nonnegative")
    if args.command == "evaluate" and args.episodes <= 0:
        parser.error("--episodes must be positive")
    for key in ("config", "demos", "init", "policy"):
        if key in vars(args) and not getattr(args, key).is_file():
            parser.error(f"--{key} must name an existing file")
    args.gpu, args.gripper_scale = 0, 1.0
    return args, load_config(args.config)


def main(argv=None):
    """Run the requested CPU learner or simulator-backed collection stage."""
    args, cfg = parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg["seed"])
    if args.command == "bc":
        return bc(args, cfg)
    with simulator_pool(args, cfg) as pool:
        return {"demos": demos, "evaluate": evaluate, "train": train}[args.command](args, cfg, pool)


if __name__ == "__main__":
    main()
