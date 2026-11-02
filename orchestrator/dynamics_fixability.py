"""Run the source-policy and low-friction fixability study."""

import argparse
import json
import math
import re
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from integrations.dynamics_rl.client import RemoteEnv
from integrations.dynamics_rl.data import EpisodeData, layout_seeds, scene_plan
from integrations.dynamics_rl.features import ACTION_HIGH, ACTION_LOW
from integrations.dynamics_rl.fixability import (
    RewardScale, bc_train, load_actor, load_learner, save_network,
)
from integrations.dynamics_rl.learning import load_policy
from integrations.dynamics_rl.parallel import (
    ThreadedVecEnv, episode_records, run_episodes, run_parallel, simulator_pool,
)
from integrations.dynamics_rl.teacher import R1Teacher
from orchestrator.dynamics_rl import simulator
from orchestrator.files import append_jsonl, run_lock, write_json

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "tasks/DynamicsRL/fixability.example.json"
CARRY_MOVES = {"lift", "carry_and_tilt", "lower_to_plate"}


def load_config(path):
    """Validate the study configuration once at its external boundary."""
    defaults = json.loads(DEFAULT_CONFIG.read_text())
    config = dict(defaults)
    if path is not None:
        supplied = json.loads(Path(path).read_text())
        if not isinstance(supplied, dict) or set(supplied) - set(config):
            raise ValueError("Unknown configuration keys or non-object configuration")
        config.update(supplied)
    counts = {"num_envs", "learner_threads", "demonstration_attempts", "bc_updates", "bc_batch_size",
              "n_steps", "batch_size", "n_epochs", "value_warmup_rollouts", "rollouts_per_round",
              "max_rounds", "dev_episodes", "seed"}
    for key, default in defaults.items():
        value = config[key]
        if key in counts:
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(f"Invalid integer setting: {key}")
        elif type(default) in (int, float):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid positive setting: {key}")
        elif isinstance(default, str):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Invalid string setting: {key}")
    for key in ("gamma", "gae_lambda", "clip_range", "stop_online_success_rate"):
        if config[key] > 1:
            raise ValueError(f"{key} must be at most one")
    values = config["log_std"]
    if (not isinstance(values, list) or len(values) != 7 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in values)):
        raise ValueError("log_std needs seven finite log standard deviations")
    noise = config["demonstration_noise"]
    if (not isinstance(noise, list) or not noise or any(
            not isinstance(pair, list) or len(pair) != 2 or any(
                type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in pair)
            for pair in noise)):
        raise ValueError("demonstration_noise needs nonnegative pairs")
    if config["batch_size"] < 2 or config["num_envs"] * config["n_steps"] < 2:
        raise ValueError("Advantage normalization needs batches and rollouts larger than one")
    return config


def new_run(path, command, config, arguments):
    """Refuse populated directories and record the command's inputs once."""
    path = Path(path).resolve()
    if path.exists() and any(p.name != ".protocol.lock" for p in path.iterdir()):
        raise FileExistsError("A new run cannot overwrite existing evidence")
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "config.json", config)
    write_json(path / "run.json", {"experiment": "dynamics-fixability-v1", "command": command,
                                  "started_at": time.time(), **arguments})
    return path


def parameters(friction):
    return {"friction_scale": friction, "mass_scale": 1.0, "restitution": 0.0}


def named_plan(split, count, seed, name):
    return [{**item, "name": f"{name}/E{item['index']:03d}"}
            for item in scene_plan(split, count, seed)]


class FeasibilityTeacher(R1Teacher):
    """Scale exactly the three transport moves without changing teacher actions."""

    def __init__(self, state, scale):
        super().__init__(state)
        self.moves = tuple(replace(move, steps=round(move.steps * scale))
                           if move.name in CARRY_MOVES else move for move in self.moves)


def validate_carry_scale(scale):
    """Reject invalid move counts and native-budget overflow at the CLI boundary."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Carry scale must be positive and finite")
    teacher = FeasibilityTeacher({"sample": {"cube_position": [0, 0, 0]},
                                  "plate_position": [0, 0, 0]}, scale)
    if any(m.steps < 1 for m in teacher.moves):
        raise ValueError("Carry scale must produce positive step counts")
    if sum(m.steps for m in teacher.moves) > 500:
        raise ValueError("Scaled teacher exceeds 500 actions")


def noisy_action(clean, noise, rng):
    scale = np.r_[2 * noise[0] / (ACTION_HIGH - ACTION_LOW), np.full(3, noise[1] / 180)]
    action = clean.copy()
    action[:6] += rng.normal(size=6) * scale
    return np.clip(action, -1, 1)


def teacher_episode(client, item, folder, record, *, noise, carry_scale):
    """Record executed teacher controls and, for demonstrations, clean supervision."""
    data, labels, trace = EpisodeData(), [], []
    rng = np.random.default_rng(item["layout_seed"])
    record.update(layout_seed=item["layout_seed"], noise=noise, carry_time_scale=carry_scale)
    try:
        response = client.reset(item["scene"], folder)
        teacher = (R1Teacher(response["state"]) if noise is not None else
                   FeasibilityTeacher(response["state"], carry_scale))
        write_json(folder / "teacher.json", teacher.metadata())
        trace.append(response["state"]["sample"]["cube_position"])
        while not response["terminated"] and not response["truncated"]:
            clean = teacher.act(response["state"])
            action = clean if noise is None else noisy_action(clean, noise, rng)
            following = client.step(action)
            data.append(response, action, following)
            labels.append(clean)
            record["actions"] = len(data.rows)
            response = following
            trace.append(response["state"]["sample"]["cube_position"])
        record.update(complete=True, success=bool(response["info"]["success"]),
                      terminated=response["terminated"], truncated=response["truncated"])
    finally:
        folder.mkdir(parents=True, exist_ok=True)
        arrays = data.arrays()
        record["reward"] = float(arrays["rewards"].sum())
        if noise is not None:
            arrays["labels"] = np.asarray(labels, dtype=np.float32).reshape(-1, 7)
        record["data_path"] = str(folder / "transitions.npz")
        np.savez_compressed(folder / "transitions.npz", **arrays)
        np.save(folder / "cube-trajectory.npy", np.asarray(trace))


def demos(run, config, gpu):
    plan = named_plan("fix_demonstration", config["demonstration_attempts"], config["seed"], "demos")
    write_json(run / "demonstration-layouts.json", plan)
    records = episode_records(plan)

    def episode(client, item, folder, record):
        teacher_episode(client, item, folder, record,
                        noise=config["demonstration_noise"][item["index"] % len(config["demonstration_noise"])],
                        carry_scale=1)

    try:
        with simulator_pool(run, config, gpu, parameters(1.0), "state") as clients:
            run_parallel(clients, plan, records, episode)
    finally:
        write_json(run / "demonstrations.json", records)
    return {"attempts": len(records), "successes": sum(r["success"] for r in records)}


def bc(run, config, demonstration_run):
    records = json.loads((Path(demonstration_run) / "demonstrations.json").read_text())
    observations, labels = [], []
    for record in records:
        if record["complete"] and record["success"]:
            with np.load(Path(demonstration_run) / record["data_path"]) as data:
                observations.append(data["obs"])
                labels.append(data["labels"])
    model = bc_train(np.concatenate(observations), np.concatenate(labels), config, run_dir=run)
    save_network(model, run, {"id": "BC", "demonstrations": str(demonstration_run),
                              "training_config": config})
    return {"transitions": sum(len(obs) for obs in observations)}


def select_network(rows):
    return max(rows, key=lambda row: (row["dev_successes"], row["online_success_rate"], -row["round"]))


def train(run, config, gpu, init, friction):
    """Keep one pool alive through PPO rounds and deterministic development evaluation."""
    from stable_baselines3.common.logger import configure

    started = time.time()
    deadline = started + config["max_train_hours"] * 3600
    dev = scene_plan("fix_dev", config["dev_episodes"], config["seed"])
    write_json(run / "dev-layouts.json", dev)
    initial = Path(init) / "training.zip" if Path(init).is_dir() else Path(init)
    rows, status = [], "max_rounds"
    with simulator_pool(run, config, gpu, parameters(friction), "state") as clients:
        envs = [RemoteEnv(client) for client in clients]
        vector = ThreadedVecEnv(envs)
        scaled = RewardScale(vector, config["reward_scale"])
        try:
            model = load_learner(initial, scaled, config)
            model.run_dir, model.started = run, started
            model.set_logger(configure(format_strings=[]))
            for round_index in range(config["max_rounds"] + 1):
                if round_index and time.time() >= deadline:
                    status = "max_train_hours"
                    break
                name, episodes, rollouts = f"R{round_index:02d}", [], 0

                def finished(record):
                    episodes.append(record)
                    model.rollout_episodes.append(record)
                    append_jsonl(run / "online-episodes.jsonl", {"round": round_index, **record})

                if round_index:
                    count = config["rollouts_per_round"] * config["n_steps"] + 1
                    seeds = [layout_seeds("fix_online", count, int(np.random.SeedSequence(
                        [config["seed"], round_index, k]).generate_state(1)[0])) for k in range(len(envs))]
                    write_json(run / f"online-layouts-{name}.json", seeds)
                    model.rollout_episodes.clear()
                    for k, env in enumerate(envs):
                        env.scene_seeds, env.episode_index = seeds[k], 0
                        env.episode_root = clients[k].run_dir / "online" / name
                        env.on_episode = finished
                    for _ in range(config["rollouts_per_round"]):
                        if time.time() >= deadline:
                            status = "max_train_hours"
                            break
                        model.learn(config["n_steps"] * len(envs), reset_num_timesteps=False)
                        rollouts += 1
                    for env in envs:
                        env._finish_episode(complete=False, error=("Unused vector auto-reset" if
                            env.episode_actions == 0 else "Round ended before episode termination"))
                    model._last_obs = None
                    if not rollouts:
                        break
                complete = [row for row in episodes if row["complete"]]
                successes = sum(row["success"] for row in complete)
                rate = successes / len(complete) if complete else (0.0 if round_index else -1.0)
                metadata = {"id": name, "round": round_index, "init": str(initial),
                            "parameters": parameters(friction), "training_config": config}
                save_network(model, run / "networks" / name, metadata)
                plan = [{**item, "name": f"dev/{name}/E{item['index']:03d}"} for item in dev]
                results = run_episodes(clients, plan, model, deterministic=True)
                row = {"id": name, "round": round_index, "online_episodes": len(complete),
                       "online_successes": successes, "online_success_rate": rate,
                       "incomplete_episodes": len(episodes) - len(complete), "rollouts": rollouts,
                       "dev_successes": sum(r["success"] for r in results), "dev_records": results,
                       "elapsed_seconds": time.time() - started}
                rows.append(row)
                append_jsonl(run / "rounds.jsonl", row)
                if time.time() >= deadline:
                    status = "max_train_hours"
                    break
                if (round_index and row["dev_successes"] == config["dev_episodes"] and
                        rate >= config["stop_online_success_rate"]):
                    status = "early_stop"
                    break
        finally:
            scaled.close()
    result = {"selected": select_network(rows)["id"], "table": rows, "status": status}
    write_json(run / "selection.json", result)
    return result


def evaluate(run, config, gpu, network, friction, episodes, stochastic=False, render=0):
    """Evaluate held-out paired layouts, then render a separate single-client replay."""
    network = Path(network)
    policy = (load_actor(network / "policy.pt", seed=config["seed"]) if network.is_dir()
              else load_policy(network))
    plan = named_plan("fix_eval", episodes, config["seed"], "evaluation")
    write_json(run / "evaluation-layouts.json", plan)
    started = time.time()
    with simulator_pool(run, config, gpu, parameters(friction), "state") as clients:
        records = run_episodes(clients, plan, policy, deterministic=not stochastic)
    rendered = []
    if render:
        with simulator(run, "render", config, gpu, "render", parameters=parameters(friction)) as client:
            rendered = run_episodes([client], plan[:render], policy, deterministic=not stochastic)
    result = {"records": records, "successes": sum(r["success"] for r in records),
              "episodes": episodes, "mode": "stochastic" if stochastic else "deterministic",
              "friction_scale": friction, "rendered": rendered, "elapsed_seconds": time.time() - started}
    write_json(run / "evaluation.json", result)
    return result


def feasibility(run, config, gpu, friction, carry_scale, episodes):
    started = time.time()
    plan = named_plan("fix_dev", episodes, config["seed"], "feasibility")
    write_json(run / "feasibility-layouts.json", plan)
    records = episode_records(plan)

    def episode(client, item, folder, record):
        teacher_episode(client, item, folder, record, noise=None, carry_scale=carry_scale)

    try:
        with simulator_pool(run, config, gpu, parameters(friction), "state") as clients:
            run_parallel(clients, plan, records, episode)
    finally:
        result = {"records": records, "successes": sum(r["success"] for r in records),
                  "friction_scale": friction, "carry_time_scale": carry_scale, "reference_only": True,
                  "elapsed_seconds": time.time() - started}
        write_json(run / "feasibility.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demos", "bc", "train", "evaluate", "feasibility"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--run-dir", type=Path, required=True)
        if name != "bc":
            command.add_argument("--gpu", type=int, default=0)
            command.add_argument("--friction-scale", type=float, required=True)
        if name == "bc":
            command.add_argument("--demonstrations", type=Path, required=True)
        if name == "train":
            command.add_argument("--init", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--network", type=Path, required=True)
            command.add_argument("--stochastic", action="store_true")
            command.add_argument("--render", type=int, default=0)
        if name in {"evaluate", "feasibility"}:
            command.add_argument("--episodes", type=int, default=50 if name == "evaluate" else 20)
        if name == "feasibility":
            command.add_argument("--carry-time-scale", type=float, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if not re.match(r"^\d{12}(?:_|$)", args.run_dir.name):
        parser.error("--run-dir must start with YYYYMMDDHHMM")
    if args.command != "bc" and (args.gpu < 0 or not math.isfinite(args.friction_scale) or
                                 not 0.05 <= args.friction_scale <= 2):
        parser.error("Invalid GPU or friction (expected [0.05, 2])")
    if args.command == "demos" and args.friction_scale != 1:
        parser.error("Demonstrations require friction 1.0")
    if args.command in {"evaluate", "feasibility"} and args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.command == "evaluate":
        if not 0 <= args.render <= args.episodes:
            parser.error("--render must lie between zero and --episodes")
        if args.stochastic and args.network.is_file():
            parser.error("P0 supports deterministic evaluation only")
    if args.command == "feasibility":
        validate_carry_scale(args.carry_time_scale)
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                 if k not in {"command", "config", "run_dir"}}
    with run_lock(args.run_dir):
        run = new_run(args.run_dir, args.command, config, arguments)
        try:
            if args.command == "demos":
                result = demos(run, config, args.gpu)
            elif args.command == "bc":
                result = bc(run, config, args.demonstrations)
            elif args.command == "train":
                result = train(run, config, args.gpu, args.init, args.friction_scale)
            elif args.command == "evaluate":
                result = evaluate(run, config, args.gpu, args.network, args.friction_scale,
                                  args.episodes, args.stochastic, args.render)
            else:
                result = feasibility(run, config, args.gpu, args.friction_scale,
                                     args.carry_time_scale, args.episodes)
        except BaseException as exc:
            write_json(run / "error.json", {"type": type(exc).__name__, "error": str(exc),
                                            "traceback": traceback.format_exc()})
            raise
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
