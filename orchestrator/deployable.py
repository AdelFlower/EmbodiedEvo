"""Calibrate, prepare and train a deployable cube-to-plate neural policy."""

import argparse
import json
from pathlib import Path
import re
import sys
from threading import Lock
import time
import traceback

import imageio.v2 as imageio
import numpy as np
import torch

from integrations.deployable.detector import (apply_homography, centroids, fit_homography,
                                              sigma_rule)
from integrations.deployable.observation import ACTOR_OBS_DIM, CRITIC_OBS_DIM, Observer
from integrations.deployable.reward import reward
from integrations.deployable.script import DetectionScriptController
from integrations.friction_rl.network import ACT_DIM, run_net_episode
from integrations.friction_rl.ppo import ActorCritic, act, bc_fit, optimizers, ppo_update, save
from integrations.deployable.layout import build_scene
from integrations.friction_rl.simulation import SOURCE_PARAMETERS, start_pool
from orchestrator.experiment_jobs import save_training
from orchestrator.files import append_jsonl, write_json
from orchestrator.p0 import CONFIG_KEYS, INTEGER_KEYS, NUMBER_KEYS, VECTOR_SIZES, summary

PLATE_BOX = [[0., .03], [-.03, 0.]]
STREAMS = {"dev": (37, 20), "eval": (38, 50), "render_eval": (39, 20)}
TRAIN_INTS = ("episodes_per_iter", "warmup_iterations", "epochs", "minibatch",
              "max_iterations", "dev_every")
TRAIN_FLOATS = ("gamma", "gae_lambda", "clip", "actor_lr", "critic_lr", "target_kl", "bc_weight")


def load_config(path):
    """Validate the DP configuration once, before creating an attempt."""
    cfg = json.loads(Path(path).read_text())
    if type(cfg) is not dict or set(cfg) != CONFIG_KEYS | {"training", "calibration", "plate_box"}:
        raise ValueError("Config must contain exactly the deployable policy keys")
    training = cfg["training"]
    if type(training) is not dict or set(training) != set(TRAIN_INTS + TRAIN_FLOATS) | {"explore"}:
        raise ValueError("Invalid training keys")
    for values, integers, numbers in ((cfg, INTEGER_KEYS, NUMBER_KEYS),
                                       (training, TRAIN_INTS, TRAIN_FLOATS)):
        for key in integers:
            if type(values[key]) is not int or values[key] < (0 if key == "seed" else 1):
                raise ValueError(f"Invalid integer: {key}")
        for key in numbers:
            if type(values[key]) not in (int, float) or not np.isfinite(values[key]) or values[key] <= 0:
                raise ValueError(f"Invalid positive number: {key}")
    for key, size in VECTOR_SIZES.items():
        if (type(cfg[key]) is not list or len(cfg[key]) != size
                or any(type(x) not in (int, float) or not np.isfinite(x) for x in cfg[key])):
            raise ValueError(f"Invalid vector: {key}")
    for key in ("simulator_python", "calibration"):
        if type(cfg[key]) is not str or not cfg[key]:
            raise ValueError(f"{key} must be a nonempty path")
    script = json.loads((Path(__file__).resolve().parents[1]
                         / "tasks/DynamicsAdapt/p0.example.json").read_text())["script"]
    if type(cfg["script"]) is not dict or set(cfg["script"]) != set(script):
        raise ValueError("Invalid script keys")
    for key, value in cfg["script"].items():
        if type(value) not in ((int,) if key.startswith("t_") else (int, float)):
            raise ValueError(f"Invalid script value: {key}")
    fixed = {"layout_radius": .03, "cube_shift": [0, 0], "plate_box": PLATE_BOX, "script_force": 15.,
             "demo_episodes": 64, "demo_noise": [.002, .002, .001], "demo_noise_rho": .95,
             "bc_steps": 20000, "bc_batch": 256, "bc_lr": .001,
             "init_std": [.02, .02, .02, .1, .25], "script": script}
    registered = {"episodes_per_iter": 8, "warmup_iterations": 2, "gamma": .99,
                  "gae_lambda": .95, "clip": .2, "epochs": 8, "minibatch": 512,
                  "actor_lr": .0002, "critic_lr": .0005, "target_kl": .04, "bc_weight": 1.}
    if (any(cfg[key] != value for key, value in fixed.items()) or cfg["num_envs"] > 2
            or any(training[key] != value for key, value in registered.items())
            or training["explore"] not in ([0, 1, 2, 3, 4], [3, 4])
            or any(type(x) is not int for x in training["explore"])
            or training["max_iterations"] > 80):
        raise ValueError("Configuration differs from the registered DP settings")
    return cfg


def episode_seed(cfg, stream, k):
    return int(np.random.SeedSequence([cfg["seed"], stream, k]).generate_state(1)[0])


def policy(path=None):
    model = ActorCritic(ACTOR_OBS_DIM, ACT_DIM, CRITIC_OBS_DIM)
    if path is not None:
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def collect(args, cfg, pool, calibration, stream, indices, name, model=None,
            stochastic=False, detector="simulated", demos=False):
    """Retain episodes with independent layout and shared detection/action RNGs."""
    lock = Lock()

    def episode(client, k):
        rng = np.random.default_rng([cfg["seed"], stream, k, 1])
        observer = Observer(detector, calibration, rng)
        if model is None:
            noise = cfg["demo_noise"] if demos and k >= 8 else np.zeros(3)
            controller = DetectionScriptController(observer, cfg["script"], cfg["script_force"],
                                                   noise, cfg["demo_noise_rho"], rng)
        else:
            controller = lambda t, obs, state, decoder: act(
                model, obs, rng, stochastic, explore=cfg["training"]["explore"])
        folder = client.run_dir / f"{name}_e{k:04d}"
        result = run_net_episode(client, build_scene(episode_seed(cfg, stream, k), cfg),
                                 folder, controller, observer=observer)
        if stream == 31:
            result[2]["reward"] = reward(result[1])
            np.save(folder / "rewards.npy", result[2]["reward"])
        with lock:
            append_jsonl(args.run_dir / "episodes.jsonl", {
                "stream": stream, "k": k, "episode_dir": str(folder), **result[0]})
        return result

    return pool.map(episode, list(indices))


def evaluate(args, cfg, pool, calibration):
    stream, count = STREAMS[args.stream]
    results = collect(args, cfg, pool, calibration, stream, range(count), "eval",
                      None if args.script else policy(args.policy), args.stochastic, args.detector)
    result = summary([r for r, _, _ in results])
    result["detection_errors"] = [r["observer"]["error"] for r, _, _ in results]
    if args.script and args.stream == "dev":
        result["gate_passed"] = result["strict_successes"] >= 18
    write_json(args.run_dir / "summary.json", result)
    return result


def calibrate(args, cfg, pool):
    def reset(client, item):
        stream, k = item
        folder = client.run_dir / f"cal_s{stream}_e{k:02d}"
        state = client.reset(build_scene(episode_seed(cfg, stream, k), cfg), folder)["state"]
        truth = [state["sample"]["cube_position"][:2], state["plate_position"][:2]]
        return imageio.imread(folder / "reset_cam_head.png"), truth

    fit = pool.map(reset, [(20, k) for k in range(40)])
    points = [centroids(frame) for frame, _ in fit]
    if any(point is None for point in points):
        raise ValueError("Calibration fit detection failed")
    points, truth = np.array(points), np.array([xy for _, xy in fit])
    matrices = [fit_homography(points[:, j], truth[:, j]) for j in range(2)]
    validation = pool.map(reset, [(21, k) for k in range(40)])
    errors, records = [], []
    for k, (frame, xy) in enumerate(validation):
        point = centroids(frame)
        error = None if point is None else np.array([
            apply_homography(matrices[j], point[j]) - xy[j] for j in range(2)])
        records.append({"k": k, "error": None if error is None else error.tolist()})
        if error is not None:
            errors.append(error)
    p95 = np.percentile(np.linalg.norm(errors, axis=2), 95, axis=0) if errors else None
    result = {"detections": len(errors), "N": 40, "validation": records,
              "p95": None if p95 is None else p95.tolist(),
              "gate_passed": len(errors) == 40 and bool(np.all(p95 <= [.004, .008]))}
    calibration = {"homographies": np.asarray(matrices).tolist(),
                   "sigma": sigma_rule(errors).tolist() if errors else None, **result}
    write_json(args.run_dir / "calibration.json", calibration)
    return result


def demos(args, cfg, pool, calibration):
    results = collect(args, cfg, pool, calibration, 30, range(cfg["demo_episodes"]), "demo", demos=True)
    data = {key: np.concatenate([b[key] for _, _, b in results]) for key in ("obs", "critic_obs")}
    np.savez(args.run_dir / "demos.npz", **data,
             labels=np.concatenate([b["stored"] for _, _, b in results]),
             episode=np.concatenate([np.full(len(b["obs"]), k) for k, (_, _, b) in enumerate(results)]))
    return {**summary([r for r, _, _ in results]),
            "gate_passed": all(r["strict_success"] for r, _, _ in results[:8])}


def bc(args, cfg):
    model = policy()
    with np.load(args.demos) as data:
        records = bc_fit(model, data["obs"], data["labels"], cfg["bc_steps"], cfg["bc_batch"],
                         cfg["bc_lr"], cfg["seed"], cfg["init_std"])
    for record in records:
        append_jsonl(args.run_dir / "bc.jsonl", record)
    save(args.run_dir / "policy.pt", model)
    return {"steps": cfg["bc_steps"], "loss": records[-1]["loss"]}


def train(args, cfg, pool, calibration):
    training, model = cfg["training"], policy(args.policy)
    actor, critic = optimizers(model, training)
    with np.load(args.demos) as data:
        prior = {key: data[key] for key in ("obs", "labels")}
    selected = None

    def development(iteration):
        nonlocal selected
        rows = collect(args, cfg, pool, calibration, 37, range(20), f"dev_it{iteration:03d}", model)
        candidate = {"iteration": iteration, "policy": f"policy_it{iteration:03d}.pt",
                     **summary([r for r, _, _ in rows])}
        append_jsonl(args.run_dir / "dev.jsonl", candidate)
        rank = lambda row: (row["strict_successes"], row["successes"])
        if selected is None or rank(candidate) > rank(selected):
            selected = candidate
        write_json(args.run_dir / "selection.json", selected)
        return candidate

    save(args.run_dir / "policy_it000.pt", model)
    save_training(args.run_dir / "training.pt", model, actor, critic, 0, 0)
    baseline = development(0)
    online = collect(args, cfg, pool, calibration, 31, range(8), "online_it000", model, True)
    online_rate = sum(r["strict_success"] for r, _, _ in online) / 8
    write_json(args.run_dir / "bc_baseline.json", {"development": baseline, "online_rate": online_rate})
    if baseline["strict_successes"] < 6 and online_rate < .3:
        return {"gate_passed": False, "reason": "BC below development and online gates"}
    consecutive = 0
    for i in range(training["max_iterations"]):
        results = collect(args, cfg, pool, calibration, 31, range((i+1)*8, (i+2)*8),
                          f"train_it{i+1:03d}", model, True)
        batch = {key: np.concatenate([b[key] for _, _, b in results])
                 for key in ("obs", "critic_obs", "stored", "reward", "done")}
        warmup = i < training["warmup_iterations"]
        metrics = ppo_update(model, batch, prior, training, actor, critic,
                             actor_frozen=warmup, explore=training["explore"])
        outcomes = summary([r for r, _, _ in results])
        append_jsonl(args.run_dir / "iterations.jsonl", {
            "iteration": i+1, "warmup": warmup, **outcomes, **metrics,
            "mean_return": float(sum(b["reward"].sum() for _, _, b in results) / 8)})
        save(args.run_dir / f"policy_it{i+1:03d}.pt", model)
        save_training(args.run_dir / "training.pt", model, actor, critic, i+1, (i+1)*8)
        if (i+1) % training["dev_every"] == 0:
            development(i+1)
        below = not warmup and online_rate - outcomes["strict_successes"] / 8 > .25
        consecutive = consecutive + 1 if below else 0
        if consecutive == 3 and training["explore"] == [0, 1, 2, 3, 4]:
            return {"stopped": "online strict success regression", "fallback_required": True,
                    "iteration": i+1, "selection": selected}
    return {"selection": selected, "bc_online_rate": online_rate,
            "strict_difference": selected["strict_successes"] - baseline["strict_successes"],
            "lenient_difference": selected["successes"] - baseline["successes"]}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "demos", "bc", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--run-dir", required=True, type=Path)
        if name != "bc":
            command.add_argument("--gpu", type=int, default=0)
        if name in ("demos", "train", "evaluate"):
            command.add_argument("--calibration", type=Path, required=name == "train")
        if name in ("bc", "train"):
            command.add_argument("--demos", type=Path, required=True)
        if name == "train":
            command.add_argument("--policy", type=Path, required=True)
        if name == "evaluate":
            policy_args = command.add_mutually_exclusive_group(required=True)
            policy_args.add_argument("--policy", type=Path)
            policy_args.add_argument("--script", action="store_true")
            command.add_argument("--stream", choices=STREAMS, default="dev")
            command.add_argument("--detector", choices=("simulated", "image"), default="simulated")
            command.add_argument("--stochastic", action="store_true")
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    if not re.match(r"^\d{12}(?:_|$)", args.run_dir.name) or args.run_dir.exists():
        parser.error("--run-dir must be new and start with YYYYMMDDHHMM")
    if args.command != "bc" and args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    cfg = load_config(args.config)
    if args.command == "evaluate" and args.detector == "simulated" and args.calibration is None:
        parser.error("--detector simulated requires --calibration")
    if args.command in ("demos", "evaluate") and args.calibration is None:
        args.calibration = Path(cfg["calibration"])
    for key in ("config", "demos", "policy", "calibration"):
        value = vars(args).get(key)
        if value is not None and not value.is_file():
            parser.error(f"--{key} must name an existing file")
    return args, cfg


def execute(args, cfg):
    """Retain terminal status even on failure, without catching execution errors."""
    args.run_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.run_dir / "config.json", cfg)
    write_json(args.run_dir / "run.json", {
        "command": args.command, "parameters": SOURCE_PARAMETERS, "started_at": time.time(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    torch.manual_seed(cfg["seed"])
    record, started = {"status": "failed"}, time.monotonic()
    try:
        if args.command == "bc":
            result = bc(args, cfg)
        else:
            calibration = None if args.command == "calibrate" else json.loads(args.calibration.read_text())
            if calibration is not None and not calibration["gate_passed"]:
                raise ValueError("Calibration gate failed; stop and consult the user")
            mode = "render" if args.command == "calibrate" or (
                args.command == "evaluate" and args.detector == "image") else "state"
            with start_pool(args.run_dir, cfg, SOURCE_PARAMETERS, mode, args.gpu) as pool:
                result = calibrate(args, cfg, pool) if args.command == "calibrate" else {
                    "demos": demos, "train": train, "evaluate": evaluate}[args.command](args, cfg, pool, calibration)
        record.update(status="stopped" if result.get("fallback_required") else "complete", result=result)
        if not result.get("gate_passed", True):
            record["status"] = "gate_failed"
            raise SystemExit("DP gate failed; stop and consult the user")
        return result
    finally:
        if record["status"] == "failed" and sys.exception() is not None:
            record.update(error=str(sys.exception()), traceback=traceback.format_exc())
        write_json(args.run_dir / "result.json", {**record, "elapsed_seconds": time.monotonic() - started})


def main(argv=None):
    args, cfg = parse_args(argv)
    result = execute(args, cfg)
    if result.get("fallback_required"):
        original = args.run_dir
        args = argparse.Namespace(**{**vars(args), "run_dir": original.with_name(original.name + "_gripper")})
        cfg = {**cfg, "training": {**cfg["training"], "explore": [3, 4]}}
        write_json(original / "fallback.json", {"reason": result["stopped"], "run_dir": str(args.run_dir),
                                                "policy": str(args.policy), "explore": [3, 4]})
        result = execute(args, cfg)
    return result


if __name__ == "__main__":
    main()
