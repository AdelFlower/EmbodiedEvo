"""Run retained friction rollouts and PPO rounds with optimizer continuation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from integrations.friction_rl.network import ACT_DIM, OBS_DIM, run_net_episode
from integrations.friction_rl.ppo import ActorCritic, load, optimizers, ppo_update, save
from integrations.friction_rl.reward_sandbox import run_sandbox
from orchestrator.files import append_jsonl, write_json
from integrations.friction_rl.network import policy_controller
from integrations.friction_rl.simulation import build_scene, start_pool

from orchestrator.runtime import ROOT


def development_episodes(seed, indices=range(4)):
    """Define the four fixed development modes and independent random streams."""
    return [{"name": f"D{k}", "deterministic": k < 2, "seed": [seed, 7, k]} for k in indices]


def evaluation_episodes(cfg):
    """Define the held-out deterministic and stochastic evaluation episodes."""
    return [{"name": f"E{k}", "deterministic": k < cfg["eval_deterministic"], "seed": [cfg["seed"], 8, k]}
            for k in range(cfg["eval_deterministic"] + cfg["eval_stochastic"])]


def rollout_spec(cfg, network, parameters, episodes):
    """Describe a rendered rollout at the fixed scene."""
    return {"kind": "rollout", "config": cfg, "network": str(network),
            "parameters": parameters, "episodes": episodes}


def start_job(directory, spec):
    """Persist a job specification and launch its independent process group."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "spec.json", spec)
    with (directory / "job.log").open("x") as log:
        return subprocess.Popen([sys.executable, "-m", "orchestrator.experiment_jobs", str(directory)],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def run_job(directory, spec):
    """Execute one new synchronous host job with retained evidence."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "spec.json", spec)
    return execute_job(directory, spec)


def load_training(parent, cfg):
    """Restore both Adam states or create them for the initial policy."""
    parent = Path(parent)
    if parent.name == "policy.pt":
        model = load(parent)
        actor, critic = optimizers(model, cfg)
        return model, actor, critic, 0, 0
    checkpoint = torch.load(parent, map_location="cpu", weights_only=True)
    model = ActorCritic(OBS_DIM, ACT_DIM)
    model.load_state_dict(checkpoint["model"])
    actor, critic = optimizers(model, cfg)
    actor.load_state_dict(checkpoint["actor_optimizer"])
    critic.load_state_dict(checkpoint["critic_optimizer"])
    return model, actor, critic, checkpoint["iterations"], checkpoint["episodes"]


def save_training(path, model, actor, critic, iterations, episodes):
    """Save the model, both Adam states and cumulative training counters."""
    torch.save({"model": model.state_dict(), "actor_optimizer": actor.state_dict(),
                "critic_optimizer": critic.state_dict(), "iterations": iterations, "episodes": episodes}, path)


def needs_warmup(parent, reward_text, parent_reward_text):
    """Warm up the critic for N000 or a change of reward program text."""
    return parent == "N000" or reward_text != parent_reward_text


def _rollout(directory, spec, result):
    """Retain every planned episode, including those interrupted or never started."""
    cfg = spec["config"]
    episodes = result["episodes"] = [
        {"name": item["name"], "complete": False, "strict_success": False, "success_lenient": False,
         "record": None, "episode_dir": None} for item in spec["episodes"]]
    model = load(spec["network"])
    scene = build_scene(0, {**cfg, "layout_radius": 0})

    def episode(client, pair):
        item, row = pair
        row["episode_dir"] = str(client.run_dir / item["name"])
        record, _, _ = run_net_episode(client, scene, row["episode_dir"],
                                      policy_controller(model, item["seed"], not item["deterministic"]))
        row.update(complete=True, strict_success=record["strict_success"],
                   success_lenient=record["success"], record=record)
        return row

    simulations = directory / time.strftime("%Y%m%d%H%M_simulations")
    with start_pool(simulations, cfg, spec["parameters"], "render", cfg["gpu"]) as pool:
        pool.map(episode, list(zip(spec["episodes"], episodes)))


def _train(directory, spec, result):
    """Continue the parent PPO state with sandbox rewards and fixed source demonstrations."""
    cfg, training = spec["config"], spec["config"]["training"]
    torch.manual_seed(int(np.random.SeedSequence([cfg["seed"], 1, spec["round"]]).generate_state(1)[0]))
    model, actor, critic, iterations, episodes = load_training(spec["parent"], training)
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    reward_text = Path(spec["reward"]).read_text()
    if Path(spec["reward"]) != output / "reward.py":
        (output / "reward.py").write_text(reward_text)
    with np.load(spec["demos"]) as data:
        prior = {key: data[key] for key in ("obs", "labels")}
    scene = build_scene(0, {**cfg, "layout_radius": 0})
    simulations = directory / time.strftime("%Y%m%d%H%M_simulations")
    with start_pool(simulations, cfg, spec["parameters"], "state", cfg["gpu"]) as pool:
        for i in range(spec["iterations"]):
            started = time.monotonic()
            seeds = [[cfg["seed"], 1, spec["round"], i, j] for j in range(training["episodes_per_iter"])]

            def episode(client, item):
                j, seed = item
                folder = client.run_dir / f"it{i:03d}_E{j}"
                return folder, run_net_episode(client, scene, folder, policy_controller(model, seed, True))

            collected = pool.map(episode, list(enumerate(seeds)))
            folders, results = zip(*collected)
            rewards = run_sandbox(reward_text, {f"E{j}": trace for j, (_, trace, _) in enumerate(results)},
                                  directory / f"rewards_it{i:03d}", cfg["sandbox"])
            if not rewards["ok"]:
                raise ValueError(rewards["error"])
            for j, (_, _, batch) in enumerate(results):
                batch["reward"] = rewards["rewards"][f"E{j}"]
                np.save(folders[j] / "rewards.npy", batch["reward"])
            batch = {key: np.concatenate([b[key] for _, _, b in results])
                     for key in ("obs", "stored", "reward", "done")}
            warmup = spec["warmup"] and i < training["warmup_iterations"]
            update = ppo_update(model, batch, prior, training, actor, critic, actor_frozen=warmup)
            records = [record for record, _, _ in results]
            append_jsonl(output / "iterations.jsonl", {
                "iteration": i, "warmup": warmup, "strict_successes": sum(r["strict_success"] for r in records),
                "lenient_successes": sum(r["success"] for r in records),
                "mean_return": float(np.mean([b["reward"].sum() for _, _, b in results])),
                **{f"mean_{key}": float(np.mean([r[key] for r in records]))
                   for key in ("max_clear", "slip", "force_mean_held")},
                **{key: update[key] for key in ("approx_kl", "policy_loss", "value_loss")},
                "wall_time": time.monotonic() - started})
            save(output / f"policy_it{i:03d}.pt", model)
            iterations += 1
            episodes += len(results)
            result.update(last_iteration=list(map(str, folders)), iterations=iterations, episodes=episodes)
    save(output / "policy.pt", model)
    save_training(output / "training.pt", model, actor, critic, iterations, episodes)


def execute_job(directory, spec=None):
    """Write exactly one terminal result and retain failures with their traceback."""
    directory, started = Path(directory), time.monotonic()
    result = {"status": "failed"}
    try:
        if spec is None:
            spec = json.loads((directory / "spec.json").read_text())
        {"rollout": _rollout, "train": _train}[spec["kind"]](directory, spec, result)
        result["status"] = "complete"
    except BaseException as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(directory / "result.json", result)
    return result


def main(argv=None):
    """Execute the persisted job in the supplied directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    result = execute_job(parser.parse_args(argv).directory)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
