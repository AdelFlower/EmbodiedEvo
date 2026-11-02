"""Run retained rollout and training jobs for dynamics adaptation."""

import argparse
import json
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

from integrations.dynamics_rl.client import RemoteEnv
from integrations.dynamics_rl.data import round_layout_seeds, round_seed
from integrations.dynamics_rl.learning import load_policy
from orchestrator.dynamics_rl import ROOT, run_episode, simulator
from orchestrator.files import write_json


def start_job(job_dir, spec):
    """Persist a trusted specification and launch an independent job process group."""
    job_dir = Path(job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=False)
    write_json(job_dir / "spec.json", spec)
    with (job_dir / "job.log").open("x") as log:
        return subprocess.Popen(
            [sys.executable, "-m", "orchestrator.adaptation_jobs", str(job_dir)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )


def run_job(job_dir, spec):
    """Run a new job synchronously, preserving its exception and result on failure."""
    job_dir = Path(job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=False)
    write_json(job_dir / "spec.json", spec)
    return execute_job(job_dir)


def _episode(client, item, policy, episodes, index):
    """Attach a layout to the existing episode record, including a partial attempt."""
    name = f"E{item['index']:03d}"
    folder = client.run_dir / name
    try:
        record = run_episode(client, item["scene"], name, policy=policy)
    except BaseException:
        record = json.loads((folder / "episode-summary.json").read_text())
        episodes[index] = {**record, "layout": item["index"]}
        raise
    episodes[index] = {"error": None, **record, "layout": item["index"]}


def _rollout(job_dir, spec, result):
    """Stop on the first exception and retain every planned layout in the result."""
    episodes = result["episodes"] = [
        {"layout": item["index"], "success": False, "complete": False,
         "not_started": True, "actions": 0, "error": None}
        for item in spec["layouts"]
    ]
    policy = load_policy(spec["network"])
    # simulator() derives its worker run-dir prefix from this timestamped parent.
    simulations = job_dir / time.strftime("%Y%m%d%H%M_simulations")
    with ExitStack() as stack:
        if not spec["fresh_simulator_per_episode"]:
            client = stack.enter_context(simulator(
                simulations, "rollout", spec, spec["gpu"], "render",
                parameters=spec["parameters"],
            ))
        for index, item in enumerate(spec["layouts"]):
            if spec["fresh_simulator_per_episode"]:
                with simulator(simulations, f"E{item['index']:03d}", spec, spec["gpu"],
                               "render", parameters=spec["parameters"]) as client:
                    _episode(client, item, policy, episodes, index)
            else:
                _episode(client, item, policy, episodes, index)


def _train(job_dir, spec, result):
    """Continue PPO with fresh rollouts and the round's shared seeds."""
    import torch

    from integrations.dynamics_rl.adaptation import load_round_model, save_network, train_round

    torch.set_num_threads(spec["training"]["learner_threads"])
    output = Path(spec["output"])
    seeds = round_layout_seeds(spec["seed"], spec["round"], spec["actions"])
    write_json(job_dir / "online-layout-seeds.json", seeds)
    episodes = []
    simulations = job_dir / time.strftime("%Y%m%d%H%M_simulations")
    with simulator(simulations, "train", spec, spec["gpu"],
                   parameters=spec["parameters"]) as client, ExitStack() as stack:
        env = RemoteEnv(client, scene_seeds=seeds)
        stack.callback(env.close)
        model = load_round_model(
            spec["parent"], env, spec["training"],
            round_seed(spec["seed"], spec["round"]),
        )
        initial_steps = model.num_timesteps
        train_round(model, spec["actions"], on_episode=episodes.append,
                    run_dir=output,
                    checkpoint_interval=spec["training"]["checkpoint_interval"])
        metadata = {
            "id": output.name, "parent": Path(spec["parent"]).parent.name,
            "round": spec["round"], "parameters": spec["parameters"],
            "round_actions": model.num_timesteps - initial_steps,
        }
        save_network(model, output, metadata)
        result.update(
            round_actions=model.num_timesteps - initial_steps,
            online_steps=int(model.num_timesteps), policy_updates=model.policy_updates,
            online_episodes=len(episodes),
            online_successes=sum(row["complete"] and row["success"] for row in episodes),
        )


def execute_job(job_dir):
    """Execute a persisted job, writing exactly one terminal job record."""
    job_dir = Path(job_dir)
    started = time.time()
    result = {"status": "failed"}
    try:
        spec = json.loads((job_dir / "spec.json").read_text())
        {"rollout": _rollout, "train": _train}[spec["kind"]](job_dir, spec, result)
        result["status"] = "complete"
    except BaseException as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        if "episodes" in result:
            # Write after worker shutdown, which also writes the last episode result.
            for record in result["episodes"]:
                if "episode_dir" in record:
                    write_json(Path(record["episode_dir"]) / "result.json", record)
            result["successes"] = sum(row["complete"] and row["success"]
                                      for row in result["episodes"])
        result["elapsed_seconds"] = time.time() - started
        write_json(job_dir / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    args = parser.parse_args(argv)
    execute_job(args.job_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
