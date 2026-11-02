"""Episode transition records and deterministic train/validation scene splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from integrations.dynamics_rl.features import OBS_DIM, sample_scene

FIELDS = ("obs", "actions", "next_obs", "rewards", "terminated", "truncated")


def empty_data():
    return {
        "obs": np.empty((0, OBS_DIM), np.float32),
        "actions": np.empty((0, 7), np.float32),
        "next_obs": np.empty((0, OBS_DIM), np.float32),
        "rewards": np.empty(0, np.float32),
        "terminated": np.empty(0, bool), "truncated": np.empty(0, bool),
    }


class EpisodeData:
    def __init__(self):
        self.rows = []

    def append(self, previous, action, response):
        self.rows.append((np.array(previous["obs"], copy=True), np.array(action, copy=True),
                          np.array(response["obs"], copy=True), response["reward"],
                          response["terminated"], response["truncated"]))

    def arrays(self):
        if not self.rows:
            return empty_data()
        columns = zip(*self.rows, strict=True)
        return {name: np.asarray(values, dtype=bool if name in {"terminated", "truncated"}
                                else np.float32)
                for name, values in zip(FIELDS, columns, strict=True)}

    def save(self, path):
        np.savez_compressed(path, **self.arrays())


def concatenate(datasets):
    datasets = list(datasets)
    if not datasets:
        return empty_data()
    return {name: np.concatenate([data[name] for data in datasets]) for name in FIELDS}


def layout_seeds(split, count, seed):
    """Disjoint RNG streams, while native physics always keeps its source seed=0."""
    stream = {"demonstration": 11, "online": 23, "validation": 37,
              "evaluation": 53, "rollout": 67, "adaptation_dev": 71,
              "adaptation_online": 73, "adaptation_eval": 79, "trainability_eval": 83,
              "fix_demonstration": 101, "fix_online": 103, "fix_dev": 107, "fix_eval": 109}[split]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), stream]))
    # Namespaces make cross-split seed overlap impossible, not just unlikely.
    seeds = rng.choice(2**40, size=count, replace=False).astype(np.uint64)
    seeds += np.uint64(stream << 48)
    return seeds.tolist()


def round_seed(seed, round_index):
    """Share the learner and online-layout seed across adaptation branches."""
    return 1000 * seed + round_index


def round_layout_seeds(seed, round_index, round_actions):
    """Allow one reset per action plus the initial reset in every branch."""
    return layout_seeds("adaptation_online", round_actions + 1, round_seed(seed, round_index))


def scene_plan(split, count, seed, radius=0.03):
    seeds = layout_seeds(split, count, seed)
    return [{"index": index, "layout_seed": int(value), "scene": sample_scene(value, radius)}
            for index, value in enumerate(seeds)]


def import_demonstration_prefix(source, destination, plan, config, *, teacher_version,
                                observation_contract, teacher_archive):
    """Reference an unchanged prefix of prior attempts, including every failure.

    This resumes data collection only. It neither loads network weights nor retries
    an attempted scene. The caller supplies the remaining wall-time budget.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Demonstration inheritance needs a separate new run directory")
    source_config = json.loads((source / "config.json").read_text())
    for key in ("demonstration_attempts", "seed", "position_radius"):
        if source_config[key] != config[key]:
            raise ValueError(f"Demonstration source has different {key}")
    if json.loads((source / "demonstration-scenes.json").read_text()) != plan:
        raise ValueError("Demonstration source scene plan differs from the requested plan")
    if json.loads((source / "contract.json").read_text()) != observation_contract:
        raise ValueError("Demonstration source observation/action contract differs")
    provenance = json.loads((source / "teacher-provenance.json").read_text())
    if provenance.get("adaptation") != teacher_version:
        raise ValueError("Demonstration source teacher adaptation differs")
    if json.loads((source / "supplied-teacher.json").read_text()) != teacher_archive:
        raise ValueError("Demonstration source supplied teacher policy differs")
    records = json.loads((source / "demonstrations.json").read_text())
    if not isinstance(records, list) or len(records) > len(plan):
        raise ValueError("Demonstration source must be a prefix of the requested attempts")
    inherited = []
    for index, record in enumerate(records):
        if record.get("layout_seed") != plan[index]["layout_seed"]:
            raise ValueError(f"Demonstration source attempt {index} is out of plan order")
        path = (source / record["data_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Demonstration source transition artifact is missing: {path}")
        # Reading the transition archive also makes an empty failed attempt explicit:
        # it remains in the manifest but contributes zero samples to either learner.
        with np.load(path, allow_pickle=False) as archive:
            count = len(archive["obs"])
            if any(len(archive[name]) != count for name in FIELDS):
                raise ValueError(f"Demonstration source attempt {index} has inconsistent arrays")
        inherited.append({**record, "data_path": str(path)})
    inheritance = {
        "source_run": str(source), "stage": "demonstration_collection_only",
        "inherited_attempts": len(inherited),
        "inherited_successes": sum(bool(row["success"] and row["complete"])
                                   for row in inherited),
        "inherited_incomplete": sum(not row["complete"] for row in inherited),
        "next_plan_index": len(inherited), "planned_attempts": len(plan),
        "retry_failed_attempts": False, "model_checkpoint_loaded": False,
        "remaining_time_budget_hours": config["max_train_hours"],
    }
    for name, value in (("demonstration-inheritance.json", inheritance),
                        ("demonstrations.json", inherited)):
        (destination / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return inherited


def load_demonstrations(root):
    root = Path(root)
    records = json.loads((root / "demonstrations.json").read_text())
    all_data, successful = [], []
    for record in records:
        path = root / record["data_path"]
        with np.load(path, allow_pickle=False) as archive:
            data = {name: archive[name] for name in FIELDS}
        all_data.append(data)
        if record["success"] and record["complete"]:
            successful.append(data)
    return concatenate(all_data), concatenate(successful)
