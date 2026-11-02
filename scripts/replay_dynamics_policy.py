"""Replay an archived frozen policy in fresh source and target simulators, without a model.

This is a diagnostic: it never starts an agent, never selects or tunes a perturbation and
never qualifies an experiment. Each kind runs the experiment's own three fresh frozen
replays at the configured target_friction_scale, and the result reports whether the
experiment's qualification thresholds would have been met.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orchestrator.dynamics_repair import (
    EXPERIMENT,
    REPETITIONS,
    archive_policy,
    environment_settings,
    load_config,
    read_json,
    replay_batch,
    start_run,
)
from orchestrator.files import run_lock, write_json
from orchestrator.resources import gpu_lease


def refreeze(run, archive):
    """Freeze the archived program again under this run; it must be unchanged."""
    archive = Path(archive)
    original = read_json(archive / "policy.json")
    policy = archive_policy(
        archive / "submitted-policy.json", run / "policies" / original["version"],
        workspace=archive, version=original["version"],
    )
    if policy["files"] != original["files"]:
        raise ValueError("The re-frozen program differs from the archived policy")
    return policy


def replay(config, run, archive):
    """Re-freeze the archived program unchanged, then replay it per kind."""
    archive = Path(archive)
    policy = refreeze(run, archive)
    settings = environment_settings(config, config["target_friction_scale"])
    write_json(run / "settings.json", settings)
    result = {
        "experiment": EXPERIMENT, "command": "replay", "purpose": "diagnostic_replay",
        "formal_test": False, "model_calls": 0, "status": "running",
        "policy_archive": str(archive.absolute()), "policy": policy["version"],
        "target_friction_scale": config["target_friction_scale"],
    }
    write_json(run / "result.json", result)
    for kind in ("source", "target"):
        result[kind] = replay_batch(config, run / "replay" / kind, policy, settings, kind=kind)
        write_json(run / "result.json", result)
    source, target = result["source"], result["target"]
    result["thresholds"] = {
        "source_qualified": source["successes"] == REPETITIONS,
        "target_complete": target["incomplete"] == 0,
        "target_failures_established": target["incomplete"] == 0 and target["successes"] == 0,
    }
    result["status"] = "complete"
    write_json(run / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-archive", type=Path, required=True,
                        help="An archived policy directory, e.g. <run>/policies/P0")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config["gpu"] = args.gpu
    with run_lock(args.run_dir), gpu_lease(args.gpu):
        run = start_run(args.run_dir, "replay", config)
        result = replay(config, run, args.policy_archive)
    print(json.dumps({key: result[key] for key in ("status", "thresholds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
