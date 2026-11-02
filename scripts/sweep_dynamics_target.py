"""Development calibration of the target friction scale with a frozen probe policy.

No model is called. The probe policy is an archived program, re-frozen unchanged. The
selection rule is fixed here, before any candidate is replayed:

1. The probe policy gets three fresh source replays and must succeed 3/3.
2. Candidates are replayed in ascending order, three fresh target replays each.
3. A candidate passes when all three replays are complete physical failures (0/3, none
   incomplete). The sweep stops at the first candidate that does not pass; it is never
   retried.
4. The selected scale is the highest passing candidate, so it and every lower candidate
   made the probe fail.

The selection shows only that this probe policy fails at the chosen scale. It does not
establish that the target is recoverable, and an experiment learns its own P0 anew. Freeze
the selected value in the experiment configuration before starting an experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orchestrator.dynamics_repair import (
    EXPERIMENT,
    REPETITIONS,
    environment_settings,
    load_config,
    replay_batch,
    start_run,
)
from orchestrator.files import run_lock, write_json
from orchestrator.resources import gpu_lease
from scripts.replay_dynamics_policy import refreeze

CANDIDATES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def sweep(config, run, archive, candidates=CANDIDATES):
    if list(candidates) != sorted(set(candidates)) or not all(0 < c < 1 for c in candidates):
        raise ValueError("Candidates must be distinct, ascending and strictly between 0 and 1")
    policy = refreeze(run, archive)
    result = {
        "experiment": EXPERIMENT, "command": "sweep", "purpose": "development_calibration",
        "formal_test": False, "model_calls": 0, "status": "running",
        "probe_policy_archive": str(Path(archive).absolute()), "policy": policy["version"],
        "candidates": list(candidates), "recoverability": "not_validated",
        "selected_target_friction_scale": None, "targets": {},
    }
    write_json(run / "result.json", result)
    settings = environment_settings(config, candidates[0])
    write_json(run / "source-settings.json", settings["source"])
    result["source"] = replay_batch(config, run / "replay/source", policy, settings, kind="source")
    write_json(run / "result.json", result)
    if result["source"]["successes"] != REPETITIONS:
        result["status"] = "probe_not_qualified_in_source"
        write_json(run / "result.json", result)
        return result
    result["status"] = "no_candidate_passed"
    for scale in candidates:
        settings = environment_settings(config, scale)
        directory = run / "replay" / f"target-{scale}"
        target = replay_batch(config, directory, policy, settings, kind="target")
        write_json(directory / "settings.json", settings["target"])
        passed = target["incomplete"] == 0 and target["successes"] == 0
        result["targets"][str(scale)] = {**target, "passed": passed}
        if passed:
            result["selected_target_friction_scale"] = scale
            result["status"] = "selected"
        write_json(run / "result.json", result)
        if not passed:
            result["stopped_at"] = scale
            break
    result["complete"] = True
    write_json(run / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-archive", type=Path, required=True,
                        help="An archived probe policy directory, e.g. <run>/policies/P0")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--candidates", type=float, nargs="+", default=list(CANDIDATES),
                        help="Ascending scales, fixed before the sweep and recorded in result.json")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config["gpu"] = args.gpu
    with run_lock(args.run_dir), gpu_lease(args.gpu):
        run = start_run(args.run_dir, "sweep", config)
        result = sweep(config, run, args.policy_archive, args.candidates)
    print(json.dumps({key: result.get(key) for key in (
        "status", "selected_target_friction_scale", "stopped_at")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
