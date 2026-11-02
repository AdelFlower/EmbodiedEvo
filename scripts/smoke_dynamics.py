"""Start the research source simulator and hold its joints once without a model."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from orchestrator.dynamics_repair import EXPERIMENT, environment_settings, load_config, start_run
from orchestrator.dynamics_session import DynamicsService
from orchestrator.files import run_lock, write_json
from orchestrator.resources import gpu_lease

HOLD_SOURCE = """print(get_instruction())
state = get_observation()['state']
action = {name: value for name, value in state.items() if name.endswith('joint_state')}
step(action)
"""


def run_smoke(config, run, gpu):
    result = {
        "experiment": EXPERIMENT,
        "command": "smoke",
        "purpose": "startup_validation",
        "status": "incomplete",
        "passed": False,
        "model_calls": 0,
        "formal_test": False,
    }
    service = None
    try:
        service = DynamicsService(
            run / "source", environment_settings(config), config["simulator_python"],
            "source", budget=1, episode_budget=1, target_budget=0, action_budget=1,
            execution_seconds=config["execution_timeout_sec"],
            startup_seconds=config["startup_timeout_sec"], fps=config["feedback_fps"],
            gpu=gpu, allow_reset=False,
        )
        result["startup"] = service.start("source")
        if result["startup"]["ok"]:
            feedback = service.handle({
                "op": "exec", "files": [{"path": "submission/smoke.py", "code": HOLD_SOURCE}],
            })
            result["feedback"] = feedback
            result["passed"] = bool(feedback["ok"] and not feedback.get("fatal")
                                    and feedback["native_steps_used"] == 1)
            if not result["passed"]:
                result["failure_classification"] = feedback.get("classification") or "unclassified"
        else:
            result["failure_classification"] = (
                result["startup"].get("classification") or "unclassified"
            )
    except (Exception, KeyboardInterrupt) as exc:
        result.update(error=str(exc), traceback=traceback.format_exc(),
                      failure_classification="unclassified", passed=False)
        if isinstance(exc, KeyboardInterrupt):
            raise
    finally:
        if service is not None:
            try:
                service.close()
            except Exception as exc:
                result.update(cleanup_error=str(exc), failure_classification="unclassified",
                              passed=False)
            result["session"] = service.status()
            if not service.valid:
                result.update(passed=False, failure_classification="unclassified")
        result["status"] = "passed" if result["passed"] else "incomplete"
        write_json(run / "smoke.json", result)
        write_json(run / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config["gpu"] = args.gpu
    with run_lock(args.run_dir), gpu_lease(args.gpu):
        run = start_run(args.run_dir, "smoke", config)
        result = run_smoke(config, run, args.gpu)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
