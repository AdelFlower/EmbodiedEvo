"""Test-only host script exercising the public adaptation client without a model call."""

import json
import runpy
from pathlib import Path


request = runpy.run_path("scripts/adapt.py")["request"]


def call(payload):
    response = request(payload)
    print(json.dumps(response), flush=True)
    assert response["ok"], response
    return response


def complete(payload):
    job = call(payload)["job"]
    while True:
        status = call({"op": "wait", "job": job, "seconds": 900})["job"]["status"]
        if status != "running":
            assert status == "complete", status
            break
    return job


refused = request({"op": "train", "parameters": {"friction_scale": 0.25}, "actions": 2000,
                   "rationale": "Must be refused before reproduction"})
print(json.dumps(refused), flush=True)
assert not refused["ok"], refused
layout = next(record["layout"] for path in sorted(Path("observations/baseline/target").glob("L*/result.json"))
              if not ((record := json.loads(path.read_text()))["complete"] and record["success"]))
# The smoke deliberately uses the true target friction so reproduction is deterministic.
friction = complete({"op": "probe", "network": "N000", "parameters": {"friction_scale": 0.25},
                     "layouts": [layout], "hypothesis": "Lower friction permits slipping",
                     "prediction": "Cube motion relative to the hand increases during carry"})
mass = complete({"op": "probe", "network": "N000", "parameters": {"mass_scale": 2.0},
                 "layouts": [layout], "hypothesis": "Greater mass can also cause slipping",
                 "prediction": "Carry may resemble the friction probe; settling may differ"})
training = complete({"op": "train", "parameters": {"friction_scale": 0.25}, "actions": 2000,
                     "rationale": "Smoke test of continued training and automatic deployment"})
diagnosis = {
    "estimated_target_parameters": {"friction_scale": 0.25, "mass_scale": 1.0, "restitution": 0.0},
    "conclusion": "Scripted interface smoke; no scientific diagnosis is claimed",
    "evidence_jobs": [friction, training],
    "competing_hypotheses": [{"hypothesis": "Increased mass", "test_jobs": [mass],
                              "outcome": "not_rejected", "reason": "Requires researcher inspection"}],
    "limitations": "The script does not analyze physical evidence", "preferred_network": "N000",
}
Path("submission/diagnosis.json").write_text(json.dumps(diagnosis, indent=2))
call({"op": "submit", "diagnosis": diagnosis})
