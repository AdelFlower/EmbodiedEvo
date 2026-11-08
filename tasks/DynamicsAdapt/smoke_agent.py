"""Exercise the host-only smoke path without making a scientific diagnosis claim."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "scripts"))
from adapt import request


def main():
    """Check the gate, reproduce one known failure, train four iterations and submit."""
    parameters = {"friction_scale": .25, "mass_scale": 1., "restitution": 0.}
    train = {"op": "train", "parameters": parameters, "iterations": 4,
             "reward": "default", "rationale": "Test-only smoke round"}
    refused = request(train)
    assert not refused["ok"] and "Reproduce" in refused["error"], refused
    status = request({"op": "status"})
    episode = status["networks"][0]["strict_outcomes"].index(False)
    probe = request({"op": "probe", "network": "N000", "parameters": parameters, "episodes": [episode],
                     "hypothesis": "Test-only supplied physics", "prediction": "Reproduce the failure"})
    assert probe["ok"], probe
    while request({"op": "wait", "job": probe["job"], "seconds": 900})["job"]["status"] == "running":
        pass
    trained = request(train)
    assert trained["ok"], trained
    while request({"op": "wait", "job": trained["job"], "seconds": 900})["job"]["status"] == "running":
        pass
    diagnosis = {"estimated_target_parameters": parameters, "conclusion": "Smoke test; no diagnosis claim",
                 "evidence_jobs": [probe["job"], trained["job"]], "competing_hypotheses": [],
                 "limitations": "True parameters supplied by the host test script", "preferred_network": "N000",
                 "reward_rationale": "Exercise the supplied default reward"}
    Path("submission").mkdir(exist_ok=True)
    Path("submission/diagnosis.json").write_text(json.dumps(diagnosis, indent=2))
    submitted = request({"op": "submit", "diagnosis": diagnosis})
    assert submitted["ok"], submitted


if __name__ == "__main__":
    main()
