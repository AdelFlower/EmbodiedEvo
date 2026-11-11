"""Run-kind selection of the frozen DP observer, never exported as evidence."""

import numpy as np

from integrations.deployable.observation import Observer, actor_observe
from integrations.friction_rl.network import observe

DETECTION_MODES = {
    "baseline_source": "image", "baseline_target": "image", "probe": "image",
    "judgment": "image", "deploy": "image", "final_evaluation": "image",
    "training": "simulated", "source_evaluation": "simulated",
}


def policy_inputs(state, t, pad_cmd, detection):
    """Single-replica actor (21) and actor ⊕ privileged observe (54)."""
    actor = actor_observe(t, pad_cmd, state["sample"]["gripper_opening"]["right"], detection)
    return actor, np.concatenate((actor, observe(state, t, pad_cmd)))


def make_observer(run_kind, calibration, rng):
    if run_kind not in DETECTION_MODES:
        raise ValueError("unknown cube-to-plate run kind")
    return Observer(DETECTION_MODES[run_kind], calibration, rng)
