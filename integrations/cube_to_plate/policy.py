"""Fixed DP bundle boundary; incompatible manifests or weights are errors."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from integrations.friction_rl.ppo import ActorCritic

BUNDLE_MANIFEST = {
    "adapter": "deployable", "actor_obs": {"dim": 21, "layout": "dp_actor_v1"},
    "critic_obs": {"dim": 54, "layout": "dp_critic_v1"},
    "action": {"dim": 5, "decoder": "cube_to_plate_v1"},
    "architecture": {"hidden": [256, 256], "activation": "tanh", "log_std": True},
    "weights": "policy.pt", "training_state": None, "calibration": "calibration.json",
}


def validate_manifest(manifest):
    expected = deepcopy(BUNDLE_MANIFEST)
    if not isinstance(manifest, dict) or set(manifest) != set(expected):
        raise ValueError("incompatible DP bundle: exact manifest keys required")
    if manifest["training_state"] not in (None, "training.pt"):
        raise ValueError("incompatible DP bundle: training_state")
    expected["training_state"] = manifest["training_state"]
    # JSON canonicalization also distinguishes true/1 and integer/float dimensions.
    if json.dumps(manifest, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("incompatible DP bundle: fixed dimensions, layouts and architecture required")


def load_directory(directory, complete_training_state=None):
    directory = Path(directory)
    manifest = json.loads((directory / "policy_bundle.json").read_text())
    validate_manifest(manifest)
    if complete_training_state is not None and complete_training_state != (manifest["training_state"] is not None):
        raise ValueError("incompatible DP bundle: training-state declaration disagrees")
    for name in ("weights", "calibration", "training_state"):
        if manifest[name] is not None and not (directory / manifest[name]).is_file():
            raise ValueError(f"incompatible DP bundle: missing {name} file")
    model = ActorCritic(21, 5, 54)
    model.load_state_dict(torch.load(directory / manifest["weights"], map_location="cpu", weights_only=True),
                          strict=True)
    model.eval()
    return Policy(model, json.loads((directory / manifest["calibration"]).read_text()))


class Policy:
    def __init__(self, model, calibration):
        self.model, self.calibration = model, calibration

    def act(self, actor_batch, deterministic=True):
        if deterministic is not True:
            raise ValueError("host DP inference requires deterministic mean actions")
        actor_batch = np.asarray(actor_batch, dtype=np.float32)
        if actor_batch.ndim != 2 or actor_batch.shape[1] != 21:
            raise ValueError("DP actor input requires shape [B, 21]")
        with torch.no_grad():
            return self.model.actor(torch.as_tensor(actor_batch)).numpy()


class PolicyAdapter:
    def load(self, bundle):
        return load_directory(bundle.checkpoint_ref, bundle.complete_training_state)
