"""Explicit deployable actor features and a privileged training-only critic."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from integrations.deployable.detector import image_detection, simulated_detection
from integrations.friction_rl.network import HORIZON, observe
from integrations.friction_rl.primitive import CUBE_HALF, TABLE_Z

ACTOR_OBS_DIM = 21
CRITIC_OBS_DIM = 54
PLATE_Z = .7828


def detection_positions(detection):
    """Attach nominal heights to the one-shot cube/plate XY detections."""
    return np.column_stack((detection, [TABLE_Z + CUBE_HALF, PLATE_Z]))


def actor_observe(t, pad_cmd, opening, detection):
    """Encode only clock, commands, fixed detections and the opening encoder."""
    cube, plate = detection_positions(detection)
    angles = 2 * np.pi * np.array([1, 2, 4, 8, 16]) * t / HORIZON
    return np.r_[
        t / HORIZON, np.column_stack((np.sin(angles), np.cos(angles))).ravel(),
        (np.asarray(pad_cmd) - [.15, -.15, .85]) / [.2, .2, .1],
        (cube - pad_cmd) / .05, (plate - pad_cmd) / .3, opening,
    ].astype(np.float32)


class Observer:
    """Detect once at reset; keep diagnostic truth outside actor inputs."""

    def __init__(self, mode, calibration, rng):
        self.mode, self.calibration, self.rng = mode, calibration, rng

    def reset(self, state, episode_dir):
        if self.mode == "simulated":
            self.detection = simulated_detection(state, self.rng, self.calibration["sigma"])
        else:
            self.detection = image_detection(
                imageio.imread(Path(episode_dir) / "reset_cam_head.png"), self.calibration)
        self.truth = np.array([state["sample"]["cube_position"][:2], state["plate_position"][:2]])

    def __call__(self, state, t, pad_cmd):
        actor = actor_observe(t, pad_cmd, state["sample"]["gripper_opening"]["right"], self.detection)
        return actor, np.concatenate((actor, observe(state, t, pad_cmd)))

    def record(self):
        return {"detector": self.mode, "detection": self.detection.tolist(),
                "truth": self.truth.tolist(), "error": (self.detection - self.truth).tolist()}
