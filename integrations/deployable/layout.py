"""Deployable-policy layouts: cube uniform in a square, plate in a reachable box."""

import json

import numpy as np

from integrations.friction_rl.control import SCENE_PATH


def build_scene(seed, cfg):
    """Cube offset uniform +-layout_radius; plate offset uniform in plate_box [[x0, x1], [y0, y1]]."""
    scene = json.loads(SCENE_PATH.read_text())
    rng = np.random.default_rng(int(seed))
    radius = cfg["layout_radius"]
    (x0, x1), (y0, y1) = cfg["plate_box"]
    offsets = {"cube": rng.uniform(-radius, radius, 2) + cfg["cube_shift"],
               "plate": rng.uniform([x0, y0], [x1, y1])}
    for category, offset in offsets.items():
        position = scene["Rigid"][category][0]["default_pos"]
        position[:2] = (np.asarray(position[:2]) + offset).tolist()
    return scene
