"""Host-seeded DP layouts, with no dependence on scheduling or batch position."""

import json
from zlib import crc32

import numpy as np

from integrations.deployable.layout import build_scene
from integrations.friction_rl.control import SCENE_PATH

DP_LAYOUT = {"layout_radius": .03, "cube_shift": [0, 0], "plate_box": [[0, .03], [-.03, 0]]}


def _layout(stream, index, sequence):
    scene_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
    scene = build_scene(scene_seed, DP_LAYOUT)
    nominal = json.loads(SCENE_PATH.read_text())
    offsets = {
        f"{name}_offset": (np.asarray(scene["Rigid"][name][0]["default_pos"][:2]) -
                           nominal["Rigid"][name][0]["default_pos"][:2]).tolist()
        for name in ("cube", "plate")}
    return {"stream": stream, "index": index, "scene_seed": scene_seed, **offsets}


def make_layouts(stream, seed, count):
    return [_layout(stream, k, np.random.SeedSequence([seed, crc32(stream.encode()), k]))
            for k in range(count)]


def training_layout(seed, stream, task, iteration, replica):
    # A separate spawn domain keeps the training tuple distinct from table indices.
    return _layout(stream, replica, np.random.SeedSequence(
        [seed, crc32(stream.encode()), task, iteration, replica], spawn_key=(1,)))
