"""Extract raw RGB frames from native simulator observations."""

import numpy as np


def rgb_images(obs: dict) -> dict[str, np.ndarray]:
    """Extract only RGB already present in the public policy observation."""
    frames = {}
    for name, value in obs.items():
        if name.startswith("video."):
            frames[name.removeprefix("video.")] = np.asarray(value)
        elif name.endswith("_image") and isinstance(value, np.ndarray):
            frames[name.removesuffix("_image")] = value
        elif isinstance(value, dict):
            rgb = value.get("rgb", value.get("color"))
            if rgb is None:
                rgb = value.get("images", {}).get("rgb")
            if rgb is not None:
                frames[name] = np.asarray(rgb)
    if isinstance(obs.get("observation"), dict):
        frames.update(rgb_images(obs["observation"]))
    if isinstance(obs.get("vision"), dict):
        frames.update(rgb_images(obs["vision"]))
    return frames
