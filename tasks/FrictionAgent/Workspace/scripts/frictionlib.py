"""NumPy geometry and trace loading for public friction episodes."""

from itertools import product

import numpy as np


def clearance(position, quaternion):
    """Measure the lowest cube corner above the table."""
    w, x, y, z = np.asarray(quaternion, dtype=float)
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    corners = np.array(list(product((-.0125, .0125), repeat=3)))
    return float((corners @ rotation.T + np.asarray(position))[:, 2].min() - .765)


def grasp_point(pose):
    """Return the pad centre for an end-effector pose."""
    w, x, y, z = pose[3:]
    axis = np.array([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)])
    return np.asarray(pose[:3]) + .145 * axis


def hand_offsets(poses, cubes):
    """Express cube offsets from the pad centres in the hand frames."""
    poses = np.asarray(poses)
    offsets = np.asarray(cubes) - np.array([grasp_point(p) for p in poses])
    inverse_xyz = -poses[:, 4:]
    return offsets + 2 * np.cross(
        inverse_xyz, np.cross(inverse_xyz, offsets) + poses[:, 3:4] * offsets)


def load_trace(path):
    """Load public trajectory arrays and decode the scalar stop reason."""
    with np.load(path, allow_pickle=False) as data:
        trace = dict(data)
    trace["stop_reason"] = str(trace["stop_reason"])
    return trace
