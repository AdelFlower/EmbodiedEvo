"""Shared geometric measurements and released-cube placement checks."""

import numpy as np

from integrations.friction_rl.primitive import (
    CUBE_HALF, PHASES, TABLE_Z, grasp_point,
)
from simulators.dynamics_scoring import Criteria, cube_vertices

PLATE_REST_RADIUS = 0.045
REST_STEPS = Criteria().stable_steps


def clearance(position, quaternion):
    """Measure the lowest cube corner above the table."""
    return float(cube_vertices(position, quaternion, 2 * CUBE_HALF)[:, 2].min() - TABLE_Z)


def rest_row(phase, contacts, fingers, opening, position, quaternion, plate_position):
    """Recognize a released cube supported inside the plate without robot contact."""
    criteria = Criteria()
    if (phase < PHASES.index("release") or not contacts[1] or contacts[0] or contacts[2]
            or any(fingers) or opening < .8):
        return False
    lowest = cube_vertices(position, quaternion, criteria.cube_size)[:, 2].min()
    return bool(criteria.plate_bottom - .002 <= lowest <= criteria.plate_top + .003
                and np.linalg.norm(np.asarray(position)[:2] - np.asarray(plate_position)[:2])
                <= PLATE_REST_RADIUS)


def motion_peaks(poses):
    """Measure peak grasp-point speed and acceleration from 25 Hz pose samples."""
    points = np.array([grasp_point(pose) for pose in poses]).reshape(-1, 3)
    velocity = np.diff(points, axis=0) / .04
    acceleration = np.diff(velocity, axis=0) / .04
    return (float(np.linalg.norm(velocity, axis=1).max(initial=0.)),
            float(np.linalg.norm(acceleration, axis=1).max(initial=0.)))


def hand_offsets(poses, cubes):
    """Express cube offsets from the pad centres in the hand frames."""
    poses = np.asarray(poses)
    offsets = np.asarray(cubes) - np.array([grasp_point(p) for p in poses])
    inverse_xyz = -poses[:, 4:]
    return offsets + 2 * np.cross(
        inverse_xyz, np.cross(inverse_xyz, offsets) + poses[:, 3:4] * offsets)
