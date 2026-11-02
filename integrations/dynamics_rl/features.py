"""Shared, NumPy-only state/action contract for teacher, learner and simulator."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

SCHEMA = "dynamics-rl-state-v1"
OBS_DIM = 57
ACTION_DIM = 7
ACTION_LOW = np.array([-0.10, -0.45, 0.85], dtype=float)
ACTION_HIGH = np.array([0.40, 0.10, 1.20], dtype=float)
REFERENCE_QUATERNION = np.array([0.5, -0.5, 0.5, 0.5])
SCENE_PATH = Path(__file__).resolve().parents[2] / "tasks/DynamicsRepair/scene.json"


def unit_quaternion(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Expected a finite nonzero scalar-first quaternion")
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def quaternion_product(a, b):
    return np.r_[a[0] * b[0] - np.dot(a[1:], b[1:]),
                 a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])]


def encode_action(right_pose, grip):
    """Invert the decoder; reject teacher targets outside the declared workspace."""
    pose = np.asarray(right_pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite seven-dimensional EE pose")
    conjugate = REFERENCE_QUATERNION * [1, -1, -1, -1]
    delta = unit_quaternion(quaternion_product(unit_quaternion(pose[3:]), conjugate))
    length = np.linalg.norm(delta[1:])
    rotation = (
        delta[1:] * (2 * np.arctan2(length, delta[0]) / length)
        if length > 1e-12 else np.zeros(3)
    )
    action = np.r_[2 * (pose[:3] - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW) - 1,
                   rotation / np.pi, 2 * float(np.asarray(grip).reshape(-1)[0]) - 1]
    if not np.isfinite(action).all() or np.any(np.abs(action) > 1 + 1e-6):
        raise ValueError(f"Teacher target is outside the RL action space: {action}")
    return np.clip(action, -1, 1).astype(np.float32)


def decode_action(action, left_pose, left_grip):
    action = np.asarray(action, dtype=float)
    if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
        raise ValueError("Expected seven finite normalized actions")
    if np.any(np.abs(action) > 1 + 1e-6):
        raise ValueError("Normalized actions must lie in [-1, 1]")
    action = np.clip(action, -1, 1)
    position = ACTION_LOW + (action[:3] + 1) * (ACTION_HIGH - ACTION_LOW) / 2
    rotation = action[3:6] * np.pi
    angle = np.linalg.norm(rotation)
    delta = np.r_[np.cos(angle / 2), rotation * np.sinc(angle / (2 * np.pi)) / 2]
    quaternion = unit_quaternion(quaternion_product(delta, REFERENCE_QUATERNION))
    return {
        "left_ee_pose": np.asarray(left_pose, dtype=float).tolist(),
        "right_ee_pose": np.r_[position, quaternion].tolist(),
        "left_ee_joint_state": np.asarray(left_grip, dtype=float).reshape(-1).tolist(),
        "right_ee_joint_state": [(float(action[6]) + 1) / 2],
    }


def _position(value):
    return (np.asarray(value, dtype=float) - [0, -0.1, 0.8]) / [0.3, 0.3, 0.2]


def encode_observation(state, previous_action, step_count):
    """Fixed scaling (never fitted on evaluation); no private score/contact features."""
    sample, public = state["sample"], state["public_state"]
    cube = np.asarray(sample["cube_position"])
    plate = np.asarray(state["plate_position"])
    hand = np.asarray(sample["gripper_position"]["right"])
    ee = np.asarray(public["right_ee_pose"])
    values = np.concatenate([
        _position(cube), unit_quaternion(sample["cube_orientation"]),
        np.asarray(sample["cube_linear_velocity"]) / 0.5,
        np.asarray(sample["cube_angular_velocity"]) / 5,
        _position(plate), unit_quaternion(state["plate_orientation"]),
        np.asarray(public["right_arm_joint_state"]) / np.pi,
        np.asarray(state["right_arm_joint_velocity"]) / 3,
        _position(ee[:3]), unit_quaternion(ee[3:]),
        [sample["gripper_opening"]["right"]], _position(hand),
        (cube - hand) / 0.3, (plate - cube) / 0.3,
        np.asarray(previous_action), [1 - step_count / 500],
    ]).astype(np.float32)
    if values.shape != (OBS_DIM,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid {SCHEMA} observation: shape {values.shape}")
    return values


def potential(state):
    """Host-only progress potential; release/carry gates use the unchanged scorer."""
    sample, scoring = state["sample"], state["scoring"]
    cube = np.asarray(sample["cube_position"])
    hand = np.asarray(sample["gripper_position"]["right"])
    plate = np.asarray(state["plate_position"])
    reach = 1 - np.tanh(np.linalg.norm(cube - hand) / 0.06)
    grasp = float(all(sample["finger_contacts"]["right"]))
    lift = grasp * np.clip((cube[2] - 0.7775) / 0.06, 0, 1)
    carry = lift * (1 - np.tanh(np.linalg.norm(cube[:2] - plate[:2]) / 0.15))
    return float(reach + 2 * grasp + 3 * lift + 4 * carry
                 + 4 * scoring["transport_completed"]
                 + 6 * scoring["released_over_plate"]
                 + 2 * min(scoring["stable_steps"] / 13, 1))


def shaped_reward(previous_state, next_state, *, terminated, truncated,
                  native_reward, gamma=0.99):
    following = 0.0 if terminated or truncated else potential(next_state)
    return float(100 * native_reward + gamma * following - potential(previous_state) - 0.01)


def sample_scene(seed, radius=0.03):
    if not np.isfinite(radius) or not 0 <= radius <= 0.03:
        raise ValueError("This experiment supports XY randomization up to 0.03 m")
    scene = json.loads(SCENE_PATH.read_text())
    rng = np.random.default_rng(int(seed))
    for category in ("cube", "plate"):
        position = scene["Rigid"][category][0]["default_pos"]
        position[:2] = (np.asarray(position[:2]) + rng.uniform(-radius, radius, 2)).tolist()
    return scene


def contract():
    return copy.deepcopy({
        "schema": SCHEMA, "observation_dim": OBS_DIM, "action_dim": ACTION_DIM,
        "action_position_low": ACTION_LOW.tolist(), "action_position_high": ACTION_HIGH.tolist(),
        "action_reference_quaternion": REFERENCE_QUATERNION.tolist(),
        "position_offset": [0, -0.1, 0.8], "position_scale": [0.3, 0.3, 0.2],
        "joint_position_scale": float(np.pi), "joint_velocity_scale": 3.0,
        "cube_linear_velocity_scale": 0.5, "cube_angular_velocity_scale": 5.0,
        "relative_position_scale": 0.3, "action_limit": 500, "control_hz": 25,
        "observation_privilege": "object_ground_truth_and_robot_state",
    })
