"""Position-adapted R1 demonstration controller; never used by neural inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from integrations.dynamics_rl.features import encode_action

DEFAULT_TEACHER = (
    Path(__file__).resolve().parents[2]
    / "runs/202610301439_dynamics_calibration_v4_0.3/policies/reference/policy.json"
)
TEACHER_VERSION = "r1-relative-position-fixed-left-v1"
DOWN_QUATERNION = np.array([0.5, -0.5, 0.5, 0.5])
TILTED_QUATERNION = np.array([0.35355339, -0.35355339, 0.61237244, 0.61237244])


@dataclass(frozen=True)
class Move:
    name: str
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    grip: float
    steps: int


class R1Teacher:
    """Preserve R1 interpolation/timing while translating its two object targets.

    The worker applies the same fixed-left controller and action decoder used by
    the network. This is an adapted supplied development policy, not an unchanged
    replay of the archived R1 and not a learned skill.
    """

    def __init__(self, initial_state):
        cube = np.asarray(initial_state["sample"]["cube_position"], dtype=float)
        plate = np.asarray(initial_state["plate_position"], dtype=float)
        if cube.shape != (3,) or plate.shape != (3,) or not np.isfinite([cube, plate]).all():
            raise ValueError("Teacher requires finite initial cube and plate positions")
        gx, gy = float(cube[0]), float(cube[1] + 0.0012)
        px, py = float(plate[0] + 0.0843), float(plate[1] + 0.0012)
        retreat = (float(plate[0] + 0.18), float(plate[1] - 0.04), 1.08)
        down, tilted = tuple(DOWN_QUATERNION), tuple(TILTED_QUATERNION)
        self.moves = (
            Move("approach", (gx, gy, 1.02), down, 1.0, 60),
            Move("lower", (gx, gy, 0.925), down, 1.0, 45),
            Move("close", (gx, gy, 0.925), down, 0.0, 40),
            Move("lift", (gx, gy, 0.99), down, 0.0, 20),
            Move("carry_and_tilt", (px, py, 0.965), tilted, 0.0, 35),
            Move("lower_to_plate", (px, py, 0.9355), tilted, 0.0, 10),
            Move("release", (px, py, 0.9355), tilted, 1.0, 25),
            Move("withdraw", (px, py, 1.03), tilted, 1.0, 30),
            Move("retreat", retreat, down, 1.0, 40),
            Move("hold", retreat, down, 1.0, 40),
        )
        self.stage = self.stage_step = self.steps = 0
        self.start = self.target = None
        self.last_pose = None
        self.last_grip = 1.0

    @property
    def finished(self):
        return self.stage >= len(self.moves)

    @property
    def stage_name(self):
        return "final_hold" if self.finished else self.moves[self.stage].name

    def act(self, state):
        if self.finished:
            self.steps += 1
            return encode_action(self.last_pose, self.last_grip)
        move = self.moves[self.stage]
        if self.stage_step == 0:
            start = np.asarray(state["public_state"]["right_ee_pose"], dtype=float).copy()
            if start.shape != (7,) or not np.isfinite(start).all():
                raise ValueError("Teacher requires a finite seven-value right_ee_pose")
            if start[3] < 0:
                start[3:] *= -1
            target = np.array([*move.position, *move.quaternion])
            if np.dot(start[3:], target[3:]) < 0:
                target[3:] *= -1
            self.start, self.target = start, target
        alpha = min(1.0, (self.stage_step + 1) / max(1, move.steps - 10))
        pose = (1.0 - alpha) * self.start + alpha * self.target
        pose[3:] /= np.linalg.norm(pose[3:])
        self.last_pose, self.last_grip = pose, move.grip
        self.stage_step += 1
        self.steps += 1
        if self.stage_step >= move.steps:
            self.stage += 1
            self.stage_step = 0
        return encode_action(pose, move.grip)

    def metadata(self):
        return {
            "teacher_version": TEACHER_VERSION,
            "source_kind": "supplied_development_policy",
            "changes": ["object-relative XY targets", "fixed-left action decoder"],
            "native_action_budget": 500,
            "scripted_actions": sum(move.steps for move in self.moves),
            "waypoints": [
                {"name": m.name, "position": list(m.position), "quaternion": list(m.quaternion),
                 "grip": m.grip, "steps": m.steps}
                for m in self.moves
            ],
        }
