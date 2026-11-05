"""Object-relative pick, lift, carry and release policy."""

import numpy as np

from integrations.dynamics_rl.features import (
    REFERENCE_QUATERNION, encode_action, quaternion_product, unit_quaternion,
)
from simulators.dynamics_scoring import Criteria

Q_DOWN = REFERENCE_QUATERNION
GRASP_OFFSET = 0.145
TABLE_Z = Criteria().table_height
CUBE_HALF = Criteria().cube_size / 2
LOWER_DZ = 0.04
LOWER_STEPS = 30
PARAMETER_NAMES = (
    "grasp_dx", "grasp_dy", "grasp_dz", "yaw", "grip_cmd", "t_descend",
    "t_close", "lift_clear", "t_lift", "tilt", "t_carry", "release_dx",
    "release_dy", "t_settle",
)
LOW = np.array([-.008, -.008, -.015, -15, 0, 10, 6, .05, 8, 0, 12, -.03, -.03, 0])
HIGH = np.array([.008, .008, .010, 15, .4, 40, 30, .08, 40, 40, 100, .03, .03, 20])
PHASES = ("approach", "descend", "close", "lift", "carry", "settle",
          "lower", "release", "withdraw", "hold")


def physical(u):
    """Map normalized coordinates to physical parameters and integer durations."""
    values = LOW + (np.asarray(u) + 1) * (HIGH - LOW) / 2
    return {name: int(round(value)) if name.startswith("t_") else float(value)
            for name, value in zip(PARAMETER_NAMES, values)}


def normalized(params):
    """Map physical parameters to normalized coordinates."""
    return 2 * (np.array([params[name] for name in PARAMETER_NAMES]) - LOW) / (HIGH - LOW) - 1


def approach_axis(q):
    """Return the hand's local x axis in world coordinates."""
    w, x, y, z = q
    return np.array([1 - 2 * (y*y + z*z), 2 * (x*y + w*z), 2 * (x*z - w*y)])


def grasp_point(pose):
    """Return the pad centre for an end-effector pose."""
    return np.asarray(pose[:3]) + GRASP_OFFSET * approach_axis(pose[3:])


def ee_position(point, q):
    """Return the end-effector position for a pad centre and orientation."""
    return np.asarray(point) - GRASP_OFFSET * approach_axis(q)


def hand_q(yaw, tilt):
    """Compose world y tilt and world z yaw, specified in degrees."""
    yaw, tilt = np.deg2rad([yaw, tilt]) / 2
    qy = np.array([np.cos(tilt), 0, np.sin(tilt), 0])
    qz = np.array([np.cos(yaw), 0, 0, np.sin(yaw)])
    return unit_quaternion(quaternion_product(qy, quaternion_product(qz, Q_DOWN)))


def slerp(a, b, fraction):
    """Interpolate along the shortest quaternion arc."""
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        return unit_quaternion(a + fraction * (b - a))
    angle = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b) / np.sin(angle)


class PrimitivePolicy:
    """Generate absolute end-effector commands from phase-entry observations."""

    def __init__(self, params):
        self.params = params
        self.lengths = (40, params["t_descend"], params["t_close"], params["t_lift"],
                        params["t_carry"], params["t_settle"], LOWER_STEPS, 10, 15, 30)
        self.phase_index = self.phase_step = 0
        self.finished = self.phase_end = False
        self.commanded_pose = None

    def _enter(self, state):
        p, phase = self.params, PHASES[self.phase_index]
        if self.commanded_pose is None:
            self.commanded_pose = np.asarray(state["public_state"]["right_ee_pose"], dtype=float)
        self.start_point = grasp_point(self.commanded_pose)
        self.start_q = unit_quaternion(self.commanded_pose[3:])
        self.target_point, self.target_q = self.start_point.copy(), self.start_q.copy()
        if phase in ("approach", "descend"):
            self.target_point = np.asarray(state["sample"]["cube_position"]) + [
                p["grasp_dx"], p["grasp_dy"], p["grasp_dz"] + (.06 if phase == "approach" else 0)]
            self.target_q = hand_q(p["yaw"], 0)
        elif phase == "lift":
            self.target_point[2] = TABLE_Z + CUBE_HALF + p["lift_clear"] + p["grasp_dz"]
        elif phase == "carry":
            self.target_point[:2] = np.asarray(state["plate_position"])[:2] + [
                p["release_dx"], p["release_dy"]]
            self.target_q = hand_q(p["yaw"], p["tilt"])
        elif phase == "lower":
            self.target_point[2] -= LOWER_DZ
        elif phase == "withdraw":
            self.target_point[2] += .08
        self.grip = 1.0 if phase in ("approach", "descend", "release", "withdraw", "hold") else p["grip_cmd"]

    def act(self, state):
        """Return the next normalized action and its phase name."""
        if self.finished:
            raise RuntimeError("Primitive policy is finished")
        while self.phase_step == self.lengths[self.phase_index]:
            self.phase_index += 1
            self.phase_step = 0
        if self.phase_step == 0:
            self._enter(state)
        self.phase_step += 1
        tau = self.phase_step / self.lengths[self.phase_index]
        fraction = 10*tau**3 - 15*tau**4 + 6*tau**5
        point = self.start_point + fraction * (self.target_point - self.start_point)
        q = slerp(self.start_q, self.target_q, fraction)
        self.commanded_pose = np.r_[ee_position(point, q), q]
        self.phase_end = self.phase_step == self.lengths[self.phase_index]
        self.finished = self.phase_index == len(PHASES) - 1 and self.phase_end
        return encode_action(self.commanded_pose, self.grip), PHASES[self.phase_index]
