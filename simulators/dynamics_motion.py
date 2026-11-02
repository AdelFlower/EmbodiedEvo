"""Measure task motion from poses at every native physics step."""

from __future__ import annotations

import numpy as np


def _unit_quaternion(value):
    quaternion = np.asarray(value, dtype=float)
    return quaternion / np.linalg.norm(quaternion)


class PoseMotion:
    """Retain independent peak translation and rotation rates per control interval.

    Quaternions use native scalar-first ordering. Angular velocity is expressed
    in world coordinates and uses the shortest rotation between adjacent poses.
    Reading a completed interval never changes its samples or advances time.
    """

    def __init__(self, position, orientation):
        self._position = np.array(position, dtype=float, copy=True)
        self._orientation = _unit_quaternion(orientation)
        self._pending_linear = np.zeros(3)
        self._pending_angular = np.zeros(3)
        self._pending_steps = 0
        self._linear = np.zeros(3)
        self._angular = np.zeros(3)
        self._steps = 0

    def record(self, position, orientation, dt):
        """Record one completed physics step using its actual simulation dt."""
        position = np.array(position, dtype=float, copy=True)
        orientation = _unit_quaternion(orientation)
        linear = (position - self._position) / dt

        # Relative rotation q_new * conjugate(q_old), in world coordinates.
        old_w, old_vector = self._orientation[0], self._orientation[1:]
        new_w, new_vector = orientation[0], orientation[1:]
        delta_w = new_w * old_w + np.dot(new_vector, old_vector)
        delta_vector = (
            old_w * new_vector - new_w * old_vector - np.cross(new_vector, old_vector)
        )
        if delta_w < 0:
            delta_w, delta_vector = -delta_w, -delta_vector
        axis_length = np.linalg.norm(delta_vector)
        angular = (
            delta_vector * (2 * np.arctan2(axis_length, delta_w) / (axis_length * dt))
            if axis_length > 0 else np.zeros(3)
        )

        if np.dot(linear, linear) > np.dot(self._pending_linear, self._pending_linear):
            self._pending_linear = linear
        if np.dot(angular, angular) > np.dot(self._pending_angular, self._pending_angular):
            self._pending_angular = angular
        self._pending_steps += 1
        self._position, self._orientation = position, orientation

    def finish_interval(self):
        """Commit physics samples; an empty interval cannot manufacture rest."""
        if self._pending_steps:
            self._linear, self._angular = self._pending_linear, self._pending_angular
            self._steps = self._pending_steps
            self._pending_linear = np.zeros(3)
            self._pending_angular = np.zeros(3)
            self._pending_steps = 0
        return self.snapshot()

    def snapshot(self):
        """Return only the last completed interval, without consuming it."""
        return {
            "linear_velocity": self._linear.tolist(),
            "angular_velocity": self._angular.tolist(),
            "physics_steps": self._steps,
        }
