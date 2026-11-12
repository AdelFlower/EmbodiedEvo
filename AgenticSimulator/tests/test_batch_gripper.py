"""CPU checks for independent gripper modes, without starting Isaac Sim."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


def load_controller():
    # Only the Isaac imports are replaced; control code and tensor operations
    # come from the real batch controller and PyTorch on CPU.
    modules = {
        name: ModuleType(name)
        for name in (
            "isaaclab", "isaaclab.controllers", "isaaclab.utils",
            "isaaclab.utils.math",
        )
    }
    controllers = modules["isaaclab.controllers"]
    controllers.DifferentialIKControllerCfg = SimpleNamespace
    controllers.DifferentialIKController = lambda cfg, **kwargs: SimpleNamespace(
        cfg=cfg, **kwargs
    )

    def unused_math(*args, **kwargs):
        raise AssertionError("Gripper checks must not invoke mocked kinematics")

    for name in (
        "compute_pose_error", "matrix_from_quat", "quat_apply",
        "subtract_frame_transforms",
    ):
        setattr(modules["isaaclab.utils.math"], name, unused_math)

    source = Path(__file__).resolve().parents[1] / "tasks/pick_cube/batch_controller.py"
    spec = importlib.util.spec_from_file_location("batch_gripper_under_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module.BatchPickCubeController


class Robot:
    def __init__(self, n=3):
        self.is_fixed_base = True
        self.joint_names = [f"panda_joint{i}" for i in range(1, 8)] + [
            "panda_finger_joint1", "panda_finger_joint2",
        ]
        self.data = SimpleNamespace(
            joint_pos=torch.zeros(n, 9),
            default_joint_pos=torch.zeros(n, 9),
            # Different gains per environment expose accidental global resets.
            joint_stiffness=torch.arange(n * 9).reshape(n, 9).float() + 2000,
            joint_damping=torch.arange(n * 9).reshape(n, 9).float() + 100,
        )
        self.actuators = {"panda_hand": SimpleNamespace(
            stiffness=self.data.joint_stiffness[:, 7:].clone(),
            damping=self.data.joint_damping[:, 7:].clone(),
        )}
        self.gravity = torch.arange(n * 9).reshape(n, 9).float() / 10
        self.root_physx_view = SimpleNamespace(
            get_gravity_compensation_forces=lambda: self.gravity.clone()
        )
        self.effort = torch.zeros(n, 9)
        self.position = torch.zeros(n, 9)
        self.gain_writes = []

    def find_joints(self, name):
        return [self.joint_names.index(name)], [name]

    def find_bodies(self, name):
        return [8], [name]

    def write_joint_stiffness_to_sim(self, values, joint_ids):
        self.data.joint_stiffness[:, joint_ids] = values
        self.gain_writes.append("stiffness")

    def write_joint_damping_to_sim(self, values, joint_ids):
        self.data.joint_damping[:, joint_ids] = values
        self.gain_writes.append("damping")

    def set_joint_effort_target(self, values, joint_ids):
        self.effort[:, joint_ids] = values

    def set_joint_position_target(self, values, joint_ids):
        self.position[:, joint_ids] = values


class BatchGripperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controller_class = load_controller()

    def setUp(self):
        self.robot = Robot()
        self.obj = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(3, 3)))
        self.controller = self.controller_class(
            self.robot, self.obj, self.obj, "cpu", grip_force=15.0
        )
        self.original_kp = self.robot.data.joint_stiffness.clone()
        self.original_kd = self.robot.data.joint_damping.clone()

    def mask(self, *indices):
        mask = torch.zeros(3, dtype=torch.bool)
        mask[list(indices)] = True
        return mask

    def test_initial_position_control_has_no_closing_force(self):
        self.controller._send()
        torch.testing.assert_close(self.robot.effort[:, 7:], torch.zeros(3, 2))
        torch.testing.assert_close(self.robot.position[:, 7:], torch.full((3, 2), 0.04))
        self.assertEqual(self.robot.gain_writes, [])

    def test_close_only_changes_selected_environments_and_preserves_arm(self):
        self.controller._enter(self.mask(0, 2), self.controller.CLOSE)
        self.controller._send()
        expected_kp = self.original_kp.clone()
        expected_kd = self.original_kd.clone()
        expected_kp[[0, 2], 7:] = 0
        expected_kd[[0, 2], 7:] = 0
        torch.testing.assert_close(self.robot.data.joint_stiffness, expected_kp)
        torch.testing.assert_close(self.robot.data.joint_damping, expected_kd)
        torch.testing.assert_close(
            self.robot.effort[:, 7:],
            torch.tensor([[-15., -15.], [0., 0.], [-15., -15.]]),
        )
        torch.testing.assert_close(self.robot.effort[:, :7], self.robot.gravity[:, :7])
        torch.testing.assert_close(
            self.robot.actuators["panda_hand"].stiffness, expected_kp[:, 7:]
        )
        torch.testing.assert_close(
            self.robot.actuators["panda_hand"].damping, expected_kd[:, 7:]
        )

    def test_open_restores_own_gains_and_clears_force_without_releasing_others(self):
        self.controller._enter(self.mask(0, 1, 2), self.controller.CLOSE)
        self.controller._send()
        self.controller._enter(self.mask(1), self.controller.OPEN)
        self.controller._send()
        expected_kp = torch.zeros(3, 2)
        expected_kd = torch.zeros(3, 2)
        expected_kp[1] = self.original_kp[1, 7:]
        expected_kd[1] = self.original_kd[1, 7:]
        torch.testing.assert_close(self.robot.data.joint_stiffness[:, 7:], expected_kp)
        torch.testing.assert_close(self.robot.data.joint_damping[:, 7:], expected_kd)
        torch.testing.assert_close(
            self.robot.effort[:, 7:],
            torch.tensor([[-15., -15.], [0., 0.], [-15., -15.]]),
        )

    def test_transport_keeps_force_and_does_not_rewrite_gains(self):
        self.controller._enter(self.mask(0), self.controller.CLOSE)
        self.controller._send()
        for state in (self.controller.LIFT, self.controller.TRANSFER, self.controller.LOWER):
            self.controller._enter(self.mask(0), state)
            self.controller._send()
            torch.testing.assert_close(self.robot.effort[0, 7:], torch.full((2,), -15.))
        self.assertEqual(self.robot.gain_writes, ["stiffness", "damping"])

    def test_multiple_mode_changes_are_submitted_once_per_send(self):
        self.controller._enter(self.mask(0), self.controller.CLOSE)
        self.controller._send()
        self.robot.gain_writes.clear()
        self.controller._enter(self.mask(0), self.controller.OPEN)
        self.controller._enter(self.mask(2), self.controller.CLOSE)
        self.controller._send()
        self.assertEqual(self.robot.gain_writes, ["stiffness", "damping"])
        torch.testing.assert_close(self.robot.effort[:, 7:], torch.tensor([
            [0., 0.], [0., 0.], [-15., -15.],
        ]))

    def test_empty_transitions_do_not_rewrite_gains(self):
        self.controller._enter(self.mask(), self.controller.CLOSE)
        self.controller._enter(self.mask(), self.controller.OPEN)
        self.controller._send()
        self.assertEqual(self.robot.gain_writes, [])

    def test_failure_preserves_gripper_mode_and_cannot_be_reentered(self):
        self.controller._enter(self.mask(0), self.controller.CLOSE)
        self.controller._finish(self.mask(0), False, 1)
        self.controller._finish(self.mask(1), False, 1)
        self.controller._enter(self.mask(0), self.controller.OPEN)
        self.controller._enter(self.mask(1), self.controller.CLOSE)
        self.controller._send()
        torch.testing.assert_close(self.robot.effort[:, 7:], torch.tensor([
            [-15., -15.], [0., 0.], [0., 0.],
        ]))
        self.assertEqual(self.controller.state.tolist(), [12, 12, 0])

    def test_force_is_recorded_for_all_outcomes_and_summary(self):
        self.assertEqual(self.controller.summary()["grip_force_per_finger_n"], 15.0)
        self.assertTrue(all(
            row["grip_force_per_finger_n"] == 15.0
            for row in self.controller.outcomes()
        ))

    def test_invalid_force_is_rejected(self):
        for value in (0, -1, 201, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.controller_class(self.robot, self.obj, self.obj, "cpu", grip_force=value)


if __name__ == "__main__":
    unittest.main()
