"""CPU regression checks for the standalone cube-lifting task."""

import argparse
import ast
import importlib.util
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

TASK_ROOT = Path(__file__).resolve().parents[1] / "AgenticSimulator/tasks/lift_cube/IsaacLab-Arena"


def load_task_math():
    # The task's numerical functions need no simulator. Load their definitions without
    # importing Isaac Lab, which requires SimulationApp and GPU runtime initialization.
    tree = ast.parse((TASK_ROOT / "isaaclab_arena/mdp.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

    def transform(t, q, point):
        w, xyz = q[:, :1], q[:, 1:]
        cross = torch.cross(xyz, point, dim=1)
        return t + point + 2 * (w * cross + torch.cross(xyz, cross, dim=1)), None

    def inverse_transform(t, q, point):
        inverse = q.clone()
        inverse[:, 1:] *= -1
        return transform(torch.zeros_like(t), inverse, point - t)

    namespace = {
        "torch": torch,
        "wp": SimpleNamespace(to_torch=lambda value: value),
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
        "combine_frame_transforms": transform,
        "subtract_frame_transforms": inverse_transform,
    }
    source = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(source), "lift_cube_mdp", "exec"), namespace)
    return SimpleNamespace(**{node.name: namespace[node.name] for node in functions})


@unittest.skipIf(torch is None, "PyTorch is needed for task math checks")
class LiftCubeMathTests(unittest.TestCase):
    def setUp(self):
        self.mdp = load_task_math()
        self.robot_pose = torch.tensor([[1., 2., 0., 1., 0., 0., 0.]])
        self.cube_pose = torch.tensor([[1.2, 2., 0.3, 1., 0., 0., 0.]])
        self.command = torch.tensor([[0.2, 0., 0.3, 1., 0., 0., 0.]])
        self.env = SimpleNamespace(
            num_envs=1, device="cpu",
            scene={
                "robot": SimpleNamespace(data=SimpleNamespace(root_state_w=self.robot_pose)),
                "object": SimpleNamespace(data=SimpleNamespace(root_pos_w=self.cube_pose[:, :3])),
                "ee_frame": SimpleNamespace(data=SimpleNamespace(target_pos_w=self.cube_pose[:, None, :3].clone())),
            },
            command_manager=SimpleNamespace(get_command=lambda _: self.command),
        )

    def test_reaching_reward_at_contact_and_at_distance(self):
        self.assertAlmostEqual(self.mdp.object_ee_distance(self.env, 0.1).item(), 1.)
        self.env.scene["ee_frame"].data.target_pos_w[:, :, 0] += 0.1
        expected = 1 - torch.tanh(torch.tensor(1.)).item()
        self.assertAlmostEqual(self.mdp.object_ee_distance(self.env, 0.1).item(), expected, places=5)

    def test_goal_reward_uses_robot_frame_and_height_gate(self):
        reward = self.mdp.object_goal_distance(self.env, 0.3, 0.04, "object_pose")
        self.assertAlmostEqual(reward.item(), 1.)
        self.cube_pose[:, 2] = 0.04
        self.assertEqual(self.mdp.object_is_lifted(self.env, 0.04).item(), 0.)
        self.assertEqual(self.mdp.object_goal_distance(self.env, 0.3, 0.04, "object_pose").item(), 0.)

    def test_rotated_robot_observation_and_goal(self):
        half_sqrt = 2 ** -0.5
        self.robot_pose[:, 3:] = torch.tensor([half_sqrt, 0., 0., half_sqrt])
        self.cube_pose[:, :3] = torch.tensor([1., 2.2, 0.3])
        position = self.mdp.object_position_in_frame(self.env, SimpleNamespace(name="robot"))
        torch.testing.assert_close(position, torch.tensor([[0.2, 0., 0.3]]), atol=1e-6, rtol=1e-5)
        self.assertTrue(self.mdp.lift_object_rl_success(self.env).item())

    def test_success_disabled_only_during_training(self):
        self.assertFalse(self.mdp.lift_object_rl_success(self.env, rl_training=True).item())
        self.assertTrue(self.mdp.lift_object_rl_success(self.env, rl_training=False).item())
        self.cube_pose[:, 0] += 0.06
        self.assertFalse(self.mdp.lift_object_rl_success(self.env, rl_training=False).item())

    def test_gripper_observation_preserves_order_sign_and_copy(self):
        joints = torch.tensor([[0.01, 0.02]])
        self.env.scene["robot"].data.joint_pos = joints
        observed = self.mdp.gripper_pos(self.env)
        joints.zero_()
        torch.testing.assert_close(observed, torch.tensor([[0.02, -0.01]]))

    def test_cube_reset_only_writes_selected_parallel_environments(self):
        writes = {}
        cube = SimpleNamespace(
            write_root_pose_to_sim=lambda pose, **kwargs: writes.update(pose=pose, pose_ids=kwargs["env_ids"]),
            write_root_velocity_to_sim=lambda velocity, **kwargs: writes.update(velocity=velocity, velocity_ids=kwargs["env_ids"]),
        )
        scene = type("Scene", (dict,), {})(dex_cube=cube)
        scene.env_origins = torch.tensor([[0., 0., 0.], [30., 0., 0.], [60., 0., 0.]])
        self.env.scene = scene
        self.mdp.reset_cube(self.env, torch.tensor([2, 0]))
        torch.testing.assert_close(writes["pose"][:, :3], torch.tensor([[60.5, 0., 0.055], [0.5, 0., 0.055]]))
        torch.testing.assert_close(writes["pose"][:, 3:], torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]))
        self.assertEqual(torch.count_nonzero(writes["velocity"]).item(), 0)
        self.assertEqual(writes["pose_ids"].tolist(), [2, 0])
        self.assertEqual(writes["velocity_ids"].tolist(), [2, 0])

    def test_policy_quaternions_stay_xyzw(self):
        self.command[:, 3:] = torch.tensor([0.5, 0.1, 0.2, 0.3])
        observed = self.mdp.generated_object_command(self.env)
        torch.testing.assert_close(observed[:, 3:], torch.tensor([[0.1, 0.2, 0.3, 0.5]]))
        self.env.scene["ee_frame"].data.target_quat_w = self.command[:, None, 3:]
        torch.testing.assert_close(self.mdp.ee_frame_quat(self.env), observed[:, 3:])



class LiftCubeLauncherTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("cube_train", TASK_ROOT / "train.py")
        self.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.launcher)

    def test_failed_training_keeps_log_and_exit_status_and_prevents_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "202611131200_failure"
            def fail(command, **kwargs):
                kwargs["stdout"].write("original training failure\n")
                self.assertEqual(kwargs["cwd"], run_dir)
                self.assertIn("isaaclab_arena.train", command)
                self.assertEqual(command[-2:], ["--num_envs", "2"])
                return subprocess.CompletedProcess(command, 7)
            with patch.object(self.launcher.subprocess, "run", side_effect=fail):
                self.assertEqual(self.launcher.main(["--run-dir", str(run_dir), "--num_envs", "2"]), 7)
            self.assertEqual((run_dir / "training.log").read_text(), "original training failure\n")
            with self.assertRaises(FileExistsError):
                self.launcher.main(["--run-dir", str(run_dir)])

    def test_clean_process_exit_without_training_completion_is_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "202611131200_incomplete"
            with patch.object(self.launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
                self.assertEqual(self.launcher.main(["--run-dir", str(run_dir)]), 1)
            self.assertTrue((run_dir / "training.log").exists())

    def test_invalid_run_prefix_and_task_override_fail_before_launch(self):
        with patch.object(self.launcher.subprocess, "run") as run, patch.object(argparse.ArgumentParser, "error", side_effect=ValueError):
            with self.assertRaises(ValueError):
                self.launcher.main(["--run-dir", "invalid"])
            with self.assertRaises(ValueError):
                self.launcher.main(["--run-dir", "202611131200_test", "--task", "another_task"])
            run.assert_not_called()


class LiftCubePlaybackTests(unittest.TestCase):
    def test_saved_config_retains_native_tuple_and_slice_types(self):
        spec = importlib.util.spec_from_file_location("cube_playback", TASK_ROOT / "isaaclab_arena/play.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "env.yaml"
            config.write_text("eye: !!python/tuple [1, 2, 3]\nids: !!python/object/apply:builtins.slice [0, null, 1]\n")
            loaded = module.load_config(config)
        self.assertEqual(loaded["eye"], (1, 2, 3))
        self.assertEqual((loaded["ids"].start, loaded["ids"].stop, loaded["ids"].step), (0, None, 1))

    def test_playback_resolves_checkpoint_and_keeps_training_files_unchanged(self):
        spec = importlib.util.spec_from_file_location("cube_play", TASK_ROOT / "play.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "training" / "model_1600.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"original checkpoint")
            (checkpoint.parent / "params").mkdir()
            for name in ["env.yaml", "agent.yaml"]:
                (checkpoint.parent / "params" / name).write_text("{}\n")
            run_dir = root / "202611131900_play"
            def play(command, **kwargs):
                self.assertIn("isaaclab_arena.play", command)
                self.assertEqual(command[command.index("--checkpoint") + 1], str(checkpoint.resolve()))
                self.assertEqual(kwargs["cwd"], run_dir)
                (run_dir / "playback_complete.json").write_text("{}\n")
                return subprocess.CompletedProcess(command, 0)
            with patch.object(launcher.subprocess, "run", side_effect=play):
                self.assertEqual(launcher.main(["--checkpoint", str(checkpoint), "--run-dir", str(run_dir)]), 0)
            self.assertEqual(checkpoint.read_bytes(), b"original checkpoint")
            self.assertTrue((run_dir / "play.log").is_file())
