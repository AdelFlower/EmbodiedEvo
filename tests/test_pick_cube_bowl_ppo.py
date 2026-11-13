"""CPU regression checks for the new contact-only PPO task."""
import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from AgenticSimulator.tasks.pick_cube_plate.RL.PPO.runtime import parser_for, run_directory

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from AgenticSimulator.tasks.pick_cube_plate.RL.PPO.task_math import shaping, touching


class TestRunDirectory(unittest.TestCase):
    def test_explicit_valid_timestamp_required(self):
        self.assertEqual(run_directory("runs/202611131900_ppo").name, "202611131900_ppo")
        for value in ("runs/latest", "runs/202613131900_ppo", "runs/202611132500_ppo"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                run_directory(value)
        self.assertTrue(next(a for a in parser_for("train")._actions if a.dest == "run_dir").required)
        self.assertTrue(next(a for a in parser_for("eval")._actions if a.dest == "checkpoint").required)


@unittest.skipUnless(HAS_TORCH, "PyTorch is needed for contact and reward checks")
class TestCubeBowlContact(unittest.TestCase):
    def load_mdp(self):
        # Stub only the Isaac boundary; execute the real sensor latch methods.
        class ContactSensor:
            def _initialize_impl(self):
                pass
            def _update_buffers_impl(self, env_ids):
                pass
            def reset(self, env_ids=None):
                pass
        sensors = ModuleType("isaaclab.sensors")
        sensors.ContactSensor = ContactSensor
        math = ModuleType("isaaclab.utils.math")
        math.quat_apply = math.subtract_frame_transforms = lambda *a: None
        modules = {"isaaclab": ModuleType("isaaclab"), "isaaclab.sensors": sensors,
                   "isaaclab.utils": ModuleType("isaaclab.utils"), "isaaclab.utils.math": math}
        name = "AgenticSimulator.tasks.pick_cube_plate.RL.PPO.mdp"
        source = Path(__file__).resolve().parents[1] / "AgenticSimulator/tasks/pick_cube_plate/RL/PPO/mdp.py"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        return module

    def test_filtered_contact_accepts_any_direction_and_ignores_zero_force(self):
        force = torch.zeros(3, 1, 2, 3)
        force[1, 0, 1, 2] = -0.1
        force[2, 0, 0, 0] = 0.2
        torch.testing.assert_close(touching(force), torch.tensor([False, True, True]))
        with self.assertRaises(RuntimeError):
            touching(None)

    def test_brief_contact_survives_next_physics_step_and_partial_reset(self):
        mdp = self.load_mdp()
        sensor = mdp.BowlContactSensor()
        sensor._num_envs, sensor._device = 3, "cpu"
        sensor._initialize_impl()
        force = torch.zeros(3, 1, 1, 3)
        sensor._data = SimpleNamespace(force_matrix_w=force)
        ids = torch.arange(3)
        force[0, 0, 0, 2] = 1.
        force[2, 0, 0, 1] = 1.
        sensor._update_buffers_impl(ids)
        force.zero_()
        sensor._update_buffers_impl(ids)
        torch.testing.assert_close(sensor.touched, torch.tensor([True, False, True]))
        sensor.reset(torch.tensor([0]))
        sensor._update_buffers_impl(torch.tensor([0]))
        torch.testing.assert_close(sensor.touched, torch.tensor([False, False, True]))
        sensor.reset()
        self.assertFalse(sensor.touched.any())

    def test_success_needs_only_contact_and_returns_transition_snapshot(self):
        mdp = self.load_mdp()
        sensor = SimpleNamespace(touched=torch.tensor([True, False]))
        env = SimpleNamespace(scene={"bowl_contact": sensor})
        result = mdp.cube_touches_bowl(env)
        sensor.touched.zero_()  # auto-reset must not erase termination result
        torch.testing.assert_close(result, torch.tensor([True, False]))

    def test_reward_requires_real_rise_and_is_translation_invariant(self):
        cube = torch.tensor([[0.15, 0., 0.7625], [0.15, 0., 0.8425]])
        bowl = torch.tensor([[-0.1, 0., 0.78]]).repeat(2, 1)
        grasp = cube.clone()
        initial = torch.full((2,), 0.7625)
        values = shaping(grasp, cube, bowl, initial)
        torch.testing.assert_close(values[0], torch.ones(2))
        torch.testing.assert_close(values[1], torch.tensor([0., 1.]), atol=1.e-6, rtol=0.)
        self.assertEqual(values[2][0].item(), 0.)
        self.assertGreater(values[2][1].item(), 0.)
        offset = torch.tensor([4., -6., 2.])
        shifted = shaping(grasp + offset, cube + offset, bowl + offset, initial + offset[2])
        for a, b in zip(values, shifted):
            torch.testing.assert_close(a, b, atol=1.e-5, rtol=1.e-5)

    def test_terminal_bonus_outweighs_stationary_dense_return(self):
        # RewardManager integrates shaping at 0.02 seconds; PPO gamma=0.98.
        self.assertGreater(200., (1. + 15. + 16.) * 0.02 / (1. - 0.98))


if __name__ == "__main__":
    unittest.main()
