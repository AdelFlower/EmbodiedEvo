"""CPU regression tests for masked sequence training and GRU checkpoints."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("torch", "yaml"))
if HAS_DEPS:
    import torch

    from AgenticSimulator.tasks.pick_cube.RL.BC.dataset import collate_episodes
    from AgenticSimulator.tasks.pick_cube.RL.BC.eval_bc import load_policy
    from AgenticSimulator.tasks.pick_cube.RL.BC.model import GRUBCPolicy
    from AgenticSimulator.tasks.pick_cube.RL.BC.schema import (
        INITIAL_PREVIOUS_ACTION, OBS_SLICES, SCHEMA_VERSION,
    )
    from AgenticSimulator.tasks.pick_cube.RL.BC.train import parse_args, run_epoch, train


@unittest.skipUnless(HAS_DEPS, "GRU training tests require torch and pyyaml")
class BCGRUTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.args = SimpleNamespace(grip_loss_weight=1.0, force_loss_weight=1.0)
        self.mean = torch.zeros(48)
        self.std = torch.ones(48)

    def episode(self, length, offset=0):
        obs = torch.zeros(length, 48)
        obs[:, 0] = torch.arange(length) * 0.01 + offset
        for field in ("ee_pose", "cube_pose", "plate_pose"):
            obs[:, OBS_SLICES[field].start + 3] = 1
        actions = torch.tensor(INITIAL_PREVIOUS_ACTION).repeat(length, 1)
        actions[:, 0] = torch.arange(length) * 0.001
        actions[1:, 6:] = torch.tensor([1.0, 0.0])
        obs[0, 40:] = torch.tensor(INITIAL_PREVIOUS_ACTION)
        obs[1:, 40:] = actions[:-1]
        return obs, actions

    def epoch(self, model, batches, optimizer=None):
        return run_epoch(model, batches, torch.device("cpu"), self.mean, self.std,
                         self.args, optimizer=optimizer)

    def test_padding_does_not_change_loss_or_parameter_update(self):
        original = collate_episodes([self.episode(2), self.episode(5, 1)])
        obs, actions, mask = original
        # More padding with arbitrary values must have no effect on training.
        padded_obs = torch.full((2, 8, 48), 123.0)
        padded_actions = torch.full((2, 8, 8), -456.0)
        padded_mask = torch.zeros((2, 8), dtype=torch.bool)
        padded_obs[:, :5] = obs
        padded_actions[:, :5] = actions
        padded_mask[:, :5] = mask
        padded_obs[~padded_mask] = 123.0
        padded_actions[~padded_mask] = -456.0
        model = GRUBCPolicy(hidden_dim=8, recurrent_dim=7)
        other = copy.deepcopy(model)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        first = self.epoch(model, [original], torch.optim.SGD(model.parameters(), lr=0.01))
        second = self.epoch(other, [(padded_obs, padded_actions, padded_mask)],
                            torch.optim.SGD(other.parameters(), lr=0.01))
        for key in first:
            self.assertAlmostEqual(first[key], second[key], places=6)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, other.state_dict()[key])
        self.assertTrue(any(not torch.equal(value, before[key])
                            for key, value in model.state_dict().items()))

    def test_metrics_weight_frames_and_reset_memory_between_batches(self):
        episodes = [self.episode(2), self.episode(5, 1)]
        model = GRUBCPolicy(hidden_dim=8, recurrent_dim=7)
        batches = [collate_episodes([episode]) for episode in episodes]
        separate = [self.epoch(model, [batch]) for batch in batches]
        combined = self.epoch(model, batches)
        together = self.epoch(model, [collate_episodes(episodes)])
        for key in combined:
            expected = (separate[0][key] * 2 + separate[1][key] * 5) / 7
            self.assertAlmostEqual(combined[key], expected, places=6)
            self.assertAlmostEqual(together[key], expected, places=6)

    def test_empty_mask_is_rejected(self):
        obs, actions, mask = collate_episodes([self.episode(2)])
        with self.assertRaisesRegex(ValueError, "没有有效帧"):
            self.epoch(GRUBCPolicy(hidden_dim=8, recurrent_dim=7),
                       [(obs, actions, torch.zeros_like(mask))])

    def test_train_checkpoint_reloads_for_sequential_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            for index, length in enumerate((2, 5, 3)):
                obs, actions = self.episode(length, index)
                np.savez(data / f"episode_{index}.npz", observations=obs.numpy(),
                         actions=actions.numpy(), schema_version=SCHEMA_VERSION, success=True)
            run_dir = root / "202611130000_gru_train"
            run_dir.mkdir()
            args = SimpleNamespace(
                seed=42, device="cpu", dataset=data, batch_size=2, val_fraction=0.34,
                num_workers=0, hidden_dim=8, recurrent_dim=7,
                position_scale=0.02, rotation_scale=0.2, lr=0.001, epochs=2,
                grip_loss_weight=1.0, force_loss_weight=1.0,
            )
            train(args, run_dir)
            for filename in ("best.pt", "last.pt"):
                checkpoint = torch.load(run_dir / "checkpoints" / filename, weights_only=True)
                self.assertEqual(checkpoint["model_type"], "gru")
                self.assertEqual(checkpoint["model_config"]["recurrent_dim"], 7)
                model, mean, std, metadata = load_policy(run_dir / "checkpoints" / filename, "cpu")
                self.assertEqual(metadata["model_type"], "gru")
                norm = checkpoint["normalization"]
                torch.testing.assert_close(mean, norm["obs_mean"])
                torch.testing.assert_close(std, norm["obs_std"])
                obs, _ = self.episode(5)
                obs = ((obs - norm["obs_mean"]) / norm["obs_std"]).unsqueeze(0)
                sequence, final_hidden = model.predict(obs)
                hidden = None
                steps = []
                for t in range(obs.shape[1]):
                    action, hidden = model.predict(obs[:, t], hidden)
                    steps.append(action)
                torch.testing.assert_close(torch.stack(steps, dim=1), sequence)
                torch.testing.assert_close(hidden, final_hidden)
                self.assertTrue(torch.isfinite(sequence).all())
            self.assertEqual(len((run_dir / "metrics.csv").read_text().splitlines()), 3)

    def test_cli_defaults_and_recurrent_dim_override(self):
        argv = ["train", "--dataset", "unused", "--run-dir", "202611130000_test"]
        with patch("sys.argv", argv):
            args = parse_args()
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.recurrent_dim, 128)
        with patch("sys.argv", argv + ["--recurrent-dim", "32", "--batch-size", "4"]):
            args = parse_args()
        self.assertEqual(args.recurrent_dim, 32)
        self.assertEqual(args.batch_size, 4)


if __name__ == "__main__":
    unittest.main()
