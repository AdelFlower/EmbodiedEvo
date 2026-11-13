"""CPU checks for GRU evaluation loading and shared YAML/CLI configuration."""

import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("torch", "yaml"))
if HAS_DEPS:
    import torch
    import yaml

    from AgenticSimulator.tasks.pick_cube.RL.BC import eval_bc
    from AgenticSimulator.tasks.pick_cube.RL.BC.model import GRUBCPolicy
    from AgenticSimulator.tasks.pick_cube.RL.BC.schema import SCHEMA_VERSION
    from AgenticSimulator.tasks.pick_cube.RL.BC.train import parse_args as training_args


@unittest.skipUnless(HAS_DEPS, "GRU eval tests require torch and pyyaml")
class BCGRUEvalTests(unittest.TestCase):
    def test_execution_preserves_predicted_closed_force(self):
        predicted = torch.zeros(1, 8)
        predicted[0, 6:] = torch.tensor([1., 0.5])
        torch.testing.assert_close(eval_bc.executable_action(predicted), predicted)
        predicted[0, 6] = 0
        self.assertEqual(eval_bc.executable_action(predicted)[0, 7].item(), -1)

    def checkpoint(self):
        config = dict(hidden_dim=8, recurrent_dim=7, position_scale=0.02, rotation_scale=0.3)
        policy = GRUBCPolicy(**config)
        return {
            "model_type": "gru", "schema_version": SCHEMA_VERSION,
            "obs_dim": 48, "action_dim": 8, "model_config": config,
            "model_state_dict": policy.state_dict(), "epoch": 2, "best_val_loss": 0.1,
            "normalization": {"obs_mean": torch.arange(48).float() / 100,
                              "obs_std": torch.ones(48) * 2},
        }

    def test_loaded_gru_preserves_scales_and_memory_across_steps(self):
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            payload = self.checkpoint()
            torch.save(payload, path)
            policy, mean, std, metadata = eval_bc.load_policy(path, "cpu")
            self.assertEqual(metadata["model_type"], "gru")
            self.assertFalse(policy.training)
            torch.testing.assert_close(policy.pose_scale, torch.tensor([0.02]*3 + [0.3]*3))
            obs = (torch.randn(1, 5, 48) - mean) / std
            sequence, expected_hidden = policy.predict(obs)
            hidden = None
            actions = []
            for step in range(5):
                action, hidden = policy.predict(obs[:, step], hidden)
                actions.append(action)
            torch.testing.assert_close(torch.stack(actions, dim=1), sequence)
            torch.testing.assert_close(hidden, expected_hidden)
            self.assertFalse(hidden.requires_grad)
            # Starting another episode from None reproduces the first action.
            first, _ = policy.predict(obs[:, 0], None)
            torch.testing.assert_close(first, sequence[:, 0])

    def test_rejects_corrupt_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            payload = self.checkpoint()
            payload["model_state_dict"]["gru.weight_ih_l0"][0, 0] = float("nan")
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "NaN/Inf"):
                eval_bc.load_policy(path, "cpu")

    def parse_eval(self, argv):
        isaac = ModuleType("isaaclab")
        app = ModuleType("isaaclab.app")

        class AppLauncher:
            @staticmethod
            def add_app_launcher_args(parser):
                parser.add_argument("--device", default="cuda:0")
                parser.add_argument("--headless", action="store_true")
                parser.add_argument("--enable_cameras", action="store_true")

        app.AppLauncher = AppLauncher
        isaac.app = app
        with patch.dict("sys.modules", {"isaaclab": isaac, "isaaclab.app": app}):
            return eval_bc.parse_args(argv)

    def config_fixture(self, root):
        path = root / "config.yaml"
        scene = root / "scene.yaml"
        scene.write_text("objects: []\n")
        weights = root / "best.pt"
        weights.touch()
        path.write_text(yaml.safe_dump({
            "hidden_dim": 64, "recurrent_dim": 32, "batch_size": 4, "seed": 123,
            "device": "cpu", "eval": {
                "scene_config": str(scene), "device": "cuda:1", "seed": 44,
                "episode_id": 3, "max_seconds": 12, "warmup_seconds": 0.5,
                "headless": True, "exit_on_done": True, "no_real_time": True,
            },
        }))
        return path, ["--config", str(path), "--checkpoint", str(weights),
                      "--run-dir", "202611130000_eval_config_test"]

    def test_eval_config_and_cli_overrides_are_independent_of_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, argv = self.config_fixture(Path(tmp))
            args = self.parse_eval(argv)
            self.assertEqual((args.device, args.seed, args.episode_id), ("cuda:1", 44, 3))
            self.assertEqual((args.max_seconds, args.warmup_seconds), (12., 0.5))
            self.assertTrue(args.headless and args.exit_on_done and args.no_real_time)
            override = self.parse_eval(argv + ["--gui", "--no-exit-on-done", "--real-time",
                                                "--seed", "99", "--device", "cuda:0"])
            self.assertFalse(override.headless or override.exit_on_done or override.no_real_time)
            self.assertEqual((override.seed, override.device), (99, "cuda:0"))
            with patch("sys.argv", ["train", "--config", str(path), "--dataset", "unused",
                                    "--run-dir", "202611130000_train_config_test"]):
                args = training_args()
            self.assertEqual((args.device, args.seed, args.batch_size), ("cpu", 123, 4))
            self.assertEqual(args.recurrent_dim, 32)

    def test_eval_config_rejects_typos_and_wrong_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, argv = self.config_fixture(Path(tmp))
            for settings in ({"max_second": 10}, {"headless": "false"},
                             {"checkpoint": "implicit.pt"}, {"run_dir": "implicit"},
                             {"max_seconds": float("nan")}, {"seed": True}):
                with self.subTest(settings=settings):
                    path.write_text(yaml.safe_dump({"eval": settings}))
                    with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
                        self.parse_eval(argv)


if __name__ == "__main__":
    unittest.main()
