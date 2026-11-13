"""CPU checks for episode boundaries, padding and train-only normalization."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_H5PY = importlib.util.find_spec("h5py") is not None
if HAS_TORCH:
    import torch

    from AgenticSimulator.tasks.pick_cube.RL.BC.dataset import (
        EpisodeRef, ExpertDataset, build_dataloaders, collate_episodes,
    )
    from AgenticSimulator.tasks.pick_cube.RL.BC.schema import (
        INITIAL_PREVIOUS_ACTION, OBS_DIM, OBS_SLICES, SCHEMA_VERSION,
    )


@unittest.skipUnless(HAS_TORCH, "BC sequence tests require torch")
class BCSequenceTests(unittest.TestCase):
    def arrays(self, length, offset):
        obs = np.zeros((length, OBS_DIM), dtype=np.float32)
        obs[:, 0] = offset + np.arange(length)
        for name in ("ee_pose", "cube_pose", "plate_pose"):
            obs[:, OBS_SLICES[name].start + 3] = 1
        actions = np.tile(np.array(INITIAL_PREVIOUS_ACTION, dtype=np.float32), (length, 1))
        actions[:, 0] = np.arange(length) * 0.001
        obs[0, OBS_SLICES["previous_action"]] = INITIAL_PREVIOUS_ACTION
        obs[1:, OBS_SLICES["previous_action"]] = actions[:-1]
        return obs, actions

    def write_npz(self, root, index, length, success=True):
        path = root / f"episode_{index}.npz"
        obs, actions = self.arrays(length, index * 10)
        np.savez(path, observations=obs, actions=actions,
                 schema_version=SCHEMA_VERSION, success=success)
        return path

    def test_whole_episodes_keep_order_and_previous_action_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [self.write_npz(Path(tmp), i, length)
                     for i, length in enumerate((2, 5))]
            dataset = ExpertDataset(paths)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.num_frames, 7)
            for i, length in enumerate((2, 5)):
                obs, actions = dataset[i]
                expected_obs, expected_actions = self.arrays(length, i * 10)
                np.testing.assert_array_equal(obs.numpy(), expected_obs)
                np.testing.assert_array_equal(actions.numpy(), expected_actions)
                torch.testing.assert_close(obs[1:, 40:], actions[:-1])
            torch.testing.assert_close(dataset[-1][0], dataset[1][0])
            for index in (2, -3):
                with self.assertRaises(IndexError):
                    dataset[index]

    def test_padding_mask_preserves_valid_frames(self):
        batch = [tuple(torch.from_numpy(a) for a in self.arrays(n, i * 10))
                 for i, n in enumerate((2, 5, 1))]
        obs, actions, mask = collate_episodes(batch)
        self.assertEqual(obs.shape, (3, 5, 48))
        self.assertEqual(actions.shape, (3, 5, 8))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.sum(1).tolist(), [2, 5, 1])
        torch.testing.assert_close(obs[mask], torch.cat([item[0] for item in batch]))
        torch.testing.assert_close(actions[mask], torch.cat([item[1] for item in batch]))
        self.assertEqual(obs[~mask].count_nonzero().item(), 0)
        self.assertEqual(actions[~mask].count_nonzero().item(), 0)

    def test_split_counts_and_statistics_exclude_validation_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, length in enumerate((2, 5, 3, 1)):
                self.write_npz(root, i, length, success=i != 3)
            train, val, stats, split = build_dataloaders(root, batch_size=2)
            self.assertEqual(len(train.dataset), 2)
            self.assertEqual(len(val.dataset), 1)
            self.assertEqual(split["train_samples"] + split["val_samples"], 10)
            self.assertEqual(split["excluded_episodes"][0]["reason"], "failed")
            self.assertTrue(set(split["train_episodes"]).isdisjoint(split["val_episodes"]))
            raw = train.dataset.observations
            torch.testing.assert_close(stats["obs_mean"], raw.mean(0))
            expected_std = raw.std(0, unbiased=False)
            expected_std[expected_std < 1e-6] = 1
            torch.testing.assert_close(stats["obs_std"], expected_std)
            for loader, key in ((train, "train_samples"), (val, "val_samples")):
                obs, actions, mask = next(iter(loader))
                self.assertEqual(mask.sum().item(), split[key])
                self.assertEqual(obs.shape[:2], actions.shape[:2])
                self.assertEqual(obs.shape[:2], mask.shape)

    @unittest.skipUnless(HAS_H5PY, "HDF5 test requires h5py")
    def test_hdf5_groups_remain_separate_episodes(self):
        import h5py

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard.h5"
            with h5py.File(path, "w") as handle:
                handle.attrs["schema_version"] = SCHEMA_VERSION
                for i, length in enumerate((2, 5)):
                    group = handle.create_group(f"episodes/{i}")
                    group.attrs.update(complete=True, success=True)
                    obs, actions = self.arrays(length, i * 10)
                    group.create_dataset("observations", data=obs)
                    group.create_dataset("actions", data=actions)
            dataset = ExpertDataset([EpisodeRef(path, f"episodes/{i}") for i in range(2)])
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.num_frames, 7)
            self.assertEqual([len(dataset[i][0]) for i in range(2)], [2, 5])
            np.testing.assert_array_equal(dataset[1][0].numpy(), self.arrays(5, 10)[0])


if __name__ == "__main__":
    unittest.main()
