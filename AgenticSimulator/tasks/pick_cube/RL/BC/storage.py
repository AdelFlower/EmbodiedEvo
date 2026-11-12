"""HDF5 分片写入；未完成的 episode 留在原分片中，但不会进入训练。"""

import json
from pathlib import Path

import numpy as np

from .schema import ACTION_DIM, OBS_DIM, SCHEMA_VERSION


class ShardWriter:
    def __init__(self, run_dir, episodes_per_shard=128):
        import h5py

        if episodes_per_shard < 1:
            raise ValueError("episodes_per_shard 必须大于零")
        self.h5py = h5py
        self.run_dir = Path(run_dir)
        self.shard_dir = self.run_dir / "shards"
        self.shard_dir.mkdir()
        self.episodes_per_shard = episodes_per_shard
        self.files = {}
        self.groups = {}
        self.buffers = {}
        self.finalized = set()
        self.manifest = (self.run_dir / "manifest.jsonl").open("x", encoding="utf-8")

    def _event(self, event):
        self.manifest.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.manifest.flush()

    def begin(self, episode_id, metadata):
        if episode_id in self.groups:
            raise ValueError(f"episode 已存在：{episode_id}")
        shard_id = episode_id // self.episodes_per_shard
        if shard_id not in self.files:
            path = self.shard_dir / f"shard_{shard_id:05d}.h5"
            handle = self.h5py.File(path, "x")
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["storage_version"] = "episode-groups-v1"
            handle.create_group("episodes")
            self.files[shard_id] = handle
        handle = self.files[shard_id]
        group = handle["episodes"].create_group(f"{episode_id:08d}")
        group.attrs["complete"] = False
        group.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        for key, dimension in (("observations", OBS_DIM), ("actions", ACTION_DIM)):
            group.create_dataset(
                key, shape=(0, dimension), maxshape=(None, dimension),
                dtype="f4", chunks=(256, dimension), compression="lzf",
            )
        for key, dtype in (("simulation_time", "f8"), ("expert_state", "i2")):
            group.create_dataset(key, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(256,))
        self.groups[episode_id] = group
        self.buffers[episode_id] = []
        handle.flush()
        self._event({
            "event": "started", "episode_id": episode_id,
            "file": Path(handle.filename).relative_to(self.run_dir).as_posix(),
            "group": group.name, "metadata": metadata,
        })

    def append(self, episode_id, observation, action, simulation_time, expert_state):
        if episode_id in self.finalized:
            raise ValueError("不能继续写入已结束的 episode")
        observation = np.asarray(observation, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if observation.shape != (OBS_DIM,) or action.shape != (ACTION_DIM,):
            raise ValueError("观测/动作维度与 schema 不一致")
        if not np.isfinite(observation).all() or not np.isfinite(action).all():
            raise ValueError("观测/动作包含 NaN/Inf")
        self.buffers[episode_id].append((
            observation.copy(), action.copy(), float(simulation_time), int(expert_state),
        ))

    def _flush_episode(self, episode_id):
        rows = self.buffers[episode_id]
        if not rows:
            return
        group = self.groups[episode_id]
        values = (
            np.stack([row[0] for row in rows]),
            np.stack([row[1] for row in rows]),
            np.array([row[2] for row in rows]),
            np.array([row[3] for row in rows]),
        )
        for key, array in zip(
            ("observations", "actions", "simulation_time", "expert_state"), values,
            strict=True,
        ):
            dataset = group[key]
            start = len(dataset)
            dataset.resize(start + len(rows), axis=0)
            dataset[start:] = array
        rows.clear()

    def flush(self):
        for episode_id in self.groups:
            self._flush_episode(episode_id)
        for handle in self.files.values():
            handle.flush()

    def finish(self, episode_id, outcome):
        if episode_id in self.finalized:
            return
        self._flush_episode(episode_id)
        group = self.groups[episode_id]
        complete = outcome["success"] is not None
        group.attrs["outcome_json"] = json.dumps(outcome, ensure_ascii=False)
        if complete:
            group.attrs["success"] = bool(outcome["success"])
        group.attrs["complete"] = complete
        group.file.flush()
        self._event({
            "event": "finished" if complete else "interrupted",
            "episode_id": episode_id,
            "samples": len(group["observations"]),
            "outcome": outcome,
        })
        self.finalized.add(episode_id)

    def close(self):
        try:
            self.flush()
        finally:
            for handle in self.files.values():
                handle.close()
            self.manifest.close()
