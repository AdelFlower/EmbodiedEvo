"""读取 pick-cube 专家轨迹，为离线 BC 提供原始观测和动作。

数据目录可包含 shards/*.h5（每个文件含多条 episode），也兼容原有
episode_*.npz 或 episodes/*.npz。每条完整 episode 包含：

- observations: (T, 48)，按 schema 排列的观测。
- actions: (T, 8)，前六维为 m/rad，力度分量已归一化至 [-1, 1]。
- schema_version: 标量字符串，与 SCHEMA_VERSION 一致。
- success: 标量布尔值，记录该 episode 是否成功。

按整条 episode 划分训练集和验证集，仅从训练集计算观测统计量。
每个样本是一条完整轨迹；DataLoader 返回补齐后的观测、动作和有效帧 mask。
HDF5 中 complete=False 的轨迹始终排除，并记录在 split 中。
本模块不修改数据文件、不标准化观测，也不缩放动作训练标签。
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .schema import (
    ACTION_DIM,
    GRIP_CLOSE,
    GRIP_COMMAND_INDEX,
    GRIP_FORCE_INDEX,
    GRIP_OPEN,
    OBS_DIM,
    OBS_SLICES,
    SCHEMA_VERSION,
)


@dataclass(frozen=True)
class EpisodeRef:
    path: Path
    group: str | None = None

    def label(self, root):
        name = self.path.relative_to(root).as_posix()
        return name if self.group is None else f"{name}::{self.group}"


def _as_ref(path):
    return path if isinstance(path, EpisodeRef) else EpisodeRef(Path(path))


def _read_h5_metadata(handle, group, path):
    if handle.attrs.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version 必须为 {SCHEMA_VERSION}")
    complete = group.attrs.get("complete", False)
    if not isinstance(complete, (bool, np.bool_)):
        raise ValueError(f"{path}: complete 必须为布尔值")
    if not complete:
        return None
    success = group.attrs.get("success")
    if not isinstance(success, (bool, np.bool_)):
        raise ValueError(f"{path}: success 必须为布尔值")
    for key in ("observations", "actions"):
        if key not in group:
            raise ValueError(f"{path}: 缺少字段 {key}")
    return bool(success)


def _read_metadata(archive, path: Path) -> bool:
    """检查协议版本，返回原始成功标志；不推断或改写任务结果。"""
    for key in ("observations", "actions", "schema_version", "success"):
        if key not in archive.files:
            raise ValueError(f"{path}: 缺少字段 {key}")

    version = archive["schema_version"]
    if version.shape != () or version.item() != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version 必须为 {SCHEMA_VERSION}")

    success = archive["success"]
    if success.shape != () or success.dtype.kind != "b":
        raise ValueError(f"{path}: success 必须是标量布尔值")

    return bool(success.item())


def _read_array(archive, key: str, dimension: int, path: Path) -> np.ndarray:
    array = np.asarray(archive[key])
    if array.ndim != 2 or array.shape[1] != dimension or array.shape[0] == 0:
        raise ValueError(f"{path}: {key} 必须具有形状 (T, {dimension})，且 T > 0")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{path}: {key} 必须为实数数组")

    # 按训练使用的精度检查，拒绝原始异常和转换后的溢出。
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: {key} 包含 NaN/Inf 或超出 float32 范围")

    return np.ascontiguousarray(array)


def _validate_action_values(actions: np.ndarray, name: str, path: Path):
    grip = actions[:, GRIP_COMMAND_INDEX]
    if not np.isin(grip, (GRIP_OPEN, GRIP_CLOSE)).all():
        raise ValueError(f"{path}: {name} 的夹爪开合必须为 0 或 1")

    force = actions[:, GRIP_FORCE_INDEX]
    if ((force < -1.0) | (force > 1.0)).any():
        raise ValueError(f"{path}: {name} 的归一化力度必须位于 [-1, 1]")


class ExpertDataset(Dataset):
    """将轨迹载入 CPU 内存，每次返回一条按时间排列的完整 episode。"""

    def __init__(self, episode_paths: Sequence[Path | EpisodeRef]):
        if not episode_paths:
            raise ValueError("至少需要一条 episode")

        observations = []
        actions = []

        for episode_path in episode_paths:
            ref = _as_ref(episode_path)
            path = ref.path
            if ref.group is None:
                with np.load(path, allow_pickle=False) as archive:
                    _read_metadata(archive, path)
                    obs = _read_array(archive, "observations", OBS_DIM, path)
                    act = _read_array(archive, "actions", ACTION_DIM, path)
            else:
                import h5py

                with h5py.File(path, "r") as handle:
                    group = handle[ref.group]
                    if _read_h5_metadata(handle, group, ref) is None:
                        raise ValueError(f"{ref}: episode 尚未完成")
                    obs = _read_array(group, "observations", OBS_DIM, path)
                    act = _read_array(group, "actions", ACTION_DIM, path)

            if obs.shape[0] != act.shape[0]:
                raise ValueError(f"{path}: 观测与动作的时间步数量不一致")

            _validate_action_values(act, "actions", path)
            _validate_action_values(
                obs[:, OBS_SLICES["previous_action"]], "previous_action", path
            )

            for field in ("ee_pose", "cube_pose", "plate_pose"):
                start = OBS_SLICES[field].start + 3
                norms = np.linalg.norm(obs[:, start:start + 4], axis=1)
                if not np.allclose(norms, 1.0, atol=1e-3, rtol=0):
                    raise ValueError(f"{ref}: {field} 必须包含单位四元数")

            observations.append(torch.from_numpy(obs))
            actions.append(torch.from_numpy(act))

        self.episode_offsets = [0]
        for episode in observations:
            self.episode_offsets.append(self.episode_offsets[-1] + len(episode))

        # 连续存储真实帧，归一化统计不包含 collate 时添加的 padding。
        self.observations = torch.cat(observations, dim=0)
        self.actions = torch.cat(actions, dim=0)

    def __len__(self) -> int:
        return len(self.episode_offsets) - 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("episode index out of range")
        start, end = self.episode_offsets[index:index + 2]
        return self.observations[start:end], self.actions[start:end]

    @property
    def num_frames(self) -> int:
        """真实帧总数，与 __len__ 返回的 episode 数区分。"""
        return self.observations.shape[0]


def collate_episodes(batch):
    """返回 observations[B,T,48]、actions[B,T,8]、bool mask[B,T]。

    在每条轨迹末尾补零。mask=True 为真实帧；调用方须在计算损失和
    指标时排除补齐帧，并在每个完整 episode 开始时重置循环网络状态。
    """
    observations, actions = zip(*batch)
    lengths = torch.tensor([len(obs) for obs in observations], dtype=torch.long)
    observations = pad_sequence(observations, batch_first=True, padding_value=0.0)
    actions = pad_sequence(actions, batch_first=True, padding_value=0.0)
    mask = torch.arange(observations.shape[1])[None, :] < lengths[:, None]
    return observations, actions, mask


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 16,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    success_only: bool = True,
) -> tuple[DataLoader, DataLoader, dict, dict]:
    """返回训练 loader、验证 loader、观测统计量和可保存的划分记录。

    默认仅使用明确标记为成功的轨迹，排除的失败轨迹列入 split。
    至少需要两条符合筛选条件的 episode，保证两组均非空。
    batch_size 为 episode 数；shuffle 只打乱轨迹顺序，不打乱轨迹内的帧。
    两个 loader 均返回 (observations, actions, mask)，形状见 collate_episodes。
    """
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size 必须大于零，num_workers 不能小于零")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction 必须位于 (0, 1)")
    if seed < 0:
        raise ValueError("seed 不能小于零")

    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    episode_dir = root / "episodes" if (root / "episodes").is_dir() else root
    eligible_paths = []
    excluded_episodes = []
    candidates = []
    for path in sorted(episode_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            success = _read_metadata(archive, path)
        candidates.append((EpisodeRef(path), success))
    shard_dir = root / "shards" if (root / "shards").is_dir() else root
    for path in sorted(shard_dir.glob("*.h5")):
        import h5py

        with h5py.File(path, "r") as handle:
            for name, group in sorted(handle["episodes"].items()):
                ref = EpisodeRef(path, f"episodes/{name}")
                success = _read_h5_metadata(handle, group, ref)
                candidates.append((ref, success))
    if not candidates:
        raise FileNotFoundError(f"{root}: 未找到 NPZ 或 HDF5 episode")
    for ref, success in candidates:
        if success is None or (success_only and not success):
            excluded_episodes.append({
                "file": ref.label(root),
                "success": success,
                "reason": "incomplete" if success is None else "failed",
            })
        else:
            eligible_paths.append(ref)

    if len(eligible_paths) < 2:
        raise ValueError("至少需要两条符合筛选条件的 episode，不能按帧拆分替代")

    order = np.random.default_rng(seed).permutation(len(eligible_paths))
    val_count = min(
        len(eligible_paths) - 1,
        max(1, math.ceil(len(eligible_paths) * val_fraction)),
    )
    val_paths = [eligible_paths[int(i)] for i in order[:val_count]]
    train_paths = [eligible_paths[int(i)] for i in order[val_count:]]

    train_dataset = ExpertDataset(train_paths)
    val_dataset = ExpertDataset(val_paths)

    # 总体方差；验证集不参与计算，单样本也不会产生无偏估计的 NaN。
    variance, obs_mean = torch.var_mean(
        train_dataset.observations, dim=0, unbiased=False
    )
    obs_std = variance.sqrt()
    # 常量或近似常量维度使用 1，避免除以极小值放大微小偏差。
    obs_std = torch.where(obs_std < 1e-6, torch.ones_like(obs_std), obs_std)
    stats = {"obs_mean": obs_mean, "obs_std": obs_std}

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
        collate_fn=collate_episodes,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_episodes,
    )

    split = {
        "schema_version": SCHEMA_VERSION,
        "data_dir": str(root),
        "seed": seed,
        "val_fraction": val_fraction,
        "success_only": success_only,
        "train_episodes": [p.label(root) for p in train_paths],
        "val_episodes": [p.label(root) for p in val_paths],
        "excluded_episodes": excluded_episodes,
        "train_samples": train_dataset.num_frames,
        "val_samples": val_dataset.num_frames,
    }

    return train_loader, val_loader, stats, split
