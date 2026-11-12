"""离线训练 pick-cube BC 策略，不启动模拟器。"""

import argparse
import csv
import json
import random
import traceback
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from .dataset import build_dataloaders
from .model import BCPolicy
from .schema import (
    ACTION_DIM,
    ACTION_SLICES,
    GRIP_CLOSE,
    GRIP_COMMAND_INDEX,
    GRIP_FORCE_INDEX,
    OBS_DIM,
    SCHEMA_VERSION,
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练 pick-cube BC 策略")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="YAML 默认配置；命令行参数优先",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")

    # 位姿训练标签的缩放尺度，predict() 中还原，不代表动作上限。
    parser.add_argument("--position-scale", type=float, default=0.01)
    parser.add_argument("--rotation-scale", type=float, default=0.1)
    parser.add_argument("--grip-loss-weight", type=float, default=1.0)
    parser.add_argument("--force-loss-weight", type=float, default=1.0)

    # 先定位配置文件，再将其值设为默认值，最后解析命令行覆盖项。
    # 此阶段暂不检查 dataset/run-dir 等必填项。
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    config_args, _ = config_parser.parse_known_args()
    config_path = config_args.config.expanduser().resolve()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        parser.error(f"无法读取配置文件 {config_path}: {exc}")

    if config is None:
        config = {}
    if not isinstance(config, dict):
        parser.error("配置文件顶层必须是键值映射")

    # YAML 使用与 argparse 相同的字段名，例如 batch_size。
    # 运行目录、数据来源和配置文件自身仍要求由命令行指定。
    configurable = {
        action.dest: action.type or str
        for action in parser._actions
        if action.dest not in {"help", "config", "dataset", "run_dir"}
    }
    for key, value in config.items():
        if key not in configurable:
            parser.error(f"配置项 {key!r} 不受支持")
        expected_type = configurable[key]
        allowed_types = (int, float) if expected_type is float else (expected_type,)
        if type(value) not in allowed_types:
            parser.error(f"配置项 {key!r} 应为 {expected_type.__name__} 类型")
        config[key] = expected_type(value)

    parser.set_defaults(**config)
    args = parser.parse_args()
    args.config = config_path

    if min(args.epochs, args.batch_size, args.hidden_dim) <= 0:
        parser.error("epochs、batch-size、hidden-dim 必须大于零")
    if args.lr <= 0 or args.num_workers < 0:
        parser.error("lr 必须大于零，num-workers 不能小于零")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("val-fraction 必须位于 (0, 1)")
    if min(args.position_scale, args.rotation_scale) <= 0:
        parser.error("动作缩放尺度必须大于零")
    if min(args.grip_loss_weight, args.force_loss_weight) < 0:
        parser.error("损失权重不能小于零")

    prefix = args.run_dir.name[:12]
    try:
        if len(prefix) != 12 or not prefix.isdigit():
            raise ValueError
        datetime.strptime(prefix, "%Y%m%d%H%M")
    except ValueError:
        parser.error("run-dir 名称必须以有效的 YYYYMMDDHHMM 时间前缀开头")

    return args


def compute_losses(predictions, targets, args):
    """计算位姿回归、夹爪分类和力度回归损失。"""
    pos = ACTION_SLICES["delta_pos"]
    rot = ACTION_SLICES["delta_rot"]

    position_loss = F.mse_loss(
        predictions[:, pos],
        targets[:, pos],
    )

    rotation_loss = F.mse_loss(
        predictions[:, rot],
        targets[:, rot],
    )

    grip_targets = targets[:, GRIP_COMMAND_INDEX]
    grip_loss = F.binary_cross_entropy_with_logits(
        predictions[:, GRIP_COMMAND_INDEX],
        grip_targets,
    )

    # 打开时力度不参与控制，因此不监督这些样本的力度。
    closed = (grip_targets == GRIP_CLOSE).to(predictions.dtype)
    force_errors = (
        predictions[:, GRIP_FORCE_INDEX]
        - targets[:, GRIP_FORCE_INDEX]
    ).square()
    force_loss = (force_errors * closed).mean()

    total_loss = (
        position_loss
        + rotation_loss
        + args.grip_loss_weight * grip_loss
        + args.force_loss_weight * force_loss
    )

    return {
        "loss": total_loss,
        "position_loss": position_loss,
        "rotation_loss": rotation_loss,
        "grip_loss": grip_loss,
        "force_loss": force_loss,
    }


def run_epoch(
    model,
    loader,
    device,
    obs_mean,
    obs_std,
    args,
    optimizer=None,
):
    """optimizer 为 None 时执行验证，不更新参数。"""
    training = optimizer is not None
    model.train(training)

    totals = {}
    sample_count = 0

    for observations, actions in loader:
        observations = observations.to(device=device, dtype=torch.float32)
        actions = actions.to(device=device, dtype=torch.float32)

        # 只标准化输入；动作标签保持 schema 中的定义。
        observations = (observations - obs_mean) / obs_std

        targets = actions.clone()
        targets[..., :6] = actions[..., :6] / model.pose_scale

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(observations)
            losses = compute_losses(predictions, targets, args)

            if not torch.isfinite(losses["loss"]).item():
                raise RuntimeError("损失出现 NaN/Inf，停止本次训练")

            if training:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

        batch_size = observations.shape[0]
        sample_count += batch_size

        for name, value in losses.items():
            totals[name] = (
                totals.get(name, 0.0)
                + value.detach().item() * batch_size
            )

    if sample_count == 0:
        raise ValueError("数据加载器为空，无法训练或验证")

    return {
        name: value / sample_count
        for name, value in totals.items()
    }


def train(args, run_dir):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    train_loader, val_loader, stats, split = build_dataloaders(
        data_dir=args.dataset,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    obs_mean = torch.as_tensor(
        stats["obs_mean"], dtype=torch.float32, device=device
    )
    obs_std = torch.as_tensor(
        stats["obs_std"], dtype=torch.float32, device=device
    ).clamp_min(1e-6)

    if obs_mean.shape != (OBS_DIM,) or obs_std.shape != (OBS_DIM,):
        raise ValueError("观测归一化参数必须为 48 维")
    if not (
        torch.isfinite(obs_mean).all().item()
        and torch.isfinite(obs_std).all().item()
    ):
        raise ValueError("观测归一化参数包含 NaN/Inf")

    normalization = {
        "obs_mean": obs_mean.cpu(),
        "obs_std": obs_std.cpu(),
    }

    (run_dir / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model = BCPolicy(hidden_dim=args.hidden_dim, position_scale=args.position_scale, rotation_scale=args.rotation_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()

    best_val_loss = float("inf")
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    with (run_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as metrics_file:
        writer = None

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, device,
                obs_mean, obs_std, args,
                optimizer=optimizer,
            )

            val_metrics = run_epoch(
                model, val_loader, device,
                obs_mean, obs_std, args,
            )

            row = {"epoch": epoch}
            row.update({
                f"train_{key}": value
                for key, value in train_metrics.items()
            })
            row.update({
                f"val_{key}": value
                for key, value in val_metrics.items()
            })

            if writer is None:
                writer = csv.DictWriter(
                    metrics_file, fieldnames=list(row)
                )
                writer.writeheader()

            writer.writerow(row)
            metrics_file.flush()

            is_best = val_metrics["loss"] < best_val_loss
            best_val_loss = min(best_val_loss, val_metrics["loss"])

            checkpoint = {
                "schema_version": SCHEMA_VERSION,
                "obs_dim": OBS_DIM,
                "action_dim": ACTION_DIM,
                "model_config": {"hidden_dim": args.hidden_dim},
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "normalization": normalization,
                "config": config,
                "split": split,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "hidden_dim": args.hidden_dim,
                "position_scale": args.position_scale,
                "rotation_scale": args.rotation_scale,
            }

            torch.save(checkpoint, checkpoint_dir / "last.pt")
            if is_best:
                torch.save(checkpoint, checkpoint_dir / "best.pt")

            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train={train_metrics['loss']:.6f} "
                f"val={val_metrics['loss']:.6f}",
                flush=True,
            )


def main():
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.run_dir = args.run_dir.resolve()

    # 创建独立运行目录，避免覆盖以前的训练记录。
    args.run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        train(args, args.run_dir)
    except Exception:
        (args.run_dir / "error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
