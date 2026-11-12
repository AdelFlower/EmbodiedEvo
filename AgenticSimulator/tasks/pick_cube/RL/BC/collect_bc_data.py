"""批量采集 pick-cube 专家轨迹：200 Hz 物理、50 Hz 动作、每指 10 N。

从仓库根目录，使用 Isaac Lab Python 环境运行：
python -m AgenticSimulator.tasks.pick_cube.RL.BC.collect_bc_data \
    --run-dir 202611131200_collect --num-envs 1 --num-episodes 1 --gui
"""

import argparse
import json
import math
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .schema import SCHEMA_VERSION

PHYSICS_HZ = 200
CONTROL_HZ = 50
DECIMATION = PHYSICS_HZ // CONTROL_HZ
DATA_ROOT = Path(__file__).resolve().parent / "data"
SIMULATOR_ROOT = Path(__file__).resolve().parents[4]


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="data/ 下的新目录；只写目录名时自动放在 BC/data/ 下")
    parser.add_argument("--scene-config", type=Path,
                        default=SIMULATOR_ROOT / "scenes/configs/plate_cube.yaml")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-episodes", type=int, default=128,
                        help="尝试总数，包含成功和失败；不会重试直到成功")
    parser.add_argument("--episodes-per-shard", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-spacing", type=float, default=2.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--gui", action="store_true", help="打开窗口人工检查")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True, device="cuda:0")
    args = parser.parse_args()
    if min(args.num_envs, args.num_episodes, args.episodes_per_shard) < 1:
        parser.error("环境数、episode 数和分片容量必须为正整数")
    if not 0 <= args.seed < 2**63:
        parser.error("seed 必须位于 [0, 2**63)")
    if not math.isfinite(args.env_spacing) or args.env_spacing < 2:
        parser.error("env-spacing 必须至少为 2 m")
    if not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error("warmup-seconds 必须是有限非负数")
    if not args.device.startswith("cuda"):
        parser.error("采集需要 GPU 物理，请使用 --device cuda:0 等设备")
    if args.enable_cameras:
        parser.error("当前采集的是状态观测，不启用相机传感器；人工检查使用 --gui")
    if args.gui:
        args.headless = False
    path = args.run_dir.expanduser()
    args.run_dir = (DATA_ROOT / path if path.parent == Path(".") else path).resolve()
    if not args.run_dir.is_relative_to(DATA_ROOT.resolve()):
        parser.error("run-dir 必须位于 AgenticSimulator/tasks/pick_cube/RL/BC/data/ 下")
    prefix = args.run_dir.name[:12]
    try:
        if len(prefix) != 12 or not prefix.isdigit():
            raise ValueError
        datetime.strptime(prefix, "%Y%m%d%H%M")
    except ValueError:
        parser.error("run-dir 名称必须以有效的 YYYYMMDDHHMM 开头")
    args.scene_config = args.scene_config.expanduser().resolve()
    return args


def collect(args, config, report, writer):
    """AppLauncher 成功后才导入 torch/Isaac 仿真模块。"""
    import isaaclab.sim as sim_utils
    import torch

    from .control import BCExpertController
    from .scene import build_scene, reset_scene, sample_objects

    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=1 / PHYSICS_HZ, render_interval=DECIMATION, device=args.device,
    ))
    if "camera" in config:
        sim.set_camera_view(eye=tuple(config["camera"]["eye"]),
                            target=tuple(config["camera"]["target"]))
    scene = build_scene(config, args.num_envs, args.env_spacing)
    sim.reset()
    robot = scene["robot"]
    finger_ids = [robot.find_joints(f"panda_finger_joint{i}")[0][0] for i in (1, 2)]
    finger_gains = (robot.data.joint_stiffness[:, finger_ids].clone(),
                    robot.data.joint_damping[:, finger_ids].clone())
    physics_dt = sim.get_physics_dt()
    if not math.isclose(physics_dt, 1 / PHYSICS_HZ, rel_tol=1e-6):
        raise RuntimeError(f"实际物理步长不符合配置：{physics_dt}")

    for batch_start in range(0, args.num_episodes, args.num_envs):
        count = min(args.num_envs, args.num_episodes - batch_start)
        episode_ids = list(range(batch_start, batch_start + count))
        samples = sample_objects(config, range(batch_start, batch_start + args.num_envs), args.seed)
        controller = None
        finished = set()
        initial_states = None
        for env_id, episode_id in enumerate(episode_ids):
            writer.begin(episode_id, {
                "seed": args.seed, "episode_id": episode_id, "env_id": env_id,
                "initial_object_poses_env": {name: poses[env_id].tolist()
                                             for name, poses in samples.items()},
                "physics_hz": PHYSICS_HZ, "control_hz": CONTROL_HZ,
                "grip_force_per_finger_n": 10.0,
            })
            report["started"] += 1
        save_report(args.run_dir, report)
        try:
            reset_scene(scene, samples, finger_ids, finger_gains)
            scene.update(physics_dt)
            for warmup_step in range(round(args.warmup_seconds * PHYSICS_HZ)):
                if not args.app.is_running():
                    raise RuntimeError("仿真窗口在预热期间关闭")
                robot.set_joint_effort_target(robot.root_physx_view.get_gravity_compensation_forces())
                scene.write_data_to_sim()
                sim.step(render=not args.headless and warmup_step % DECIMATION == 0)
                scene.update(physics_dt)
            controller = BCExpertController(robot, scene["cube"], scene["plate"], sim.device)
            # 最后一批多余的环境仅保持姿态，不算作额外尝试。
            controller.state[count:] = controller.DONE
            initial_states = {
                "joint_pos": robot.data.joint_pos[:count].cpu().tolist(),
                "cube_root_state_w": scene["cube"].data.root_state_w[:count].cpu().tolist(),
                "plate_root_state_w": scene["plate"].data.root_state_w[:count].cpu().tolist(),
            }
            for env_id, episode_id in enumerate(episode_ids):
                writer.groups[episode_id].attrs["settled_initial_state_json"] = json.dumps({
                    key: value[env_id] for key, value in initial_states.items()
                })
            writer.flush()
            tick = 0
            print(
                f"[collect] batch={batch_start // args.num_envs} "
                f"episodes={episode_ids[0]}..{episode_ids[-1]}", flush=True,
            )
            while len(finished) < count:
                if not args.app.is_running():
                    raise RuntimeError("仿真窗口在采集期间关闭")
                observations = controller.observation()
                actions = controller.decide(1 / CONTROL_HZ)
                active = (~controller.terminal[:count]).cpu().numpy()
                # 批量搬到 CPU，不在每个环境上单独同步 GPU。
                packed = torch.cat((observations, actions), dim=-1)[:count].cpu().numpy()
                phases = controller.state[:count].cpu().numpy()
                for env_id in np.flatnonzero(active):
                    writer.append(episode_ids[env_id], packed[env_id, :48], packed[env_id, 48:],
                                  tick / CONTROL_HZ, phases[env_id])
                if active.any():
                    for substep in range(DECIMATION):
                        controller.physics_step(physics_dt)
                        scene.write_data_to_sim()
                        sim.step(render=not args.headless and substep == 0)
                        scene.update(physics_dt)
                terminal = controller.terminal[:count].cpu().numpy()
                newly_finished = [i for i in np.flatnonzero(terminal) if i not in finished]
                if newly_finished:
                    outcomes = controller.outcomes()
                    for env_id in newly_finished:
                        outcome = outcomes[env_id]
                        writer.finish(episode_ids[env_id], outcome)
                        finished.add(env_id)
                        report["succeeded" if outcome["success"] else "failed"] += 1
                        print(
                            f"[collect] episode={episode_ids[env_id]} "
                            f"success={outcome['success']} reason={outcome['reason']}", flush=True,
                        )
                    save_report(args.run_dir, report)
                tick += 1
                if tick % CONTROL_HZ == 0:
                    writer.flush()
        finally:
            outcomes = controller.outcomes() if controller is not None else None
            for env_id, episode_id in enumerate(episode_ids):
                if env_id in finished:
                    continue
                outcome = outcomes[env_id] if outcomes is not None else {
                    "env_id": env_id, "success": None, "reason": "interrupted_before_controller",
                }
                # 原始终止结果保留；未终止的 success 始终为 None。
                writer.finish(episode_id, outcome)
                if outcome["success"] is None:
                    report["interrupted"] += 1
                else:
                    report["succeeded" if outcome["success"] else "failed"] += 1


def save_report(run_dir, report):
    report["not_started"] = report["requested_episodes"] - report["started"]
    path = run_dir / "result.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    try:
        import h5py  # noqa: F401
    except ImportError as exc:
        raise SystemExit("请在 Isaac Lab 环境安装 BC/requirements.txt 中的 h5py") from exc
    config_text = args.scene_config.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    if sorted(item["name"] for item in config["objects"]) != ["Cube", "Plate"]:
        raise ValueError("采集场景必须恰好包含 Cube 和 Plate")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "scene_config.yaml").write_text(config_text, encoding="utf-8")
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    settings.update(schema_version=SCHEMA_VERSION, physics_hz=PHYSICS_HZ,
                    control_hz=CONTROL_HZ, grip_force_per_finger_n=10.0)
    (args.run_dir / "config.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    from .storage import ShardWriter

    report = {"status": "started", "requested_episodes": args.num_episodes,
              "started": 0, "succeeded": 0, "failed": 0, "interrupted": 0}
    save_report(args.run_dir, report)
    writer = None
    app = None
    try:
        from isaaclab.app import AppLauncher

        writer = ShardWriter(args.run_dir, args.episodes_per_shard)
        app = AppLauncher(args).app
        args.app = app
        collect(args, config, report, writer)
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc))
        (args.run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            save_report(args.run_dir, report)
            if app is not None:
                app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
