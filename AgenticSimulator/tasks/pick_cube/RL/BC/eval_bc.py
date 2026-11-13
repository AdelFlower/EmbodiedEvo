"""单场景 GRU BC 闭环预览：默认打开 GUI，结束后保持画面。

从仓库根目录使用 Isaac Lab Python 环境，以 python -m ...BC.eval_bc 运行。
控制动作只来自 checkpoint；专家阶段和目标路点不参与推理。
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .schema import ACTION_DIM, GRIP_CLOSE, GRIP_FORCE_INDEX, OBS_DIM, SCHEMA_VERSION

BC_ROOT = Path(__file__).resolve().parent
PHYSICS_HZ = 200
CONTROL_HZ = 50
DECIMATION = PHYSICS_HZ // CONTROL_HZ


def parse_args(argv=None):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BC_ROOT / "config.yaml",
                        help="读取 YAML 的 eval 区块；命令行参数优先")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="新运行目录；只写目录名时放入 BC/evals/，必须以 YYYYMMDDHHMM 开头")
    parser.add_argument("--scene-config", type=Path,
                        default=BC_ROOT.parents[3] / "scenes/configs/plate_cube.yaml")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--episode-id", type=int, default=0,
                        help="与 seed 一起确定物体采样；一次运行只评估一个场景")
    parser.add_argument("--max-seconds", type=float, default=30.0,
                        help="BC rollout 总仿真时间上限，非专家的逐阶段超时")
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--gui", action="store_true", help="显式打开 GUI（默认已打开）")
    parser.add_argument("--exit-on-done", action=argparse.BooleanOptionalAction, default=False,
                        help="保存结果后自动结束进程；默认 GUI 保持画面直到关闭窗口或 Ctrl+C")
    parser.add_argument("--no-real-time", action="store_true", help="GUI 不限制为实时速度")
    parser.add_argument("--real-time", dest="no_real_time", action="store_false",
                        help="GUI 按实时速度播放，可覆盖配置中的 no_real_time")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=False, device="cuda:0", no_real_time=False)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=BC_ROOT / "config.yaml")
    config_args, _ = config_parser.parse_known_args(argv)
    config_path = config_args.config.expanduser().resolve()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        parser.error(f"无法读取配置文件 {config_path}: {exc}")
    if config is None:
        config = {}
    if not isinstance(config, dict) or not isinstance(config.get("eval", {}), dict):
        parser.error("配置文件顶层和 eval 必须为键值映射")
    configurable = {
        "scene_config": str, "device": str, "seed": int, "episode_id": int,
        "max_seconds": float, "warmup_seconds": float,
        "headless": bool, "exit_on_done": bool, "no_real_time": bool,
    }
    defaults = {}
    for key, value in config.get("eval", {}).items():
        if key not in configurable:
            parser.error(f"eval 配置项 {key!r} 不受支持")
        expected = configurable[key]
        allowed = (int, float) if expected is float else (expected,)
        if type(value) not in allowed:
            parser.error(f"eval 配置项 {key!r} 应为 {expected.__name__} 类型")
        defaults[key] = Path(value) if key == "scene_config" else expected(value)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.config = config_path
    if not 0 <= args.seed < 2**63 or args.episode_id < 0:
        parser.error("seed 必须位于 [0, 2**63)，episode-id 不能小于零")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("max-seconds 必须为有限正数")
    if not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error("warmup-seconds 必须为有限非负数")
    if not args.device.startswith("cuda"):
        parser.error("本入口使用 GPU 物理，请指定 --device cuda:0 等设备")
    if args.enable_cameras:
        parser.error("GUI 使用视口渲染，不需要 --enable_cameras")
    if args.gui:
        args.headless = False
    if args.headless:
        args.exit_on_done = True
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.scene_config = args.scene_config.expanduser().resolve()
    if not args.checkpoint.is_file() or not args.scene_config.is_file():
        parser.error("checkpoint 或 scene-config 文件不存在")
    path = args.run_dir.expanduser()
    args.run_dir = (BC_ROOT / "evals" / path if path.parent == Path(".") else path).resolve()
    if not args.run_dir.is_relative_to(BC_ROOT):
        parser.error("run-dir 必须位于 BC 目录内")
    prefix = args.run_dir.name[:12]
    try:
        if len(prefix) != 12 or not prefix.isdigit():
            raise ValueError
        datetime.strptime(prefix, "%Y%m%d%H%M")
    except ValueError:
        parser.error("run-dir 名称必须以有效的 YYYYMMDDHHMM 开头")
    return args


def load_policy(checkpoint_path, device):
    """CPU 测试可直接调用；实际推理在 AppLauncher 启动后调用。"""
    import torch

    from .model import GRUBCPolicy

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint schema_version 与当前观测/动作协议不一致")
    if checkpoint.get("obs_dim") != OBS_DIM or checkpoint.get("action_dim") != ACTION_DIM:
        raise ValueError("checkpoint 的观测/动作维度不一致")
    if checkpoint.get("model_type") != "gru":
        raise ValueError("评估入口需要 model_type='gru' 的新 checkpoint；旧 MLP 权重不能用于 GRU")
    model_config = checkpoint["model_config"]
    for key in ("hidden_dim", "recurrent_dim"):
        if type(model_config.get(key)) is not int or model_config[key] <= 0:
            raise ValueError(f"checkpoint model_config.{key} 必须为正整数")
    policy = GRUBCPolicy(**model_config)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not all(torch.isfinite(value).all().item() for value in policy.state_dict().values()):
        raise ValueError("checkpoint 权重包含 NaN/Inf")
    if not (policy.pose_scale > 0).all().item():
        raise ValueError("checkpoint 动作缩放尺度必须为正")
    normalization = checkpoint["normalization"]
    mean = torch.as_tensor(normalization["obs_mean"], dtype=torch.float32)
    std = torch.as_tensor(normalization["obs_std"], dtype=torch.float32)
    if mean.shape != (OBS_DIM,) or std.shape != (OBS_DIM,):
        raise ValueError("checkpoint 观测统计量必须为 48 维")
    if not torch.isfinite(mean).all().item() or not torch.isfinite(std).all().item():
        raise ValueError("checkpoint 观测统计量包含 NaN/Inf")
    if not (std > 0).all().item():
        raise ValueError("checkpoint 观测标准差必须大于零")
    metadata = {
        "model_type": "gru",
        "schema_version": checkpoint["schema_version"], "epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "model_config": checkpoint["model_config"], "pose_scale": policy.pose_scale.tolist(),
    }
    return policy.to(device).eval(), mean.to(device), std.to(device), metadata


def fixed_force_action(predicted):
    """固定每指 10 N；previous_action 必须记录实际执行的 0/-1 力度标签。"""
    import torch

    if predicted.shape != (1, ACTION_DIM) or not torch.isfinite(predicted).all().item():
        raise ValueError("模型动作必须是有限值数组 (1, 8)")
    if not ((predicted[:, 6] == 0) | (predicted[:, 6] == 1)).all().item():
        raise ValueError("predict() 的夹爪输出必须是 0/1")
    action = predicted.clone()
    action[:, GRIP_FORCE_INDEX] = torch.where(action[:, 6] == GRIP_CLOSE, 0.0, -1.0)
    return action


class PlacementMonitor:
    """只读物理结果，不生成控制目标、不调用专家阶段。

    沿用原任务：抬升至少 6 cm、抓取距离不超过 5 cm，放置底部中心
    XY < 3 cm、垂直差 < 1 cm 且连续稳定 > 0.5 s。用实际抬升和释放
    代替专家的 LIFT/VERIFY 阶段门控；BC 使用显式的全局 rollout 时间上限。
    """

    def __init__(self):
        self.seen_lift = False
        self.seen_release = False
        self.stable_seconds = 0.0

    def update(self, dt, *, cube_rise, grasp_distance, xy_distance, height_error, closed):
        if closed and cube_rise >= 0.06 and grasp_distance <= 0.05:
            self.seen_lift = True
        if self.seen_lift and not closed:
            self.seen_release = True
        placed = xy_distance < 0.03 and height_error < 0.01
        verifying = self.seen_release and not closed and placed
        self.stable_seconds = self.stable_seconds + dt if verifying else 0.0
        return self.stable_seconds > 0.5

    def state(self):
        return {"seen_lift": self.seen_lift, "seen_release": self.seen_release,
                "placement_stable_seconds": self.stable_seconds}


class RolloutTrace:
    def __init__(self):
        self.rows = []

    def append(self, observation, predicted, action, simulation_time):
        self.rows.append((observation.copy(), predicted.copy(), action.copy(), simulation_time))

    def save(self, path):
        if self.rows:
            obs, predicted, actions, timestamps = zip(*self.rows, strict=True)
            arrays = {"observations": np.stack(obs), "predicted_actions": np.stack(predicted),
                      "actions": np.stack(actions), "simulation_time": np.array(timestamps)}
        else:
            arrays = {"observations": np.empty((0, OBS_DIM), dtype=np.float32),
                      "predicted_actions": np.empty((0, ACTION_DIM), dtype=np.float32),
                      "actions": np.empty((0, ACTION_DIM), dtype=np.float32),
                      "simulation_time": np.empty((0,))}
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, schema_version=SCHEMA_VERSION, **arrays)
        temporary.replace(path)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def scene_metrics(executor, initial_cube_z):
    import torch
    from isaaclab.utils.math import quat_apply

    cube, plate = executor.cube, executor.plate
    cube_bottom = cube.data.root_pos_w + quat_apply(
        cube.data.root_quat_w, cube.data.root_pos_w.new_tensor([[0., 0., -0.0125]]),
    )
    plate_bottom = plate.data.root_pos_w + quat_apply(
        plate.data.root_quat_w, plate.data.root_pos_w.new_tensor([[0., 0., -0.0178308]]),
    )
    difference = cube_bottom - plate_bottom
    values = torch.stack((
        cube.data.root_pos_w[0, 2] - initial_cube_z,
        (cube.data.root_pos_w - executor._grasp_position()).norm(),
        difference[0, :2].norm(), difference[0, 2].abs(),
    )).cpu().tolist()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("仿真状态包含 NaN/Inf")
    return dict(zip(("cube_rise", "grasp_distance", "xy_distance", "height_error"),
                    values, strict=True))


def snapshot(scene):
    return {
        "joint_pos": scene["robot"].data.joint_pos.cpu().tolist(),
        "joint_vel": scene["robot"].data.joint_vel.cpu().tolist(),
        "robot_root_state_w": scene["robot"].data.root_state_w.cpu().tolist(),
        "cube_root_state_w": scene["cube"].data.root_state_w.cpu().tolist(),
        "plate_root_state_w": scene["plate"].data.root_state_w.cpu().tolist(),
    }


def evaluate(args, config, policy, mean, std, report, trace):
    import isaaclab.sim as sim_utils
    import torch

    from .control import BCExpertController
    from .scene import build_scene, reset_scene, sample_objects

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=1 / PHYSICS_HZ, render_interval=DECIMATION, device=args.device,
    ))
    camera = config.get("camera")
    if camera:
        sim.set_camera_view(eye=tuple(camera["eye"]), target=tuple(camera["target"]))
    scene = build_scene(config, num_envs=1, env_spacing=2.0)
    sim.reset()
    robot = scene["robot"]
    finger_ids = [robot.find_joints(f"panda_finger_joint{i}")[0][0] for i in (1, 2)]
    gains = (robot.data.joint_stiffness[:, finger_ids].clone(),
             robot.data.joint_damping[:, finger_ids].clone())
    samples = sample_objects(config, [args.episode_id], args.seed)
    reset_scene(scene, samples, finger_ids, gains)
    physics_dt = sim.get_physics_dt()
    if not math.isclose(physics_dt, 1 / PHYSICS_HZ, rel_tol=1e-6):
        raise RuntimeError("实际物理频率必须为 200 Hz")
    scene.update(physics_dt)
    write_json(args.run_dir / "initial_state.json", snapshot(scene))
    for step in range(round(args.warmup_seconds * PHYSICS_HZ)):
        if not args.app.is_running():
            raise KeyboardInterrupt("预热期间关闭窗口")
        robot.set_joint_effort_target(robot.root_physx_view.get_gravity_compensation_forces())
        scene.write_data_to_sim()
        sim.step(render=not args.headless and step % DECIMATION == 0)
        scene.update(physics_dt)
    write_json(args.run_dir / "settled_initial_state.json", snapshot(scene))

    # 只使用 observation/set_action/physics_step；绝不调用 decide()/step()。
    executor = BCExpertController(robot, scene["cube"], scene["plate"], sim.device)
    monitor = PlacementMonitor()
    initial_cube_z = scene["cube"].data.root_pos_w[0, 2].clone()
    max_steps = math.ceil(args.max_seconds * PHYSICS_HZ)
    wall_start = time.perf_counter()
    physics_steps = 0
    # 与训练一致：预热不输入网络，新 episode 从零记忆开始。
    # 每个 50 Hz 动作仅更新一次隐藏状态；四个物理子步共用该动作。
    hidden = None
    report.update(status="running", success=None, reason="running")
    write_json(args.run_dir / "result.json", report)
    print("[eval] 模型闭环开始：50 Hz 动作、200 Hz 物理、固定每指 10 N", flush=True)
    while physics_steps < max_steps:
        if not args.app.is_running():
            raise KeyboardInterrupt("评估期间关闭窗口")
        observation = executor.observation()
        with torch.inference_mode():
            predicted, hidden = policy.predict((observation - mean) / std, hidden)
            if not torch.isfinite(hidden).all().item():
                raise ValueError("GRU 隐藏状态包含 NaN/Inf")
        action = fixed_force_action(predicted)
        trace.append(observation[0].cpu().numpy(), predicted[0].cpu().numpy(),
                     action[0].cpu().numpy(), physics_steps / PHYSICS_HZ)
        executor.set_action(action)
        closed = bool(action[0, 6].item() == GRIP_CLOSE)
        finished = False
        for substep in range(min(DECIMATION, max_steps - physics_steps)):
            executor.physics_step(physics_dt)
            scene.write_data_to_sim()
            sim.step(render=not args.headless and substep == 0)
            scene.update(physics_dt)
            physics_steps += 1
            report["simulation_seconds"] = physics_steps / PHYSICS_HZ
            report["control_steps"] = len(trace.rows)
            if executor.terminal[0].item():
                report.update(success=False, reason=executor.outcomes()[0]["reason"])
                finished = True
                break
            metrics = scene_metrics(executor, initial_cube_z)
            success = monitor.update(physics_dt, closed=closed, **metrics)
            report["metrics"] = {**metrics, **monitor.state()}
            if success:
                report.update(success=True, reason="success")
                finished = True
                break
        if finished:
            break
        if physics_steps % PHYSICS_HZ == 0:
            metrics = report["metrics"]
            print(f"[eval] t={report['simulation_seconds']:.1f}s "
                  f"grip={'close' if closed else 'open'} "
                  f"rise={metrics['cube_rise']:.3f}m "
                  f"placement_xy={metrics['xy_distance']:.3f}m", flush=True)
            trace.save(args.run_dir / "rollout.npz")
            write_json(args.run_dir / "result.json", report)
        if not args.headless and not args.no_real_time:
            delay = wall_start + physics_steps / PHYSICS_HZ - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    if report["success"] is None:
        report.update(success=False, reason="rollout_timeout")
    report.update(status="complete", wall_seconds=time.perf_counter() - wall_start)
    write_json(args.run_dir / "final_state.json", snapshot(scene))
    trace.save(args.run_dir / "rollout.npz")
    write_json(args.run_dir / "result.json", report)
    print(f"[eval] success={report['success']} reason={report['reason']} "
          f"simulation_seconds={report['simulation_seconds']:.3f}", flush=True)
    return sim


def exit_application(app, exit_code):
    # 结果和轨迹由 main 的 finally 先关闭并写入；避开已观察到的 Kit 关闭阻塞。
    # GUI 默认先保持画面；用户关闭窗口或指定 --exit-on-done 后才到这里。
    if app is not None:
        print("[eval] 结果已保存，结束仿真进程。", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


def main():
    args = parse_args()
    config_text = args.scene_config.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "scene_config.yaml").write_text(config_text, encoding="utf-8")
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    settings.update(physics_hz=PHYSICS_HZ, control_hz=CONTROL_HZ,
                    grip_force_per_finger_n=10.0,
                    termination="observed_lift_release_then_native_placement_geometry",
                    rollout_time_limit_seconds=args.max_seconds)
    write_json(args.run_dir / "config.json", settings)
    report = {"status": "started", "success": None, "reason": "not_started",
              "simulation_seconds": 0.0, "control_steps": 0}
    trace = RolloutTrace()
    app = None
    exit_code = 1
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(args).app
        args.app = app
        import torch

        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.device)
        policy, mean, std, metadata = load_policy(args.checkpoint, args.device)
        write_json(args.run_dir / "checkpoint_metadata.json", metadata)
        print(f"[eval] model={metadata['model_type']} checkpoint epoch={metadata['epoch']} "
              f"val_loss={metadata['best_val_loss']:.6f}", flush=True)
        sim = evaluate(args, config, policy, mean, std, report, trace)
        exit_code = 0
        if not args.headless and not args.exit_on_done:
            sim.pause()
            print("[eval] 结果已保存；画面暂停。关闭窗口或 Ctrl+C 结束。", flush=True)
            while app.is_running():
                sim.render()  # 处理 GUI 事件，不继续推进物理或调用模型。
                time.sleep(1 / 60)
    except KeyboardInterrupt as exc:
        if report["status"] != "complete":
            report.update(status="interrupted", success=None, reason=str(exc) or "user_interrupt")
            exit_code = 130
    except BaseException as exc:
        exit_code = 1
        report.update(status="error", error_type=type(exc).__name__, error=str(exc))
        (args.run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
    finally:
        try:
            trace.save(args.run_dir / "rollout.npz")
            write_json(args.run_dir / "result.json", report)
        finally:
            exit_application(app, exit_code)


if __name__ == "__main__":
    main()
