import argparse
import csv
import json
import math
import os
import re
import resource
import sys
import threading
import time
import traceback
from pathlib import Path

import psutil
import pynvml
import yaml

from isaaclab.app import AppLauncher


class ResourceMonitor:
    """采样本进程的 GPU 显存和 RSS，单位 MiB。"""

    def __init__(self, output, gpu_index):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.process = psutil.Process()
        self.phase = "startup"
        self.error = None
        self.samples = []

        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.total_mib = info.total / 1024**2

        self.file = output.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "elapsed_s",
            "phase",
            "process_gpu_used_mib",
            "process_rss_mib",
        ])

        self.started = time.perf_counter()
        self.event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        row = [
            time.perf_counter() - self.started,
            self.phase,
            self._process_gpu_mib(),
            self.process.memory_info().rss / 1024**2,
        ]

        self.samples.append(row)
        self.writer.writerow(row)
        self.file.flush()

    def _run(self):
        try:
            while not self.event.is_set():
                self._sample()
                self.event.wait(0.5)
        except Exception:
            self.error = traceback.format_exc()

    def start(self):
        self.thread.start()

    def stop(self):
        self.event.set()
        self.thread.join()
        try:
            self._sample()
        except Exception:
            self.error = traceback.format_exc()
        finally:
            self.file.close()
            pynvml.nvmlShutdown()

    def summary(self):
        return {
            "pid": self.process.pid,
            "gpu_total_mib": self.total_mib,
            "sampled_process_gpu_peak_mib": max(
                (row[2] for row in self.samples),
                default=None,
            ),
            "sampled_process_rss_peak_mib": max(
                (row[3] for row in self.samples),
                default=None,
            ),
            "sampled_measure_process_gpu_peak_mib": max(
                (row[2] for row in self.samples if row[1] == "measure"),
                default=None,
            ),
            "sampled_measure_process_rss_peak_mib": max(
                (row[3] for row in self.samples if row[1] == "measure"),
                default=None,
            ),
            "resource_monitor_error": self.error,
        }

    def _process_gpu_mib(self):
        values = []
        matched = False
        supported = False

        queries = (
            pynvml.nvmlDeviceGetComputeRunningProcesses,
            pynvml.nvmlDeviceGetGraphicsRunningProcesses,
        )

        for query in queries:
            try:
                processes = query(self.handle)
                supported = True
            except pynvml.NVMLError_NotSupported:
                continue

            for process in processes:
                if process.pid != self.process.pid:
                    continue

                matched = True
                used = process.usedGpuMemory

                # 部分平台无法报告进程显存，不能把未知值当成零。
                if used is None or used == (1 << 64) - 1:
                    continue

                values.append(int(used))

        if values:
            # 两个列表中的值都是应用显存，取最大值，避免重复计数。
            return max(values) / 1024**2

        if not supported:
            raise RuntimeError("当前 GPU 不支持 NVML 进程显存查询")

        if matched:
            raise RuntimeError("找到了当前进程，但显存数据不可用")

        # 启动早期还没有创建 GPU 上下文时，列表中不会出现该 PID。
        return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="盘子/方块场景的单进程批量抓放控制基准"
    )
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--grip-force", type=float, default=20.0,
        help="闭合和搬运时，每根手指向内施加的驱动力，单位 N",
    )
    parser.add_argument("--warmup-steps", type=int, default=600)
    parser.add_argument(
        "--steps", type=int, default=14400,
        help="测量步数上限；所有环境结束后提前停止（默认 120 秒仿真时间）",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-spacing", type=float, default=2.0)
    parser.add_argument("--monitor-gpu", type=int, default=0)
    parser.add_argument("--direct-exit", action="store_true")

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cuda:0", headless=True)
    args = parser.parse_args()

    if args.num_envs < 1 or args.steps < 1 or args.warmup_steps < 0:
        parser.error("环境数、测量步数必须为正，预热步数不能为负")
    if not 0.0 < args.grip_force <= 200.0:
        parser.error("--grip-force 必须大于 0，且不超过当前配置的 200 N 上限")
    if not args.device.startswith("cuda"):
        parser.error("这个基准要求使用 GPU 物理")
    if args.env_spacing < 2.0:
        parser.error("当前基准要求 env-spacing 至少为 2.0")
    if args.enable_cameras:
        parser.error("这一轮不测相机，请不要传 --enable_cameras")
    if not re.match(r"^[0-9]{12}", Path(args.run_dir).name):
        parser.error("run-dir 目录名必须以 YYYYMMDDHHMM 开头")

    config_text = Path(args.scene_config).read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    objects = {item["name"]: item for item in config["objects"]}

    if set(objects) != {"Cube", "Plate"}:
        parser.error("这个基准要求 YAML 中恰好包含 Cube 和 Plate")

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "scene_config.yaml").write_text(
        config_text, encoding="utf-8"
    )

    report = {
        "status": "started",
        "num_envs": args.num_envs,
        "device": args.device,
        "seed": args.seed,
        "render_each_step": False,
        "workload": "batch_pick_cube_no_cameras",
        "gripper_control": "constant_joint_force_close_carry_pd_open",
        "grip_force_per_finger_n": args.grip_force,
        "requested_steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "warmup_workload": "hold_default_pose_with_gravity_compensation",
        "controller_first_use_included_in_timing": True,
        "terminal_check_interval_steps": 120,
        "task_status_note": "status 表示基准执行状态，抓放成功情况见 task",
    }
    report_path = run_dir / "result.json"

    def save_report():
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    save_report()

    app = None
    monitor = None
    startup_begin = time.perf_counter()

    try:
        monitor = ResourceMonitor(
            run_dir / "resources.csv", args.monitor_gpu
        )
        monitor.start()

        launcher = AppLauncher(args)
        app = launcher.app

        # Isaac Sim 启动后才能导入这些模块。
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
        from AgenticSimulator.tasks.pick_cube.batch_controller import (
            BatchPickCubeController,
        )

        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.device)

        assets = Path(__file__).resolve().parents[1] / "assets"

        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1 / 120,
                device=args.device,
            )
        )

        # 每个物体加载后，显式补充质量和物理材质。
        # InteractiveScene 随后会复制整个源环境。
        def make_object_spawn(item):
            def spawn(prim_path, cfg, translation=None, orientation=None):
                prim = sim_utils.spawn_from_usd(
                    prim_path,
                    cfg,
                    translation=translation,
                    orientation=orientation,
                )
                path = str(prim.GetPath())

                sim_utils.define_mass_properties(
                    path,
                    sim_utils.MassPropertiesCfg(
                        mass=float(item["physics"]["mass"]),
                        density=0.0,
                    ),
                )

                material_path = f"{path}/BenchmarkPhysicsMaterial"
                material = sim_utils.RigidBodyMaterialCfg(
                    static_friction=item["material"]["static_friction"],
                    dynamic_friction=item["material"]["dynamic_friction"],
                    restitution=item["material"]["restitution"],
                    friction_combine_mode="average",
                    restitution_combine_mode="average",
                )
                material.func(material_path, material)
                sim_utils.bind_physics_material(path, material_path)
                return prim

            return spawn

        scene_cfg = InteractiveSceneCfg(
            num_envs=args.num_envs,
            env_spacing=args.env_spacing,
            replicate_physics=True,
            filter_collisions=True,
        )

        # 全局共享的地面和灯光。
        scene_cfg.ground = AssetBaseCfg(
            prim_path="/World/Ground",
            collision_group=-1,
            spawn=sim_utils.GroundPlaneCfg(),
        )
        scene_cfg.light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0),
        )

        # 每个环境自己的固定桌面。
        scene_cfg.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=(1.2, 0.8, 0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.55, 0.40, 0.25)
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.725)
            ),
        )

        robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
        robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        robot_cfg.spawn.usd_path = str(
            assets / "robots" / "franka" / "Franka.usd"
        )
        robot_cfg.spawn.articulation_props.fix_root_link = True
        robot_cfg.spawn.rigid_props.disable_gravity = False
        robot_cfg.init_state.pos = (-0.45, 0.0, 0.75)
        robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        scene_cfg.robot = robot_cfg

        for name, item in objects.items():
            asset_path = assets / item["asset"]
            if not asset_path.is_file():
                raise FileNotFoundError(asset_path)

            physics = item["physics"]

            # 这是源环境的生成位置，初始化时会对各环境重新采样。
            if "position_range" in item:
                position = tuple(
                    sum(item["position_range"][axis]) / 2
                    for axis in ("x", "y", "z")
                )
            else:
                position = tuple(item["position"])

            object_cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=sim_utils.UsdFileCfg(
                    func=make_object_spawn(item),
                    usd_path=str(asset_path.resolve()),
                    scale=tuple(item["scale"]),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=False,
                        disable_gravity=not physics["gravity"],
                        linear_damping=physics["linear_damping"],
                        angular_damping=physics["angular_damping"],
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=position,
                    rot=tuple(
                        item.get("orientation", [1.0, 0.0, 0.0, 0.0])
                    ),
                ),
            )
            setattr(scene_cfg, name.lower(), object_cfg)

        scene = InteractiveScene(scene_cfg)
        sim.reset()

        robot = scene["robot"]
        if robot.data.joint_pos.shape[0] != args.num_envs:
            raise RuntimeError("机器人实例数与 num-envs 不一致")

        # 初始化所有环境的机器人世界坐标。
        root = robot.data.default_root_state.clone()
        root[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos.clone(),
            robot.data.default_joint_vel.clone(),
        )

        # 为每个环境独立采样物体位姿。
        for name, item in objects.items():
            obj = scene[name.lower()]
            if obj.data.root_pos_w.shape[0] != args.num_envs:
                raise RuntimeError(f"{name} 实例数不正确")

            state = obj.data.default_root_state.clone()

            for axis_id, axis in enumerate(("x", "y", "z")):
                if "position_range" in item:
                    low, high = item["position_range"][axis]
                    state[:, axis_id] = low + (high - low) * torch.rand(
                        args.num_envs, device=sim.device
                    )
                else:
                    state[:, axis_id] = item["position"][axis_id]

            if "yaw_range_deg" in item:
                low, high = item["yaw_range_deg"]
                yaw = (
                    low + (high - low)
                    * torch.rand(args.num_envs, device=sim.device)
                ) * math.pi / 180

                state[:, 3:7] = 0
                state[:, 3] = torch.cos(yaw / 2)
                state[:, 6] = torch.sin(yaw / 2)
            else:
                state[:, 3:7] = state.new_tensor(item["orientation"])

            state[:, :3] += scene.env_origins
            state[:, 7:] = 0

            obj.write_root_pose_to_sim(state[:, :7])
            obj.write_root_velocity_to_sim(state[:, 7:])

        scene.reset()
        scene.update(sim.get_physics_dt())

        torch.cuda.synchronize()
        report["startup_seconds"] = time.perf_counter() - startup_begin
        report["physics_dt"] = sim.get_physics_dt()
        save_report()

        hold_targets = robot.data.default_joint_pos.clone()
        controller = None
        dt = sim.get_physics_dt()
        step_index = 0

        def check_state():
            if not torch.isfinite(robot.data.joint_pos).all().item():
                raise RuntimeError("机器人关节状态出现 NaN/Inf")

            for name in ("cube", "plate"):
                state = scene[name].data.root_state_w
                if not torch.isfinite(state).all().item():
                    raise RuntimeError(f"{name} 状态出现 NaN/Inf")

                # 预热时掉落说明场景未准备好；任务中的掉落由
                # 控制器按环境记为失败，不能中断其他环境。
                if controller is None:
                    local_pos = state[:, :3] - scene.env_origins
                    if (local_pos[:, 2] < 0.6).any().item():
                        raise RuntimeError(f"{name} 掉出了桌面范围")

        def step_once():
            nonlocal step_index

            if controller is None:
                robot.set_joint_position_target(hold_targets)
                robot.set_joint_effort_target(
                    robot.root_physx_view.get_gravity_compensation_forces()
                )
            else:
                controller.step(dt)

            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            step_index += 1

            # 每个仿真秒检查一次状态。
            if step_index % 120 == 0:
                if not app.is_running():
                    raise RuntimeError("仿真应用提前退出")
                check_state()

        print(
            f"[benchmark] 已创建 {args.num_envs} 个环境，开始预热",
            flush=True,
        )
        monitor.phase = "warmup"

        for _ in range(args.warmup_steps):
            step_once()

        torch.cuda.synchronize()
        check_state()

        # 预热只保持初始姿态，不提前消耗抓放任务。
        monitor.phase = "controller_setup"
        allocated_before_constructor = torch.cuda.memory_allocated(sim.device)
        controller = BatchPickCubeController(
            robot=robot,
            cube=scene["cube"],
            plate=scene["plate"],
            device=sim.device,
            grip_force=args.grip_force,
        )
        torch.cuda.synchronize()
        allocated_at_start = torch.cuda.memory_allocated(sim.device)
        report["controller_constructor_torch_mib"] = (
            allocated_at_start - allocated_before_constructor
        ) / 1024**2
        report["measurement_torch_allocated_at_start_mib"] = (
            allocated_at_start / 1024**2
        )
        report["controller_memory_note"] = (
            "constructor 为构造时 PyTorch 已分配显存增量；measurement peak "
            "包含控制与物理步中经 PyTorch 分配的内存。NVML 进程峰值包含整个"
            "场景和 Isaac Sim，不能视为控制器独占显存。"
        )
        torch.cuda.reset_peak_memory_stats(sim.device)
        save_report()

        monitor.phase = "measure"
        print("[benchmark] 开始计时，执行批量抓放控制", flush=True)

        executed_steps = 0
        begin = time.perf_counter()
        for _ in range(args.steps):
            step_once()
            executed_steps += 1
            # 降低 GPU 到 CPU 的同步频率。结束环境保持姿态，不重置。
            if executed_steps % 120 == 0:
                if controller.terminal.all().item():
                    break

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        check_state()
        torch_peak = torch.cuda.max_memory_allocated(sim.device)
        monitor.phase = "measure_complete"
        task_summary = controller.summary()
        (run_dir / "outcomes.json").write_text(
            json.dumps(controller.outcomes(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report.update({
            "status": "ok",
            "measured_steps": executed_steps,
            "wall_seconds": elapsed,
            "batch_steps_per_second": executed_steps / elapsed,
            "env_steps_per_second": (
                args.num_envs * executed_steps / elapsed
            ),
            "active_env_steps_per_second": (
                task_summary["active_env_steps"] / elapsed
            ),
            "realtime_factor": executed_steps * dt / elapsed,
            "measurement_torch_peak_allocated_mib": torch_peak / 1024**2,
            "measurement_torch_peak_above_start_mib": (
                torch_peak - allocated_at_start
            ) / 1024**2,
            "task": task_summary,
        })

    except BaseException as exc:
        report.update({
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        (run_dir / "error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise

    finally:
        if monitor is not None:
            monitor.stop()
            report.update(monitor.summary())
            if monitor.error is not None and report["status"] == "ok":
                report["status"] = "resource_monitor_failed"

        # Linux 下 ru_maxrss 的单位是 KiB。
        report["process_peak_rss_mib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        report["shutdown_mode"] = (
            "direct_process_exit"
            if args.direct_exit
            else "app_close"
        )

        save_report()
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

        # CSV 和结果保存完成后，可跳过可能阻塞的原生关闭流程。
        if args.direct_exit:
            exit_code = 0 if report["status"] == "ok" else 1
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(exit_code)

        if app is not None:
            print("[benchmark] 正在关闭 Isaac Sim", flush=True)
            app.close(wait_for_replicator=False)
            print("[benchmark] 关闭调用已返回", flush=True)


if __name__ == "__main__":
    main()
