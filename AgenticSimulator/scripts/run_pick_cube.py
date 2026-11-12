import argparse
import importlib
import json
import traceback
from pathlib import Path

import yaml

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(
        description="运行规则控制的方块抓放任务"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scene-config", required=True)
    parser.add_argument(
    "--grip-force",
    type=float,
    default=20.0,
    help="闭合和搬运时，每根手指向内施加的驱动力，单位 N",)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 0.0 < args.grip_force <= 200.0:
        parser.error("--grip-force 必须大于 0, 且不超过当前配置的 200 N 上限")

    # 读取场景配置。
    config_path = Path(args.scene_config).resolve()
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)

    # 创建独立运行目录，不覆盖已有记录。
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "scene_config.yaml").write_text(
        config_text,
        encoding="utf-8",
    )

    launcher = AppLauncher(args)
    app = launcher.app

    try:
        # 必须在 Isaac Sim 启动后导入。
        import isaaclab.sim as sim_utils

        from AgenticSimulator.scene_api import SceneBundle
        from AgenticSimulator.tasks.pick_cube.controller import (
            PickCubeController,
        )

        # 根据 YAML 动态加载场景构建函数。
        module_name, function_name = config["builder"].split(":", 1)
        module = importlib.import_module(module_name)
        builder = getattr(module, function_name)

        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1 / 200,
                device=args.device,
            )
        )

        camera = config["camera"]
        sim.set_camera_view(
            eye=tuple(camera["eye"]),
            target=tuple(camera["target"]),
        )

        # 构建场景。
        scene = builder(config)
        if not isinstance(scene, SceneBundle):
            raise TypeError("场景构建函数必须返回 SceneBundle")

        required = {"robot", "cube", "plate"}
        missing = required - scene.entities.keys()
        if missing:
            raise ValueError(
                f"pick_cube 任务缺少场景对象：{sorted(missing)}"
            )

        # 初始化物理对象和机器人。
        sim.reset()
        scene.initialize()

        # 给控制器传入场景中已经存在的对象。
        controller = PickCubeController(
            robot=scene.entities["robot"],
            cube=scene.entities["cube"],
            plate=scene.entities["plate"],
            device=sim.device,
            grip_force=args.grip_force,
        )

        sim.stage.Export(str(run_dir / "scene.usda"))

        dt = sim.get_physics_dt()
        result_saved = False

        print("[pick_cube] 开始执行抓放任务", flush=True)

        while app.is_running():
            # 控制器设置关节目标，不在这里执行预览控制。
            controller.step(dt)

            # 提交控制目标。
            scene.write_data_to_sim()

            # 推进物理模拟。
            sim.step()

            # 更新所有实体的状态缓存。
            scene.update(dt)

            # 成功或失败都保存；不自动重试。
            if controller.result is not None and not result_saved:
                (run_dir / "result.json").write_text(
                    json.dumps(
                        controller.result,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                result_saved = True
                print(
                    "[pick_cube] 任务结束，结果已保存；"
                    "窗口保持打开。",
                    flush=True,
                )

    except Exception:
        # 保存异常信息，保留本次运行记录。
        (run_dir / "error.txt").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise

    finally:
        app.close()


if __name__ == "__main__":
    main()