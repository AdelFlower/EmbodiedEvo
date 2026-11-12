import argparse
import importlib
from pathlib import Path

import yaml

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description="预览 YAML 指定的场景")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scene-config", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    config_path = Path(args.scene_config).resolve()
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "scene_config.yaml").write_text(
        config_text,
        encoding="utf-8",
    )

    launcher = AppLauncher(args)
    app = launcher.app

    try:
        # 必须在 AppLauncher 启动之后导入。
        import isaaclab.sim as sim_utils

        from AgenticSimulator.scene_api import SceneBundle

        module_name, function_name = config["builder"].split(":", 1)
        module = importlib.import_module(module_name)
        builder = getattr(module, function_name)

        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1 / 120,
                device=args.device,
            )
        )

        camera = config["camera"]
        sim.set_camera_view(
            eye=tuple(camera["eye"]),
            target=tuple(camera["target"]),
        )

        scene = builder(config)
        if not isinstance(scene, SceneBundle):
            raise TypeError("场景构建函数必须返回 SceneBundle")

        sim.reset()
        scene.initialize()

        sim.stage.Export(str(run_dir / "scene.usda"))

        print("[preview] 场景初始化完成", flush=True)
        dt = sim.get_physics_dt()

        while app.is_running():
            if scene.preview_step is not None:
                scene.preview_step(dt)

            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)

    finally:
        app.close()


if __name__ == "__main__":
    main()