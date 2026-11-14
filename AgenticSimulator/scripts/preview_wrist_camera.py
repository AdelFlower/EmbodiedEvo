"""复用 lift_cube adapter，保存初始相机画面和一个 policy 回合的双视角视频。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import traceback


def main() -> None:
    from isaaclab.app import AppLauncher

    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=package_root / "policies/lift_cube/model.pt",
    )
    parser.add_argument("--params-dir", type=Path, help="默认使用 checkpoint 同级的 params")
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    prefix = run_dir.name[:12]
    if len(prefix) != 12 or not prefix.isascii() or not prefix.isdigit():
        parser.error("--run-dir 名称必须以 YYYYMMDDHHMM 开头")
    try:
        datetime.strptime(prefix, "%Y%m%d%H%M")
    except ValueError:
        parser.error("--run-dir 的时间前缀不是有效日期")
    run_dir.mkdir(parents=True, exist_ok=False)

    # 直接执行本脚本时也能导入仓库内的 AgenticSimulator 包。
    sys.path.insert(0, str(package_root.parent))
    args.enable_cameras = True
    launcher = None
    adapter = None
    try:
        launcher = AppLauncher(args)

        # Isaac Sim 启动后再导入任务依赖。
        from PIL import Image
        from AgenticSimulator.tasks.lift_cube.adapter import LiftCubeAdapter

        config = {
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "device": args.device,
            "record_video": True,
        }
        if args.params_dir is not None:
            config["params_dir"] = str(args.params_dir.expanduser().resolve())
        adapter = LiftCubeAdapter(config)
        adapter.open()

        env = adapter.env.unwrapped
        env.seed(args.seed)
        adapter.env.reset()
        # 创建 overview annotator，并让渲染器预热；此处不推进物理。
        env.render(recompute=True)
        for _ in range(5):
            env.sim.render()
        for view, frame in adapter._video_frames().items():
            path = run_dir / f"{view}.png"
            Image.fromarray(frame).save(path)
            print(f"[preview] 初始画面：{path}", flush=True)

        # 复用完整 rollout，观察腕部相机随手臂运动后的取景。
        episode_dir = run_dir / "episode"
        result = adapter.rollout(
            {"episode_id": 0, "seed": args.seed}, episode_dir,
        )
        for view, filename in result["artifacts"]["videos"].items():
            print(f"[preview] {view} 视频：{episode_dir / filename}", flush=True)
    except BaseException:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        try:
            if adapter is not None:
                adapter.close()
        finally:
            if launcher is not None:
                launcher.app.close()


if __name__ == "__main__":
    main()
