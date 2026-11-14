"""源环境的仿真子进程入口，由 worker.py 启动。

用法：python -m AgenticSimulator.simulators.source --run-dir RUN --job JOB
RUN 是 worker 已创建的运行目录，名称以 YYYYMMDDHHMM 开头。
本进程等待 stdin 中的 start 后启动 Isaac Sim，再加载任务 adapter 和源物理候选。
source_dir 是 worker 为本次请求准备的候选运行副本，本入口不捕获或选择候选。

第一版只执行 rollout；source_probe 尚未实现，提交时会明确报错。
轨迹、视频由 adapter 记录；报告及日志由 worker 收集并发布。
候选路径校验只约束加载入口，不提供宿主文件访问隔离。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import traceback

from AgenticSimulator.orchestrator.files import write_json


def load_adapter(entry: str):
    """加载宿主指定的 module:Class，不在入口中写死具体任务。"""
    module_name, class_name = entry.split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)


def load_source_physics(source_dir: str, physics_path: str):
    """从候选运行副本加载物理模块，configure/before_step 由 adapter 调用。"""
    source = Path(source_dir).resolve()
    path = (source / physics_path).resolve()
    if not path.is_relative_to(source):
        raise ValueError("Source physics must stay inside the candidate")
    name = "_agenticsimulator_source_physics"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load source physics module: {path}")
    # 候选可以通过绝对模块名导入同一副本中的辅助模块。
    sys.path.insert(0, str(source))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_job(job: dict) -> dict:
    """执行一个源回合；保存原始异常，清理失败也使报告 ok=false。"""
    launcher = None
    adapter = None
    report = {"ok": False, "result": {}}

    def record_error(exc, stage):
        detail = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        report["ok"] = False
        if "error" not in report:
            report["error"] = detail
        else:
            # 清理错误不能覆盖最初的执行错误。
            report.setdefault("cleanup_errors", []).append(detail)
        # 立即保存失败报告及日志，避免 Kit 退出时丢失尚未输出的异常信息。
        write_json(Path(job["report_path"]), report)
        traceback.print_exc()
        sys.stderr.flush()

    try:
        from isaaclab.app import AppLauncher

        config = dict(job["adapter_config"])
        launcher = AppLauncher(
            headless=True,
            enable_cameras=config.get("record_video", True),
            device=config["device"],
        )
        Adapter = load_adapter(job["adapter"])
        physics = load_source_physics(job["source_dir"], job["physics"])
        adapter = Adapter(config)
        adapter.open(physics)
        report["result"] = adapter.rollout(job["episode"], Path(job["output_dir"]))
        report["ok"] = True
    except BaseException as exc:
        # 包括中断和候选代码提前退出；保留失败，不自动重跑。
        record_error(exc, "execution")
    finally:
        try:
            if adapter is not None:
                try:
                    adapter.close()
                except BaseException as exc:
                    record_error(exc, "adapter_close")
        finally:
            if launcher is not None:
                try:
                    launcher.app.close()
                except BaseException as exc:
                    record_error(exc, "app_close")

    # 成功报告由 main 在两层清理都结束后写入。
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args(argv)
    run = args.run_dir.expanduser().resolve()
    prefix = run.name[:12]
    if len(prefix) != 12 or not prefix.isascii() or not prefix.isdigit():
        parser.error("--run-dir must start with YYYYMMDDHHMM")
    try:
        datetime.strptime(prefix, "%Y%m%d%H%M")
    except ValueError:
        parser.error("--run-dir has an invalid date prefix")

    job = json.loads(args.job.expanduser().resolve().read_text(encoding="utf-8"))
    report_path = Path(job["report_path"])

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        if job["environment"] != "source":
            raise ValueError("Source entry accepts only source jobs")
        if job["op"] != "rollout":
            raise ValueError("Source entry supports only rollout; source_probe is not implemented")
        # worker 在进程启动成功、预算登记完成后发送 start；EOF 不启动实验。
        if sys.stdin.readline().strip() != "start":
            raise RuntimeError("Source job cancelled before receiving start")
        report = run_job(job)
    except BaseException as exc:
        report = {
            "ok": False,
            "result": {},
            "error": {
                "stage": "entry",
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        traceback.print_exc()
        sys.stderr.flush()

    write_json(report_path, report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
