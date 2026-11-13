# 系统启动入口
"""启动一次源仿真器校准会话。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .session_runner import run_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="启动 AgenticSimulator 校准会话。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="框架 JSON 配置文件路径",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="新的运行目录，名称以 YYYYMMDDHHMM 开头",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        run_dir = args.run_dir.expanduser().resolve()
        result = run_session(cfg, run_dir)
    except KeyboardInterrupt:
        print("运行已中断。", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(f"无法完成运行：{exc}", file=sys.stderr)
        return 1

    # 输出宿主侧运行摘要；完整记录由 runner 保存。
    print(
        json.dumps(
            {"run_dir": str(run_dir), **result},
            ensure_ascii=False,
            indent=2,
        )
    )

    if result.get("reason") == "interrupted":
        return 130

    completed = (
        result.get("reason") == "session_finished"
        and result.get("agent_returncode") == 0
        and not result.get("cleanup_errors")
    )
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
