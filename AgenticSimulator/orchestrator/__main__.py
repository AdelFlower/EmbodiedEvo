# 系统启动入口
"""启动一次源仿真器校准会话。

这是整个 orchestrator（调度模块）的命令行入口。
运行 ``python -m AgenticSimulator.orchestrator`` 时，Python 会执行此文件。
--config 指向任务目录中的 task.toml，--run-dir 指定本次运行目录。
配置加载器解析 TOML，再由 runner 保存 JSON 运行快照并负责启动和收尾。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .session_runner import run_session


def main(argv: list[str] | None = None) -> int:
    """返回退出码：0 表示正常完成，1 表示失败，130 表示中断。

    argv=None 时使用终端传入的参数；测试也可以直接传入参数列表。
    """
    # argparse 会解析 --config 等选项，并自动生成 --help 帮助信息。
    parser = argparse.ArgumentParser(
        description="启动 AgenticSimulator 校准会话。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="任务 TOML 配置文件路径（task.toml）",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="新的运行目录，名称以 YYYYMMDDHHMM 开头",
    )
    args = parser.parse_args(argv)

    try:
        # 先校验配置，再创建运行。这里的 resolve() 将路径转成绝对路径，
        # expanduser() 则把 ~/ 这样的写法展开为当前用户的主目录。
        cfg = load_config(args.config)
        run_dir = args.run_dir.expanduser().resolve()
        result = run_session(cfg, run_dir)
    except KeyboardInterrupt:
        # 错误信息写到 stderr（标准错误），便于与正常的 JSON 输出区分。
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

    # 仅“runner 确认会话完成、Agent 正常退出、清理没有报错”同时满足，
    # 才把这次运行报告为成功；这里不判断仿真任务的物理效果。
    completed = (
        result.get("reason") == "session_finished"
        and result.get("agent_returncode") == 0
        and not result.get("cleanup_errors")
    )
    return 0 if completed else 1


if __name__ == "__main__":
    # 把 main 的整数返回值交给操作系统，方便 shell/批处理判断成败。
    raise SystemExit(main())
