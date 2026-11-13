'''文件工具'''"""会话配置、状态及事件记录的文件工具。

与具体任务、policy 和仿真器无关。
同一文件由一个写入者负责，不提供并发写入协调。

JSON 文件保存一份完整状态；JSONL 文件每行是一份 JSON，适合追加事件。
调用方只需传入路径和数据，不用重复处理目录创建、编码和换行。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    """通过同目录临时文件替换，避免读到写了一半的 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    # 例如 result.json 先写成 result.json.tmp，写完再替换正式文件。
    # 临时文件与正式文件在同一目录，使 replace 能进行原子替换：
    # 读者看到旧版或新版，不会读到本次写入的一半内容。
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        # 保留中文并缩进两格，便于直接打开文件阅读。
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, data: Any) -> None:
    """追加一条 JSON 记录，并刷新到磁盘。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    # 不使用缩进，保证一条事件只占一行；字符串中的换行会被 JSON 转义。
    record = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    # a 是追加模式，保留已有时间线；与 write_json 的“替换整份状态”不同。
    with path.open("a", encoding="utf-8") as stream:
        stream.write(record + "\n")
        # flush 先把 Python 缓冲送到操作系统，fsync 再要求同步到存储设备。
        stream.flush()
        os.fsync(stream.fileno())
