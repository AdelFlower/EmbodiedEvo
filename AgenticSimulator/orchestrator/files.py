'''文件工具'''"""会话配置、状态及事件记录的文件工具。

与具体任务、policy 和仿真器无关。
同一文件由一个写入者负责，不提供并发写入协调。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    """通过同目录临时文件替换，避免读到写了一半的 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, data: Any) -> None:
    """追加一条 JSON 记录，并刷新到磁盘。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    record = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    with path.open("a", encoding="utf-8") as stream:
        stream.write(record + "\n")
        stream.flush()
        os.fsync(stream.fileno())