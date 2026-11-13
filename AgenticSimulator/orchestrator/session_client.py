# 本地通信客户端
"""基于 Unix socket 的会话通信工具。

只依赖 Python 标准库，可独立放入 Agent 工作区。
不读取实验脚本、不解释请求业务、不自动重试。
"""

from __future__ import annotations

import json
import os
import socket
from typing import BinaryIO


MAX_MESSAGE = 16 * 1024 * 1024


def receive(stream: BinaryIO) -> dict:
    """读取一条以换行符结束的 JSON 对象消息。"""
    line = stream.readline(MAX_MESSAGE + 1)

    if not line or len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
        raise ConnectionError(
            "Incomplete or oversized session message; "
            "execution may have happened. Do not retry automatically."
        )

    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")

    return value


def send(stream: BinaryIO, value: dict) -> None:
    """发送一条 JSON 对象消息。"""
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")

    data = (
        json.dumps(value, ensure_ascii=True).encode("utf-8") + b"\n"
    )
    if len(data) > MAX_MESSAGE:
        raise ValueError("Session message exceeds 16 MiB")

    stream.write(data)
    stream.flush()


def request(
    payload: dict,
    *,
    socket_path: str | None = None,
) -> dict:
    """建立连接，发送一次请求，接收一次响应。"""
    address = socket_path or os.environ.get("AGENTICSIMULATOR_SOCKET")
    if not address:
        raise RuntimeError(
            "AGENTICSIMULATOR_SOCKET is not set; "
            "run inside an initialized workspace"
        )

    # 断连不代表请求没有执行，因此不能自动重试。
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(address)

        with connection.makefile("rwb") as stream:
            send(stream, payload)
            return receive(stream)
