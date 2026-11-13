# 本地通信客户端
"""基于 Unix socket 的会话通信工具。

只依赖 Python 标准库，可独立放入 Agent 工作区。
不读取实验脚本、不解释请求业务、不自动重试。

Unix socket 是同一台机器上两个进程通信的通道，地址是一个文件系统路径。
双方约定“一行 JSON 对应一条消息”，一次连接只发送一个请求、读取一个响应。
本模块只负责传话；rollout、预算等业务含义由 session.py 解释。
"""

from __future__ import annotations

import json
import os
import socket
from typing import BinaryIO


# 消息上限为 16 MiB（按字节计算，包含末尾换行），防止无限制读取。
MAX_MESSAGE = 16 * 1024 * 1024


def receive(stream: BinaryIO) -> dict:
    """读取一条以换行符结束的 JSON 对象消息。"""
    # 多读一个字节，才能区分“刚好达到上限”和“实际超出上限”。
    line = stream.readline(MAX_MESSAGE + 1)

    # 空内容代表对端已关闭；没有换行表示消息未完整接收，不能当完整请求使用。
    if not line or len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
        raise ConnectionError(
            "Incomplete or oversized session message; "
            "execution may have happened. Do not retry automatically."
        )

    # JSON 也可以表示列表或数字，但本协议最外层只允许对象（Python 字典）。
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")

    return value


def send(stream: BinaryIO, value: dict) -> None:
    """发送一条 JSON 对象消息。"""
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")

    # ensure_ascii 把中文等字符转成 JSON 转义表示，不改变对方解析出来的值。
    # 最后的换行是消息边界，否则对方的 readline 不知道消息何时结束。
    data = (
        json.dumps(value, ensure_ascii=True).encode("utf-8") + b"\n"
    )
    if len(data) > MAX_MESSAGE:
        raise ValueError("Session message exceeds 16 MiB")

    stream.write(data)
    # makefile 返回的流可能有缓冲，flush 让消息立即交给 socket 发送。
    stream.flush()


def request(
    payload: dict,
    *,
    socket_path: str | None = None,
) -> dict:
    """建立连接，发送一次请求，接收一次响应。"""
    # 显式地址优先；通常由 runner 在容器中设置环境变量提供地址。
    address = socket_path or os.environ.get("AGENTICSIMULATOR_SOCKET")
    if not address:
        raise RuntimeError(
            "AGENTICSIMULATOR_SOCKET is not set; "
            "run inside an initialized workspace"
        )

    # 断连不代表请求没有执行，因此不能自动重试。
    # 例如服务端已启动实验、响应却丢失，再发一次可能额外消耗一份预算。
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(address)

        # rwb 是可读写的二进制流；两个 with 确保异常时也关闭流和连接。
        with connection.makefile("rwb") as stream:
            send(stream, payload)
            return receive(stream)
