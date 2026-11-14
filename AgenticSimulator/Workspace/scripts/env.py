"""源仿真校准会话的命令行客户端，只依赖 Python 标准库。

在 Agent 工作区中调用，socket 地址由 runner 注入环境变量。
本文件只发送请求并打印宿主返回的 JSON；源候选保存、仿真执行和反馈发布
由宿主负责。请求期间应等待返回，再修改 source/ 中的代码。
"""

from __future__ import annotations

import argparse
import json
import os
import socket


MAX_MESSAGE = 16 * 1024 * 1024


def receive(stream):
    """读取以换行结束的一条 JSON 响应，不把残缺消息当作完成。"""
    line = stream.readline(MAX_MESSAGE + 1)
    if not line or len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
        raise ConnectionError(
            "Incomplete session response; execution may have happened. "
            "Do not retry automatically. Query status before choosing the next action."
        )
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def send(stream, value):
    """发送一条 JSON 请求，保持与 session 的消息格式一致。"""
    data = json.dumps(value, ensure_ascii=True).encode("utf-8") + b"\n"
    if len(data) > MAX_MESSAGE:
        raise ValueError("Session message exceeds 16 MiB")
    stream.write(data)
    stream.flush()


def request(payload, *, socket_path=None):
    """连接宿主 session，发送一次请求并等待响应，不自动重试。"""
    address = socket_path or os.environ.get("AGENTICSIMULATOR_SOCKET")
    if not address:
        raise RuntimeError(
            "AGENTICSIMULATOR_SOCKET is not set; run inside an initialized workspace"
        )
    # 断连不表示实验被撤销；重试可能再次执行并消耗预算。
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(address)
        with connection.makefile("rwb") as stream:
            send(stream, payload)
            return receive(stream)


def main(argv=None):
    """解析命令；服务返回 ok=true 时退出码为 0，否则为 1。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="查询会话状态和剩余预算，不运行实验")

    rollout = sub.add_parser("rollout", help="使用固定 policy 运行源环境、目标环境或两侧")
    rollout.add_argument(
        "--environments", nargs="+", choices=("source", "target"), required=True,
        help="要执行的环境，可填写 source target；每侧最多出现一次",
    )
    rollout.add_argument("--policy-id", default="default", help="宿主配置中的 policy 标识")
    rollout.add_argument("--episode-id", type=int, default=0, help="宿主配置中的回合编号，从 0 开始")

    finish = sub.add_parser("finish", help="提交最终候选并结束会话")
    finish.add_argument(
        "--candidate-id",
        help="源实验返回的候选编号；省略则不选择候选，直接结束会话",
    )

    args = parser.parse_args(argv)
    payload = {"op": args.command}
    if args.command == "rollout":
        payload.update(
            environments=args.environments,
            policy_id=args.policy_id,
            episode_id=args.episode_id,
        )
    elif args.command == "finish" and args.candidate_id is not None:
        payload["candidate_id"] = args.candidate_id

    try:
        result = request(payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok", False) else 1
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
