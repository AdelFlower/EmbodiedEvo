"""源仿真校准与策略续训会话的命令行客户端，只依赖 Python 标准库。

在 Agent 工作区中调用，socket 地址由 runner 注入环境变量。
本文件只发送请求并打印宿主返回的 JSON；源候选保存、仿真执行、训练和反馈发布
由宿主负责。请求期间应等待返回，再修改 source/ 或 training/ 中的代码。
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
    sub.add_parser(
        "status", help="查询最新策略、对应的训练候选和剩余预算",
        description=(
            "latest_policy_id 是最新策略编号，latest_policy_candidate 是训练出它的源环境候选编号。"
            "首轮策略成功登记前，两者分别为 default（P0）和 null。查询状态不消耗实验预算。"
        ),
    )

    rollout = sub.add_parser(
        "rollout", help="用指定策略探测当前源环境、固定目标环境或两侧",
        description=(
            "rollout 只做探测，默认使用最新策略。source 使用本次请求时工作区里的 source/，"
            "并在 results.source.candidate_id 中返回这份物理代码的编号。"
            "target 使用固定目标环境，可用于判断最新策略是否达标。"
        ),
    )
    rollout.add_argument(
        "--environments", nargs="+", choices=("source", "target"), required=True,
        help="要执行的环境，可填写 source target；每侧最多出现一次",
    )
    rollout.add_argument(
        "--policy-id", default="latest",
        help="策略标识：latest 为最新策略（默认），default 为 P0，也可指定训练返回的策略编号",
    )
    rollout.add_argument("--episode-id", type=int, default=0, help="宿主配置中的回合编号，从 0 开始")
    rollout.add_argument(
        "--seed", type=int,
        help="指定随机种子以复现初始条件；省略时由宿主生成，两侧共用同一种子",
    )

    train = sub.add_parser(
        "train", help="在本轮源环境候选上从最新 checkpoint 续训一轮",
        description=(
            "使用 training/ 中的配方续训，可沿用原配方或按需修改。"
            "每次从最新 checkpoint 续训 1000 次迭代，每 100 次迭代保存 checkpoint。"
            "训练后可单独调用 rollout 观察新策略。"
        ),
    )
    train.add_argument(
        "--candidate-id",
        help=(
            "首次训练必须传入源侧 rollout 返回的候选编号。修改源物理参数后，传入本次观察得到的新编号。"
            "省略时沿用训练出最新策略的候选，可通过 status 的 latest_policy_candidate 查看。"
        ),
    )
    train.add_argument(
        "--num-envs", type=int,
        help="并行环境数量；省略时使用宿主默认值 4096",
    )

    finish = sub.add_parser("finish", help="结束会话，默认提交最新续训策略及对应的源环境候选")
    finish.add_argument(
        "--candidate-id",
        help=(
            "已有续训策略时，省略即提交最新策略及其训练候选，显式传入的编号须与该候选一致。"
            "首轮策略成功登记前，可指定已归档的源候选与 P0 一起提交，也可省略编号直接结束。"
        ),
    )

    args = parser.parse_args(argv)
    payload = {"op": args.command}
    if args.command == "rollout":
        payload.update(
            environments=args.environments,
            policy_id=args.policy_id,
            episode_id=args.episode_id,
        )
        if args.seed is not None:
            payload["seed"] = args.seed
    elif args.command == "train":
        if args.candidate_id is not None:
            payload["candidate_id"] = args.candidate_id
        if args.num_envs is not None:
            payload["num_envs"] = args.num_envs
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
