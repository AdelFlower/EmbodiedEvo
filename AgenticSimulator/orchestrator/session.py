"""与下游任务解耦的实验会话服务。

本模块负责：
- 接收和分发 Agent 请求；
- 管理请求编号、预算和会话状态；
- 调用通用实验执行接口；
- 保存请求记录，并向 session_runner 发送生命周期事件。

调用链为：main() 创建执行器 → serve() 接收 socket 消息 →
Session.handle() 校验并分发 → executor.execute() 执行具体实验。
会话层只管理通用状态；具体任务、仿真进程和证据导出由执行层负责。
执行层按照宿主指定的信息开放规则生成公开文件和公开结果；本模块负责
转交这些结果，不解释具体实验的传感器字段，也不再次过滤结果内容。

一次请求可以涉及 source、target 两侧，因此“请求次数”和“实验次数”
分别记录。请求被接收时分配编号，每侧实际开始时才扣减该侧实验预算。

阅读时可以先记住几个词：
- source 是 Agent 可修改的源仿真器，target 是用于对照的目标环境。
- rollout 是按指定 policy（控制策略）运行实验，source_probe 是在源侧运行脚本。
- executor 是实际运行实验的对象，本文件通过约定的接口调用它。
- 回调是“传给另一段代码、由它在约定时机调用的函数”，例如 on_start。
- “落盘”就是把当前信息写进文件，避免信息只留在进程内存里。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import socket
import time
import traceback
from pathlib import Path
from typing import Callable, Protocol

from .files import append_jsonl, write_json
from .session_client import receive, send


# 此标识属于 AgenticSimulator 的校准会话协议，随 status 返回给客户端。
SESSION_PROTOCOL = "sim-calibration-v1"


class RequestError(ValueError):
    """可以直接返回给 Agent 的请求错误。"""


class ExperimentExecutor(Protocol):
    """宿主侧实验调度接口，约定由 worker 模块提供具体实现。

    Protocol 可以理解为一张接口清单：提供 execute/select_candidate/close
    这些方法的对象即可接入。此处的省略号不是实验实现，也不会自动运行仿真。

    execute() 的约定：

    1. 根据 request 的 op 执行 rollout 或 source_probe。
    2. 源快照、policy 选择、任务接入和证据导出由执行层负责。
    3. 每侧实验实际开始前，调用一次 on_start(environment)。
       该回调在宿主调度线程中调用。
    4. timeout_sec 覆盖整次请求，包括两侧实验及必要的准备工作。
    5. 返回或抛出异常前，结束本次请求的 worker，并保存已有证据。
    6. 源侧代码报错、目标 rollout 失败等预期的实验失败应返回结构化结果，
       保留另一侧已有结果及失败证据。源侧可返回完整代码执行诊断，目标侧
       的错误信息按其开放规则处理。执行器自身异常可抛给会话层，由会话层
       私下保存 traceback，并向 Agent 返回通用执行错误。
    7. 源侧允许公开源仿真的完整实验信息和代码执行诊断；目标侧按宿主配置
       指定的范围公开信息，不在会话层限定具体传感器或字段。Agent 请求
       不能扩大目标侧的信息权限。源侧的完整信息不包括宿主凭据、目标私有
       数据等与源仿真实验无关的信息。
    8. adapter 原始输出先写入 private_dir；执行器再按各侧开放规则生成
       public_dir 中的文件及返回值。字段、错误说明和文件引用都须符合该
       侧规则，不能只过滤返回字典而把完整的目标轨迹放进公开目录。

    返回结构：
        {
            "ok": bool,       # 执行状态，不是任务成功或物理相似度
            "results": {
                "source": {...},
                "target": {...},
            },
        }

    results 只包含本次请求涉及的环境。
    source_probe 的 results 只有 source。
    results 必须已经是可交给 Agent 的公开结果；会话层将直接转交，不能
    将 adapter 的原始返回值未经公开处理直接作为执行器返回值。
    """

    def execute(
        self,
        request: dict,
        *,
        request_id: str,
        private_dir: Path,
        public_dir: Path,
        timeout_sec: float,
        on_start: Callable[[str], None],
    ) -> dict:
        """执行请求并返回按源/目标信息开放规则处理后的公开结果。

        private_dir 保存完整原始结果、内部日志和失败证据，仅宿主可见。
        public_dir 位于 Agent 工作区，只保存按开放规则生成的公开文件。
        信息开放规则由宿主提供，不由 request 中的 Agent 输入决定。
        """
        # Protocol 描述结构化接口，不提供实现；实际对象由工厂创建。
        ...

    def select_candidate(self, candidate_id: str) -> bool:
        """记录最终候选；候选不存在时返回 False，不启动新实验。"""
        ...

    def close(self) -> None:
        """停止并清理所属 worker。"""
        ...


class Session:
    """保存单次服务的内存状态，并协调请求、预算和结果落盘。

    serve() 串行调用本对象；执行器的 on_start 回调也在调度线程调用，
    因此这里没有为状态字段加锁。构造时从零初始化，不读取旧状态续跑；
    session-result.json 是状态快照，不是本类自动恢复会话的输入。
    """

    def __init__(
        self,
        cfg: dict,
        run: Path,
        executor: ExperimentExecutor,
        event: Callable[[dict], None],
    ):
        """接收配置、宿主运行目录、执行器和用于通知 runner 的事件函数。"""
        # run 是宿主运行目录；只有其 workspace 子目录对 Agent 开放。
        self.run = Path(run).resolve()
        self.workspace = self.run / "workspace"
        self.executor = executor
        # 生命周期事件通过外层提供的回调交给 runner，不走请求响应 socket。
        self.event = event

        settings = cfg["session"]

        # 超时约束整次请求，不能让双侧请求各自获得一份完整超时时间。
        self.execution_timeout = float(settings["execution_timeout_sec"])
        if (
            not math.isfinite(self.execution_timeout)
            or self.execution_timeout <= 0
        ):
            raise ValueError("execution_timeout_sec must be finite and positive")

        # 两侧分别计费；缺省值和显式 null 都表示该侧没有实验次数上限。
        self.limits = {
            "source": settings.get("source_budget"),
            "target": settings.get("target_budget"),
        }
        for limit in self.limits.values():
            # 使用 type(...) is int，避免将 Python 的 bool 当作整数预算。
            if limit is not None and (type(limit) is not int or limit < 0):
                raise ValueError("Experiment budgets must be nonnegative or null")

        self.used = {"source": 0, "target": 0}
        # request_index 统计已登记的执行请求，status/finish 不占用编号。
        self.request_index = 0
        # latest 保留最近一次请求；active 仅在该请求尚未正常返回时有值。
        self.latest_request = None
        self.active_request = None
        self.final_candidate = None

        # closed 表示会话是否终止，reason 表示当前阶段或终止原因。
        self.closed = False
        self.reason = "active"
        self.started_at = time.time()

        # 启动即落盘，使还未收到任何请求的会话也有状态记录。
        self.save()

    def status(self) -> dict:
        """返回可序列化的状态快照，不执行实验，也不消耗预算。"""
        # None 继续表示不设上限，不转换成一个虚构的剩余次数。
        remaining = {
            side: (
                None
                if self.limits[side] is None
                else self.limits[side] - self.used[side]
            )
            for side in self.used
        }

        return {
            "ok": True,
            "protocol": SESSION_PROTOCOL,
            "closed": self.closed,
            "reason": self.reason,
            "requests_used": self.request_index,
            # 复制计数表，避免调用者通过返回值修改内部预算状态。
            "experiments_used": dict(self.used),
            "experiments_remaining": remaining,
            "latest_request": self.latest_request,
            "active_request": self.active_request,
            "final_candidate": self.final_candidate,
        }

    def save(self) -> None:
        """将当前摘要写入宿主目录；逐次请求的证据另行保存。"""
        write_json(
            self.run / "session-result.json",
            {
                **self.status(),
                "started_at": self.started_at,
            },
        )

    @staticmethod
    def episode_id(request: dict) -> int:
        """统一校验回合编号；未指定时使用 0，拒绝布尔值和负数。"""
        value = request.get("episode_id", 0)
        if type(value) is not int or value < 0:
            raise RequestError("episode_id must be a nonnegative integer")
        return value

    def handle(self, request: dict) -> dict:
        """处理业务操作；消息格式解析和错误响应封装由 serve() 完成。"""
        op = request.get("op")

        # 即使已关闭，直接调用本方法仍可查状态；其余操作均拒绝。
        if op == "status":
            return self.status()

        if self.closed:
            raise RequestError("Session is already closed")

        if op == "rollout":
            return self.handle_rollout(request)

        if op == "source_probe":
            return self.handle_source_probe(request)

        if op == "finish":
            return self.finish(request.get("candidate_id"))

        raise RequestError("Unknown session operation")

    def handle_rollout(self, request: dict) -> dict:
        """整理通用请求，不实现 rollout 循环。"""
        environments = request.get("environments")

        # 环境列表必须非空且不重复；保留原顺序交给执行器安排实验。
        if (
            not isinstance(environments, list)
            or not environments
            or any(side not in ("source", "target") for side in environments)
            or len(environments) != len(set(environments))
        ):
            raise RequestError(
                "environments must contain source, target, or both without duplicates"
            )

        # 会话层只校验标识格式，具体 policy 是否存在由执行层判断。
        policy_id = request.get("policy_id", "default")
        if not isinstance(policy_id, str) or not policy_id:
            raise RequestError("policy_id must be a nonempty string")

        # 重新构造 payload，只将本操作约定的字段传给执行器。
        # 例如 environments=["source", "target"] 表示一个请求分别跑两侧，
        # 请求编号只增加一次，两侧开始时各扣一次对应预算。
        payload = {
            "op": "rollout",
            "environments": list(environments),
            "policy_id": policy_id,
            "episode_id": self.episode_id(request),
        }

        return self._execute(payload, environments)

    def handle_source_probe(self, request: dict) -> dict:
        """接收工作区内的脚本路径，由执行层读取并运行。"""
        script_path = request.get("script_path")

        if not isinstance(script_path, str) or not script_path.strip():
            raise RequestError("script_path must be a nonempty string")

        # 这里只检查非空；路径解析、访问边界和脚本读取由执行层负责。
        payload = {
            "op": "source_probe",
            "episode_id": self.episode_id(request),
            "script_path": script_path,
        }

        # probe 只能在源环境运行，也只检查并扣减 source 预算。
        return self._execute(payload, ["source"])

    def _execute(self, request: dict, environments: list[str]) -> dict:
        """共用的请求登记、预算和结果记录逻辑。"""
        # 先检查所有涉及的环境；任一侧预算耗尽就整体拒绝，尚不分配编号。
        for side in environments:
            limit = self.limits[side]
            if limit is not None and self.used[side] >= limit:
                raise RequestError(f"{side} experiment budget exhausted")

        # 编号至少六位，方便目录排序；一次双侧实验仍然只使用一个编号。
        self.request_index += 1
        request_id = f"{self.request_index:06d}"

        # 私有目录保存原始请求和内部错误，公开目录只保存可交给 Agent 的内容。
        # exist_ok=False 可防止覆盖已有编号的证据；此处没有自动续跑逻辑。
        private_dir = self.run / "requests" / request_id
        public_dir = self.workspace / "observations" / request_id
        private_dir.mkdir(parents=True, exist_ok=False)
        public_dir.mkdir(parents=True, exist_ok=False)

        self.latest_request = request_id
        self.active_request = request_id
        self.reason = "executing"

        # 先保存受理凭据和 running 占位结果，再启动执行器。
        # 若进程中断，磁盘上仍能区分“已受理”与“已正常返回”。
        write_json(private_dir / "request.json", request)
        write_json(
            public_dir / "result.json",
            {
                "request_id": request_id,
                "status": "running",
            },
        )

        # JSONL 记录时间线，session-result.json 则保存最近的整体状态。
        append_jsonl(
            self.run / "requests.jsonl",
            {
                "event": "accepted",
                "request_id": request_id,
                "op": request["op"],
                "at": time.time(),
            },
        )
        self.save()

        # 本次请求已启动的环境集合，同时用于去重和中断证据记录。
        started = set()

        def on_start(environment: str) -> None:
            """由执行器报告某一侧即将开始；在这里计数并保存启动记录。"""
            # 执行层必须在每侧实验实际开始前调用一次。
            if environment not in environments or environment in started:
                raise RuntimeError("Invalid experiment start notification")

            # 仅在实际启动时计费：受理请求本身不扣实验预算。
            # 启动后的失败同样保留计数，本层不会退还预算或自动重试。
            started.add(environment)
            self.used[environment] += 1

            append_jsonl(
                self.run / "requests.jsonl",
                {
                    "event": "experiment_started",
                    "request_id": request_id,
                    "environment": environment,
                    "at": time.time(),
                },
            )
            self.save()

        # 提前通知 runner 进入执行阶段，并提供整次请求的超时配置。
        self.event(
            {
                "event": "executing",
                "request_id": request_id,
                "timeout": self.execution_timeout,
            }
        )

        # 返回工作区相对路径，供 Agent 定位证据，不暴露宿主绝对路径。
        response = {
            "request_id": request_id,
            "observation_dir": str(public_dir.relative_to(self.workspace)),
        }

        try:
            outcome = self.executor.execute(
                request,
                request_id=request_id,
                private_dir=private_dir / "execution",
                public_dir=public_dir,
                timeout_sec=self.execution_timeout,
                on_start=on_start,
            )

            # 要求每个请求环境都有结果，包括结构化的失败结果。
            # 这里只验证外层契约；执行器负责按各侧开放规则处理内部字段和文件。
            # 下方直接转交 results，不再次过滤，也不能直接接入 adapter 原始结果。
            if (
                not isinstance(outcome, dict)
                or type(outcome.get("ok")) is not bool
                or not isinstance(outcome.get("results"), dict)
                or set(outcome["results"]) != set(environments)
            ):
                raise RuntimeError("Invalid experiment executor response")

            response.update(
                # ok 描述实验执行状态；不能据此推断任务达标或两侧物理一致。
                ok=outcome["ok"],
                results=outcome["results"],
            )

        except Exception:
            # 此处处理执行器自身异常或返回值违反接口约定的情况。
            # 预期的实验失败应由执行器返回结构化结果，源侧代码诊断随结果返回。
            # 未处理异常的完整 traceback 只保存到宿主，避免泄露目标配置和内部状态。
            # 已有部分证据由 executor 保留，不自动重跑。
            (private_dir / "executor-error.txt").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            response.update(
                ok=False,
                error="experiment_execution_error",
            )

        except BaseException:
            # 中断也留下记录，不伪造正常完成的结果。
            # KeyboardInterrupt/SystemExit 等继续向外传播，不进入下方
            # returned/idle 流程；active_request 也保留为被中断的请求编号。
            write_json(
                private_dir / "interrupted.json",
                {
                    "request_id": request_id,
                    "status": "interrupted",
                    "started_environments": sorted(started),
                },
            )
            raise

        # 普通成功和已捕获的执行异常都形成最终响应，并覆盖 running 占位。
        # 私有 traceback 不包含在这份公开响应中。
        write_json(public_dir / "result.json", response)
        write_json(private_dir / "result.json", response)

        append_jsonl(
            self.run / "requests.jsonl",
            {
                "event": "returned",
                "request_id": request_id,
                "ok": response["ok"],
                "started_environments": sorted(started),
                "at": time.time(),
            },
        )

        # 本次请求已返回不代表整个会话结束，Agent 仍可继续提交或 finish。
        self.active_request = None
        self.reason = "active"
        self.save()

        # execute() 必须保证 worker 已结束、已有证据已保存。
        # 对于两侧请求，只在整个请求完成后发送 idle。
        self.event({"event": "idle", "request_id": request_id})
        return response

    def finish(self, candidate_id: str | None) -> dict:
        """记录 Agent 的最终选择，不评价其物理效果。"""
        # 先交给执行器确认并记录候选；选择失败时不关闭当前会话。
        if candidate_id is not None:
            if not isinstance(candidate_id, str) or not candidate_id:
                raise RequestError("candidate_id must be a nonempty string")

            if not self.executor.select_candidate(candidate_id):
                raise RequestError("Unknown source candidate")

        # 允许没有最终候选时结束，例如全部实验均失败。
        self.final_candidate = candidate_id
        self.closed = True
        self.reason = "agent_finished"
        self.save()
        return self.status()


def serve(
    cfg: dict,
    run: Path,
    socket_path: Path,
    ready_fd: int,
    executor: ExperimentExecutor,
) -> int:
    """串行处理 Unix socket 请求，通过独立文件描述符报告生命周期。

    每个连接只读取一条请求并发送一条响应。实验执行期间不会接收下一条
    连接；runner 通过事件管道接收 ready/executing/idle/finished 通知。
    返回进程退出码，所有退出路径都尝试清理执行器并保存最终状态。
    """
    run = Path(run).resolve()
    session = None
    exit_code = 0

    def event(value: dict) -> None:
        """把一个生命周期通知编码成一行 JSON，写入 runner 的事件管道。"""
        # ready_fd 是外层传入的可写描述符；一行 JSON 对应一个事件。
        message = (json.dumps(value) + "\n").encode("utf-8")
        os.write(ready_fd, message)

    try:
        session = Session(cfg, run, executor, event)

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(socket_path))
            # 限制 socket 文件为所属用户可读写；这里不实现额外认证协议。
            os.chmod(socket_path, 0o600)
            # 8 是待接收连接的队列长度，不表示并行运行八个实验。
            listener.listen(8)
            # 定期结束 accept 等待；无连接时继续循环，不消耗实验预算。
            listener.settimeout(0.5)

            # 仅在状态初始化、绑定及监听均成功后宣布服务可用。
            event({"event": "ready"})

            while not session.closed:
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue

                # 断连不意味着实验没有执行，不能自动重复请求。
                with (
                    connection,
                    contextlib.suppress(
                        BrokenPipeError,
                        ConnectionResetError,
                        TimeoutError,
                    ),
                ):
                    # 限制连接读写等待；实验总时限另由 execution_timeout 管理。
                    connection.settimeout(10)

                    with connection.makefile("rwb") as stream:
                        try:
                            # receive 负责换行 JSON 对象的解析及消息大小限制。
                            request = receive(stream)
                        except (ConnectionError, OSError, ValueError):
                            response = {
                                "ok": False,
                                "error": "invalid_request_message",
                            }
                        else:
                            try:
                                response = session.handle(request)
                            except RequestError as exc:
                                # 可预期的参数或预算错误仅拒绝本条请求。
                                response = {
                                    "ok": False,
                                    "error": str(exc),
                                }

                        # 实验可能已完成而响应发送失败；上方只抑制传输异常，
                        # 不因此撤销计数，也不重新调用 session.handle()。
                        send(stream, response)

    except KeyboardInterrupt:
        # SIGTERM 在 main() 中也转换为该异常，统一走清理和状态保存路径。
        exit_code = 130
        # finish 已落盘后，runner 可以结束服务；保留已确认的终态。
        if session is not None and not session.closed:
            session.closed = True
            session.reason = "interrupted"

    except Exception:
        # 未被识别为请求错误的异常视为服务错误；内部堆栈只写入宿主目录。
        exit_code = 1
        (run / "session-error.txt").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        if session is not None:
            session.closed = True
            session.reason = "service_error"

    finally:
        # 即使 Session 初始化失败，也清理已由 main() 创建的执行器。
        try:
            executor.close()
        except Exception:
            # 清理失败也使进程失败，并覆盖终止原因，保留单独的错误证据。
            exit_code = 1
            (run / "session-cleanup-error.txt").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            if session is not None:
                session.closed = True
                session.reason = "cleanup_error"

        if session is not None:
            session.save()

    # 结束响应已尝试发送，状态已保存，所属 worker 已清理。
    if (
        exit_code == 0
        and session is not None
        and session.reason == "agent_finished"
    ):
        event({"event": "finished"})

    return exit_code


def main(argv=None) -> int:
    """解析宿主参数、创建执行器并启动服务，最后关闭事件管道写端。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready-fd", type=int, required=True)
    args = parser.parse_args(argv)

    def interrupt(signum, frame):
        # 转成可捕获异常，使服务有机会停止 worker、记录中断状态。
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)

    try:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))

        # 工厂函数负责按配置创建具体执行器；这条 import 表示预期接入位置，
        # 不代表上面的 Protocol 已经实现了实验。这里不直接执行 Agent 代码。
        from AgenticSimulator.simulators.worker import create_executor

        executor = create_executor(
            cfg=cfg,
            run=args.run_dir.resolve(),
        )

        return serve(
            cfg=cfg,
            run=args.run_dir,
            socket_path=args.socket,
            ready_fd=args.ready_fd,
            executor=executor,
        )
    finally:
        # 描述符由调用方传入，但当前进程负责关闭自己的这一端。
        os.close(args.ready_fd)


if __name__ == "__main__":
    raise SystemExit(main())
