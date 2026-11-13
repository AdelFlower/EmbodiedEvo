"""与下游任务解耦的实验会话服务。

本模块负责：
- 接收和分发 Agent 请求；
- 管理请求编号、预算和会话状态；
- 调用通用实验执行接口；
- 保存请求记录，并向 session_runner 发送生命周期事件。

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


SESSION_PROTOCOL = "sim-calibration-v1"


class RequestError(ValueError):
    """可以直接返回给 Agent 的请求错误。"""


class ExperimentExecutor(Protocol):
    """宿主侧实验调度接口，由 worker 模块实现。

    execute() 的约定：

    1. 根据 request 的 op 执行 rollout 或 source_probe。
    2. 源快照、policy 选择、任务接入和证据导出由执行层负责。
    3. 每侧实验实际开始前，调用一次 on_start(environment)。
       该回调在宿主调度线程中调用。
    4. timeout_sec 覆盖整次请求，包括两侧实验及必要的准备工作。
    5. 返回或抛出异常前，结束本次请求的 worker，并保存已有证据。
    6. 预期的实验失败应返回结构化结果，不应丢弃另一侧的结果。
    7. public_dir 中只能放 Agent 允许读取的内容：
       源侧允许内部诊断；目标侧仅允许视频及必要的视频元信息。
    8. 返回值只包含公开结果，不包含目标私有配置或内部 traceback。

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
        ...

    def select_candidate(self, candidate_id: str) -> bool:
        """记录最终候选；候选不存在时返回 False，不启动新实验。"""
        ...

    def close(self) -> None:
        """停止并清理所属 worker。"""
        ...


class Session:
    def __init__(
        self,
        cfg: dict,
        run: Path,
        executor: ExperimentExecutor,
        event: Callable[[dict], None],
    ):
        self.run = Path(run).resolve()
        self.workspace = self.run / "workspace"
        self.executor = executor
        self.event = event

        settings = cfg["session"]

        self.execution_timeout = float(settings["execution_timeout_sec"])
        if (
            not math.isfinite(self.execution_timeout)
            or self.execution_timeout <= 0
        ):
            raise ValueError("execution_timeout_sec must be finite and positive")

        self.limits = {
            "source": settings.get("source_budget"),
            "target": settings.get("target_budget"),
        }
        for limit in self.limits.values():
            if limit is not None and (type(limit) is not int or limit < 0):
                raise ValueError("Experiment budgets must be nonnegative or null")

        self.used = {"source": 0, "target": 0}
        self.request_index = 0
        self.latest_request = None
        self.active_request = None
        self.final_candidate = None

        self.closed = False
        self.reason = "active"
        self.started_at = time.time()

        self.save()

    def status(self) -> dict:
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
            "experiments_used": dict(self.used),
            "experiments_remaining": remaining,
            "latest_request": self.latest_request,
            "active_request": self.active_request,
            "final_candidate": self.final_candidate,
        }

    def save(self) -> None:
        write_json(
            self.run / "session-result.json",
            {
                **self.status(),
                "started_at": self.started_at,
            },
        )

    @staticmethod
    def episode_id(request: dict) -> int:
        value = request.get("episode_id", 0)
        if type(value) is not int or value < 0:
            raise RequestError("episode_id must be a nonnegative integer")
        return value

    def handle(self, request: dict) -> dict:
        op = request.get("op")

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

        if (
            not isinstance(environments, list)
            or not environments
            or any(side not in ("source", "target") for side in environments)
            or len(environments) != len(set(environments))
        ):
            raise RequestError(
                "environments must contain source, target, or both without duplicates"
            )

        policy_id = request.get("policy_id", "default")
        if not isinstance(policy_id, str) or not policy_id:
            raise RequestError("policy_id must be a nonempty string")

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

        payload = {
            "op": "source_probe",
            "episode_id": self.episode_id(request),
            "script_path": script_path,
        }

        return self._execute(payload, ["source"])

    def _execute(self, request: dict, environments: list[str]) -> dict:
        """共用的请求登记、预算和结果记录逻辑。"""
        for side in environments:
            limit = self.limits[side]
            if limit is not None and self.used[side] >= limit:
                raise RequestError(f"{side} experiment budget exhausted")

        self.request_index += 1
        request_id = f"{self.request_index:06d}"

        private_dir = self.run / "requests" / request_id
        public_dir = self.workspace / "observations" / request_id
        private_dir.mkdir(parents=True, exist_ok=False)
        public_dir.mkdir(parents=True, exist_ok=False)

        self.latest_request = request_id
        self.active_request = request_id
        self.reason = "executing"

        write_json(private_dir / "request.json", request)
        write_json(
            public_dir / "result.json",
            {
                "request_id": request_id,
                "status": "running",
            },
        )

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

        started = set()

        def on_start(environment: str) -> None:
            # 执行层必须在每侧实验实际开始前调用一次。
            if environment not in environments or environment in started:
                raise RuntimeError("Invalid experiment start notification")

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

        self.event(
            {
                "event": "executing",
                "request_id": request_id,
                "timeout": self.execution_timeout,
            }
        )

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

            if (
                not isinstance(outcome, dict)
                or type(outcome.get("ok")) is not bool
                or not isinstance(outcome.get("results"), dict)
                or set(outcome["results"]) != set(environments)
            ):
                raise RuntimeError("Invalid experiment executor response")

            response.update(
                ok=outcome["ok"],
                results=outcome["results"],
            )

        except Exception:
            # 完整错误只保存到宿主，避免泄露目标配置和内部状态。
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
            write_json(
                private_dir / "interrupted.json",
                {
                    "request_id": request_id,
                    "status": "interrupted",
                    "started_environments": sorted(started),
                },
            )
            raise

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

        self.active_request = None
        self.reason = "active"
        self.save()

        # execute() 必须保证 worker 已结束、已有证据已保存。
        # 对于两侧请求，只在整个请求完成后发送 idle。
        self.event({"event": "idle", "request_id": request_id})
        return response

    def finish(self, candidate_id: str | None) -> dict:
        """记录 Agent 的最终选择，不评价其物理效果。"""
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
    """沿用外层的 Unix socket 和事件管道。"""
    run = Path(run).resolve()
    session = None
    exit_code = 0

    def event(value: dict) -> None:
        message = (json.dumps(value) + "\n").encode("utf-8")
        os.write(ready_fd, message)

    try:
        session = Session(cfg, run, executor, event)

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(socket_path))
            os.chmod(socket_path, 0o600)
            listener.listen(8)
            listener.settimeout(0.5)

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
                    connection.settimeout(10)

                    with connection.makefile("rwb") as stream:
                        try:
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
                                response = {
                                    "ok": False,
                                    "error": str(exc),
                                }

                        send(stream, response)

    except KeyboardInterrupt:
        exit_code = 130
        # finish 已落盘后，runner 可以结束服务；保留已确认的终态。
        if session is not None and not session.closed:
            session.closed = True
            session.reason = "interrupted"

    except Exception:
        exit_code = 1
        (run / "session-error.txt").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        if session is not None:
            session.closed = True
            session.reason = "service_error"

    finally:
        try:
            executor.close()
        except Exception:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready-fd", type=int, required=True)
    args = parser.parse_args(argv)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)

    try:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))

        # 这是宿主调度器工厂，不在这里加载具体任务或执行 Agent 代码。
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
        os.close(args.ready_fd)


if __name__ == "__main__":
    raise SystemExit(main())
