
"""创建工作区，管理 Agent 与实验服务的生命周期。

Agent 负责观察、推理和修改。
session.py 负责处理实验请求，并管理源/目标 rollout。

"""

from __future__ import annotations

import json
import os
import select
import shutil
import signal
import subprocess
import tempfile
import time
import traceback
from contextlib import ExitStack, suppress
from datetime import datetime
from pathlib import Path

from . import launch_agent
from .files import write_json


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOT = PACKAGE_ROOT.parent

CONTAINER_SOCKET_DIR = "/run/agenticsimulator"
CONTAINER_SOCKET = f"{CONTAINER_SOCKET_DIR}/session.sock"


def create_workspace(cfg: dict, run: Path) -> Path:
    """从模板创建本次运行的工作区，再放入源实现。"""
    template = Path(cfg["workspace_template"]).expanduser().resolve()
    source = Path(cfg["source_root"]).expanduser().resolve()
    workspace = run / "workspace"

    # 避免把输出目录递归复制进自身。
    if run.is_relative_to(template) or run.is_relative_to(source):
        raise ValueError("run-dir cannot be inside the template or source directory")

    shutil.copytree(template, workspace)
    shutil.copytree(
        source,
        workspace / "source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
    )

    for name in ("observations", "scratch"):
        (workspace / name).mkdir(exist_ok=True)

    return workspace


def agent_command(
    cfg: dict,
    workspace: Path,
    home: Path,
    socket_dir: Path,
) -> tuple[str, list[str]]:
    """复用 launch_agent，并补上本次会话的挂载。"""
    settings = cfg["agent"]
    preset = settings["preset"]
    image = settings["image"]

    name = f"agenticsimulator-{os.getpid()}-{time.time_ns()}"

    inner = launch_agent.agent_argv(
        preset,
        (
            "Read instruction.md and tools.md. "
            "Use scripts/env.py to interact with the experiment service. "
            "Work autonomously toward the task described in instruction.md."
        ),
        model=settings["model"],
        effort=settings.get("reasoning_effort"),
    )
    inner = launch_agent.adapt_for_docker(inner, preset=preset)

    argv = launch_agent.docker_argv(
        workspace,
        home,
        inner,
        image=image,
        network=settings.get("network", launch_agent.AGENT_NETWORK),
        cpus=settings.get("cpus", 4),
        memory_mb=settings.get("memory_mb", 8192),
        container_name=name,
    )

    mounts = [
        "--env",
        f"AGENTICSIMULATOR_SOCKET={CONTAINER_SOCKET}",
        "--mount",
        (
            f"type=bind,src={socket_dir},"
            f"dst={CONTAINER_SOCKET_DIR},readonly"
        ),
    ]

    # source/ 和 scratch/ 可写。
    # 指令、客户端和原始反馈由宿主管理。
    for component in ("instruction.md", "tools.md", "scripts", "observations"):
        mounts += [
            "--mount",
            (
                f"type=bind,src={workspace / component},"
                f"dst=/workspace/{component},readonly"
            ),
        ]

    # docker_argv 的结尾固定为 [image, *inner]。
    image_index = len(argv) - len(inner) - 1
    argv[image_index:image_index] = mounts

    return name, argv


def stop_process(process: subprocess.Popen | None) -> None:
    """终止独立进程组，包括继承该进程组的 worker。"""
    if process is None:
        return

    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

    # 服务主进程可能已经退出，但同组 worker 仍然存在。
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)

    process.wait(timeout=5)


def _session_finished(run: Path) -> bool:
    """以宿主保存的终态为准，不依赖结束通知的到达顺序。"""
    path = run / "session-result.json"
    if not path.is_file():
        return False
    state = json.loads(path.read_text(encoding="utf-8"))
    return (
        state.get("closed") is True
        and state.get("reason") == "agent_finished"
        and state.get("active_request") is None
    )


def supervise(
    session: subprocess.Popen,
    read_fd: int,
    start_agent,
    *,
    run: Path,
    startup_timeout: float,
    timeout: float | None,
    event_log,
) -> tuple[str, subprocess.Popen | None]:
    """沿用外层事件管道，等待就绪并监控正在执行的实验。"""
    agent = None
    buffer = bytearray()
    pipe_open = True
    ready = False

    startup_deadline = time.monotonic() + startup_timeout
    agent_deadline = None
    execution_deadline = None

    while True:
        if pipe_open:
            readable, _, _ = select.select([read_fd], [], [], 0.2)

            if readable:
                chunk = os.read(read_fd, 65536)

                if not chunk:
                    pipe_open = False
                else:
                    buffer.extend(chunk)

                while b"\n" in buffer:
                    line, _, rest = buffer.partition(b"\n")
                    buffer = bytearray(rest)
                    event = json.loads(line)

                    event_log.write(
                        json.dumps(event, ensure_ascii=False) + "\n"
                    )
                    event_log.flush()

                    kind = event["event"]

                    if kind == "ready" and not ready:
                        ready = True
                        agent = start_agent()
                        if timeout is not None:
                            agent_deadline = time.monotonic() + timeout

                    elif kind == "executing":
                        # 留出少量收尾时间，服务端仍应自行实施执行时限。
                        execution_deadline = (
                            time.monotonic() + float(event["timeout"]) + 10
                        )

                    elif kind == "idle":
                        execution_deadline = None

                    elif kind == "finished":
                        execution_deadline = None
        else:
            time.sleep(0.2)

        now = time.monotonic()

        if not ready and now >= startup_deadline:
            return "session_startup_timeout", agent

        session_rc = session.poll()
        if session_rc is not None:
            if session_rc != 0 or not _session_finished(run):
                return "session_exited", agent

        if not pipe_open and not _session_finished(run):
            return "session_event_pipe_closed", agent

        if execution_deadline is not None and now >= execution_deadline:
            return "execution_timeout", agent

        if agent is not None and agent.poll() is not None:
            # Agent 退出并不代表已经提交的实验被取消。
            if execution_deadline is None:
                reason = (
                    "session_finished"
                    if agent.returncode == 0 and _session_finished(run)
                    else "agent_exited"
                )
                return reason, agent

        if agent_deadline is not None and now >= agent_deadline:
            return "session_time_limit", agent


def run_session(cfg: dict, run_dir: str | Path) -> dict:
    """执行一次新的校准会话；保留失败，不自动重跑。"""
    run = Path(run_dir).expanduser().resolve()

    prefix = run.name[:12]
    if len(prefix) != 12 or not prefix.isdigit():
        raise ValueError("run-dir must start with YYYYMMDDHHMM")
    datetime.strptime(prefix, "%Y%m%d%H%M")

    # 已有运行目录不覆盖。
    run.mkdir(parents=True, exist_ok=False)

    agent = None
    session = None
    container = None
    read_fd = None
    write_fd = None

    result = {
        "started_at": time.time(),
        "reason": "starting",
        "agent_returncode": None,
        "session_returncode": None,
        "cleanup_errors": [],
    }

    # 临时目录存放 socket 和凭据，避免进入实验归档。
    with (
        tempfile.TemporaryDirectory(prefix="agenticsimulator-") as temporary,
        ExitStack() as stack,
    ):
        temporary = Path(temporary)
        socket_dir = temporary / "ipc"
        socket_dir.mkdir()

        def open_log(name):
            return stack.enter_context(
                (run / name).open("w", encoding="utf-8")
            )

        agent_stdout = open_log("agent.stdout.log")
        agent_stderr = open_log("agent.stderr.log")
        session_stdout = open_log("session.stdout.log")
        session_stderr = open_log("session.stderr.log")
        cleanup_log = open_log("cleanup.log")
        event_log = open_log("events.jsonl")

        try:
            workspace = create_workspace(cfg, run)

            # 配置应只保存凭据引用，不包含 API key 等秘密值。
            # 其中的目标私有配置不会挂载到 Agent 工作区。
            config_path = run / "config.json"
            write_json(config_path, cfg)

            agent_settings = cfg["agent"]
            session_settings = cfg["session"]
            network = agent_settings.get(
                "network", launch_agent.AGENT_NETWORK
            )

            launch_agent.require_image(agent_settings["image"])
            if network != "none":
                launch_agent.ensure_network(network)

            home = launch_agent.stage_home(temporary / "agent-home")
            container, argv = agent_command(
                cfg, workspace, home, socket_dir
            )

            read_fd, write_fd = os.pipe()

            command = [
                session_settings["python"],
                "-m", "AgenticSimulator.orchestrator.session",
                "--config", str(config_path),
                "--run-dir", str(run),
                "--socket", str(socket_dir / "session.sock"),
                "--ready-fd", str(write_fd),
            ]

            session = subprocess.Popen(
                command,
                cwd=IMPORT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=session_stdout,
                stderr=session_stderr,
                pass_fds=(write_fd,),
                start_new_session=True,
            )

            os.close(write_fd)
            write_fd = None

            def start_agent():
                nonlocal agent
                agent = subprocess.Popen(
                    argv,
                    cwd=IMPORT_ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=agent_stdout,
                    stderr=agent_stderr,
                    start_new_session=True,
                )
                return agent

            reason, agent = supervise(
                session,
                read_fd,
                start_agent,
                run=run,
                startup_timeout=session_settings.get(
                    "startup_timeout_sec", 120
                ),
                timeout=session_settings.get("timeout_sec"),
                event_log=event_log,
            )
            result["reason"] = reason

        except KeyboardInterrupt:
            result["reason"] = "interrupted"

        except Exception as exc:
            result["reason"] = "runner_error"
            result["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            (run / "runner-error.txt").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )

        finally:
            # 先停止 Docker 客户端，再清理它启动的容器。
            for process in (agent,):
                try:
                    stop_process(process)
                except Exception as exc:
                    result["cleanup_errors"].append(str(exc))

            if container is not None:
                try:
                    subprocess.run(
                        ["docker", "rm", "-f", container],
                        stdin=subprocess.DEVNULL,
                        stdout=cleanup_log,
                        stderr=subprocess.STDOUT,
                        timeout=15,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    result["cleanup_errors"].append(str(exc))

            try:
                stop_process(session)
            except Exception as exc:
                result["cleanup_errors"].append(str(exc))

            for fd in (read_fd, write_fd):
                if fd is not None:
                    os.close(fd)

            result["agent_returncode"] = (
                agent.poll() if agent is not None else None
            )
            result["session_returncode"] = (
                session.poll() if session is not None else None
            )
            # runner 的正常收尾可以终止服务进程，退出码仅保留作诊断。
            # 服务若记录了实际错误，仍不能将本次会话报告为正常完成。
            if result["reason"] == "session_finished" and not _session_finished(run):
                result["reason"] = "session_exited"
            result["finished_at"] = time.time()

            write_json(run / "runtime-result.json", result)

    return result
