
"""创建工作区，管理 Agent 与实验服务的生命周期。

Agent 负责观察、推理和修改。
session.py 负责处理实验请求，并委托执行器运行源/目标 rollout。

runner 可以理解为“整个运行的管理员”，大致按这个顺序工作：
1. 创建工作区、日志和临时通信目录。
2. 启动宿主实验服务，等它报告 ready（就绪）后，再启动 Agent 容器。
3. 监控进程和时限；结束时清理容器、服务及文件描述符，并保存摘要。

两条通信通道不要混淆：Agent 通过 socket 向服务请求实验；服务通过 pipe
（父子进程间的管道）向 runner 报告生命周期，runner 不转发实验请求。

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


# PACKAGE_ROOT 是 AgenticSimulator/；IMPORT_ROOT 是它的父目录。
# 子进程从 IMPORT_ROOT 启动，才能用 python -m AgenticSimulator... 导入模块。
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOT = PACKAGE_ROOT.parent

# 服务在宿主临时目录创建 socket，Agent 通过下面的容器内路径访问同一个 socket。
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

    # 各次运行都从独立副本开始，Agent 修改 workspace/source 不会改到原始源码。
    # copytree 默认拒绝覆盖已有目标目录，避免把两次运行内容混在一起。
    shutil.copytree(template, workspace)
    shutil.copytree(
        source,
        workspace / "source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
    )

    # observations 放服务生成的反馈；scratch 供 Agent 写临时程序或中间文件。
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

    # 名字既交给 docker run，也返回给清理逻辑，用于精确删除本次容器。
    name = f"agenticsimulator-{os.getpid()}-{time.time_ns()}"

    # inner 是容器内部的 CLI 命令；它通过工作区文档了解任务和可用工具。
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

    # argv 则是宿主执行的完整 docker run 命令，包装了上述 inner。
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

    # 映射 socket 所在目录并告知客户端地址。readonly 禁止改目录内容，
    # 不会禁止客户端连接目录中现有的 Unix socket。
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
    # 在整体可写的 workspace 挂载上，再覆盖这些路径为只读。
    # 宿主仍可更新 observations，Agent 能读取更新后的内容。
    for component in ("instruction.md", "tools.md", "scripts", "observations"):
        mounts += [
            "--mount",
            (
                f"type=bind,src={workspace / component},"
                f"dst=/workspace/{component},readonly"
            ),
        ]

    # docker_argv 的结尾固定为 [image, *inner]。
    # 挂载是 Docker 选项，必须插到 image 之前，不能落到容器内命令中。
    image_index = len(argv) - len(inner) - 1
    argv[image_index:image_index] = mounts

    return name, argv


def stop_process(process: subprocess.Popen | None) -> None:
    """终止独立进程组，包括继承该进程组的 worker。"""
    if process is None:
        return

    # Popen 使用 start_new_session=True，使进程号同时成为新进程组的编号。
    # 向进程组发送信号可覆盖继承该组的子进程；已退出则忽略“进程不存在”。
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)

    # 先给 SIGTERM 和五秒收尾机会；仍存活的同组进程随后收到 SIGKILL。
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
    # 收到 finished 事件或看到进程退出都不够：必须有已关闭、Agent 主动结束、
    # 且没有执行中请求的持久化终态，才确认实验服务已完成协议流程。
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
    """持续监控服务和 Agent，返回结束原因及已启动的 Agent 进程。

    start_agent 是启动回调，等到 ready 才调用，以免 Agent 抢先连接未就绪服务。
    本函数判断何时停止监控，实际终止进程由 run_session 的 finally 统一负责。
    """
    agent = None
    # pipe 是字节流：一次读取可能收到半条事件或多条事件，需要缓冲并按换行拆分。
    buffer = bytearray()
    pipe_open = True
    ready = False

    # monotonic 是只向前走的计时器，不受系统时钟校准影响，适合计算超时。
    # 三个截止时间分别限制服务启动、Agent 总运行和当前单次实验请求。
    startup_deadline = time.monotonic() + startup_timeout
    agent_deadline = None
    execution_deadline = None

    while True:
        if pipe_open:
            # 最多等 0.2 秒，看管道是否有数据；之后仍要检查进程状态与各类超时。
            readable, _, _ = select.select([read_fd], [], [], 0.2)

            if readable:
                chunk = os.read(read_fd, 65536)

                # 空字节串表示写端已全部关闭，并非“暂时没有消息”。
                if not chunk:
                    pipe_open = False
                else:
                    buffer.extend(chunk)

                # 只处理完整行，未收齐的尾部留到下次读取再拼接。
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
                        # ready 只响应一次，防止重复启动 Agent。
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
                        # 整次请求已收尾，取消该请求的外层超时监控。
                        execution_deadline = None

                    elif kind == "finished":
                        # 通知仅清除当前执行时限；最终完成仍需核对磁盘状态。
                        execution_deadline = None
        else:
            time.sleep(0.2)

        now = time.monotonic()

        if not ready and now >= startup_deadline:
            return "session_startup_timeout", agent

        # poll 不等待：仍在运行返回 None，已退出返回退出码。
        session_rc = session.poll()
        if session_rc is not None:
            if session_rc != 0 or not _session_finished(run):
                return "session_exited", agent

        # 通道断开而服务尚未确认完成，视为生命周期通信丢失，不能猜测成功。
        if not pipe_open and not _session_finished(run):
            return "session_event_pipe_closed", agent

        if execution_deadline is not None and now >= execution_deadline:
            return "execution_timeout", agent

        if agent is not None and agent.poll() is not None:
            # Agent 退出并不代表已经提交的实验被取消。
            # 若仍有 executing 时限，继续等待实验收尾；否则再判断正常结束或提前退出。
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

    # 同时检查格式与日期合法性，例如 202613010900 虽然全是数字仍会被拒绝。
    prefix = run.name[:12]
    if len(prefix) != 12 or not prefix.isdigit():
        raise ValueError("run-dir must start with YYYYMMDDHHMM")
    datetime.strptime(prefix, "%Y%m%d%H%M")

    # 已有运行目录不覆盖。
    run.mkdir(parents=True, exist_ok=False)

    # 先设为 None，保证某一步启动失败时，finally 也能安全清理已创建的资源。
    agent = None
    session = None
    container = None
    read_fd = None
    write_fd = None

    # runtime-result 记录整个运行的结果；session-result 记录实验服务的状态。
    # 两者职责不同：前者还要考虑 Agent 退出、外层时限和清理失败。
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

        # ExitStack 管理数量较多的日志文件，离开 with 时统一关闭。
        def open_log(name):
            """打开宿主运行目录中的日志，并登记到自动关闭列表。"""
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

            # 只做启动前检查和网络准备，不在运行时构建或下载镜像。
            launch_agent.require_image(agent_settings["image"])
            if network != "none":
                launch_agent.ensure_network(network)

            # 认证材料仅放临时目录；运行归档保留日志和证据，不归档这个 HOME。
            home = launch_agent.stage_home(temporary / "agent-home")
            container, argv = agent_command(
                cfg, workspace, home, socket_dir
            )

            # 创建单向事件管道：父进程 runner 持有读端，子进程 session 持有写端。
            read_fd, write_fd = os.pipe()

            command = [
                session_settings["python"],
                "-m", "AgenticSimulator.orchestrator.session",
                "--config", str(config_path),
                "--run-dir", str(run),
                "--socket", str(socket_dir / "session.sock"),
                "--ready-fd", str(write_fd),
            ]

            # 服务运行在宿主指定的 Python 环境中，Agent 则另在容器里运行。
            # pass_fds 明确保留事件写端；其余无关描述符不向子进程传递。
            session = subprocess.Popen(
                command,
                cwd=IMPORT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=session_stdout,
                stderr=session_stderr,
                pass_fds=(write_fd,),
                start_new_session=True,
            )

            # 父进程不写事件，立即关闭自己的写端副本，才能在子进程退出时读到 EOF。
            os.close(write_fd)
            write_fd = None

            def start_agent():
                """服务就绪后启动 Docker 客户端，并把进程句柄留给收尾逻辑。"""
                # nonlocal 修改外层 agent，而非创建一个只在回调里可见的局部变量。
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

            # supervise 一直等待某种结束条件，返回后进入统一清理流程。
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
            # 启动或监控异常保留简要错误和完整堆栈，不删除运行目录或自动重跑。
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
                # Docker 客户端退出不保证容器已停止，因此还需按名字显式删除。
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

            # 最后停止实验服务及其同组子进程，避免遗留 worker。
            try:
                stop_process(session)
            except Exception as exc:
                result["cleanup_errors"].append(str(exc))

            # write_fd 可能在启动中途尚未关闭，因此两端都按实际状态检查。
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
