"""构造 coding-agent 命令，并准备其容器运行环境。

进程启动、停止、session socket 和工作区只读挂载由 session_runner 管理。
本模块不负责仿真执行、版本管理或实验评估。

本文件主要“拼命令”：先生成容器内的 Agent 命令，再包成 docker run 命令。
argv 表示参数列表，每个元素是一个完整参数；调用 subprocess 时无需通过
shell 解释，也不用把带空格的参数手动拼成一条命令字符串。
宿主机是启动 Docker 的机器；容器是 Agent 所在的隔离运行环境。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 使用 tomli 依赖。


# 以下是容器内部看到的路径，与宿主机上真实的工作区和临时 HOME 不同。
# docker_argv 会用 bind mount 将两边对应起来，即把宿主目录映射进容器。
CONTAINER_WORKSPACE = Path("/workspace")
CONTAINER_HOME = Path("/home/agent")
AGENT_NETWORK = "agenticsimulator-agent"


def agent_argv(
    preset: str,
    prompt: str,
    *,
    model: str | None = None,
    effort: str | None = None,
) -> list[str]:
    """根据 preset（CLI 类型）生成命令，返回参数列表，不启动进程。"""
    if preset == "codex":
        argv = [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
        ]
        # 用 JSON 表示字符串，让 CLI 的配置解析器正确处理引号等字符。
        if model:
            argv += ["-c", f"model={json.dumps(model)}"]
        if effort:
            argv += ["-c", f"model_reasoning_effort={json.dumps(effort)}"]
        return [*argv, prompt]

    if preset == "claude":
        # 两种 CLI 的参数不同；此分支仅转交 model，不使用 effort 参数。
        argv = ["claude", "--print", "--permission-mode", "acceptEdits"]
        if model:
            argv += ["--model", model]
        return [*argv, prompt]

    raise ValueError(f"Unknown agent preset: {preset!r}")


def adapt_for_docker(argv: list[str], *, preset: str) -> list[str]:
    """复制并调整命令，让容器内 Codex 依赖外部 Docker 提供隔离。"""
    # 复制列表，避免把调用方保留的原命令也一起修改。
    adapted = list(argv)
    if preset != "codex":
        return adapted

    try:
        index = adapted.index("--sandbox")
    except ValueError as exc:
        raise ValueError("Codex argv is missing --sandbox") from exc

    if index + 1 >= len(adapted):
        raise ValueError("Codex argv has an incomplete --sandbox option")

    # 删除 --sandbox 及其值，再插入容器内的非交互执行选项。
    # 此选项关闭的是 CLI 自身的审批和沙箱；Docker 的挂载与资源限制仍生效。
    del adapted[index : index + 2]
    adapted.insert(2, "--dangerously-bypass-approvals-and-sandbox")
    return adapted


def docker_argv(
    workspace: Path,
    home: Path,
    inner: list[str],
    *,
    image: str,
    network: str,
    cpus: int,
    memory_mb: int,
    container_name: str | None = None,
) -> list[str]:
    """构造容器命令；socket 和只读子目录挂载由调用方添加。"""
    if cpus <= 0 or memory_mb <= 0:
        raise ValueError("Docker CPU and memory limits must be positive")

    workspace = workspace.resolve()
    home = home.resolve()
    # 进程号加纳秒时间避免容器重名；runner 会保留名字用于最后强制清理。
    name = container_name or (
        f"agenticsimulator-agent-{os.getpid()}-{time.time_ns()}"
    )

    # 优先使用 Agent 专用代理，再回退到宿主通用代理；无网络模式不传代理。
    proxy_args = []
    proxy = (
        os.environ.get("AGENTICSIMULATOR_AGENT_HTTPS_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
    )
    if network != "none" and proxy:
        proxy_args = [
            "--env", f"HTTPS_PROXY={proxy}",
            "--env", f"HTTP_PROXY={proxy}",
            "--env", "NO_PROXY=localhost,127.0.0.1,::1",
        ]

    # cpuset 限定使用哪些 CPU 核；--cpus 则限制总配额，两者可以同时存在。
    cpu_args = []
    cpuset = os.environ.get("AGENTICSIMULATOR_AGENT_CPUSET")
    if cpuset:
        cpu_args = ["--cpuset-cpus", cpuset]

    return [
        "docker", "run",
        # --rm 退出后删容器；--init 帮助回收子进程；根文件系统设为只读。
        "--rm",
        "--init",
        "--read-only",
        # /tmp 为容器内临时可写区；使用宿主 UID/GID 避免生成 root 所有的文件。
        "--tmpfs", "/tmp:mode=1777",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--workdir", str(CONTAINER_WORKSPACE),

        # 这些变量只传给容器，指向本次运行专用的 HOME 和工作区。
        "--env", f"HOME={CONTAINER_HOME}",
        "--env", f"CODEX_HOME={CONTAINER_HOME / '.codex'}",
        "--env", f"AGENTICSIMULATOR_WORKSPACE={CONTAINER_WORKSPACE}",
        "--env", "CODEX_CI=1",
        "--env", "LANG=C.UTF-8",
        "--env", "LC_ALL=C.UTF-8",
        "--env", f"TERM={os.environ.get('TERM', 'xterm-256color')}",
        *proxy_args,

        # 整体工作区和临时 HOME 可写；runner 随后为指令和反馈等子路径补只读挂载。
        "--mount", f"type=bind,src={workspace},dst={CONTAINER_WORKSPACE}",
        "--mount", f"type=bind,src={home},dst={CONTAINER_HOME}",

        "--network", network,
        # 移除额外 Linux 权限，禁止进程获取新权限，并限制进程数、内存和 CPU。
        # 本命令没有挂载模拟器或 GPU，实验交互由 runner 补入的 socket 完成。
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "512",
        "--memory", f"{memory_mb}m",
        "--cpus", str(cpus),
        *cpu_args,

        "--name", name,
        # Docker 选项必须放在 image 之前；之后的 inner 才是容器内执行的命令。
        image,
        *inner,
    ]


def _docker(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """运行一次 Docker 检查命令，收集输出，把返回码交给调用方判断。"""
    return subprocess.run(
        ["docker", *argv],
        capture_output=True,
        text=True,
        check=False,
    )


def require_image(image: str) -> None:
    """沿用外层的启动检查，不构建或下载镜像。"""
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required to launch the agent")

    # docker 命令存在不代表后台 Docker 服务可用，因此还要查询服务端版本。
    daemon = _docker(["version", "--format", "{{.Server.Version}}"])
    if daemon.returncode != 0 or not daemon.stdout.strip():
        detail = daemon.stderr.strip() or "Docker daemon unavailable"
        raise RuntimeError(f"Cannot reach the Docker daemon: {detail}")

    # 这里只验证本地镜像，缺少时直接报错，不自动 pull 或 build。
    if _docker(["image", "inspect", image]).returncode != 0:
        raise RuntimeError(f"Agent image {image!r} is not available locally")


def ensure_network(network: str = AGENT_NETWORK) -> None:
    """沿用外层：确保专用 Docker bridge 存在。"""
    if _docker(["network", "inspect", network]).returncode == 0:
        return

    created = _docker(["network", "create", "--driver", "bridge", network])
    # 可能另一个进程已同时创建同名网络，所以创建失败后再查询一次。
    if (
        created.returncode != 0
        and _docker(["network", "inspect", network]).returncode != 0
    ):
        raise RuntimeError(
            f"Cannot create Docker network {network!r}: "
            f"{created.stderr.strip()}"
        )


def _toml_value(value):
    """把配置中使用的字典、列表和标量转成 TOML 文本。"""
    # 字典使用 { key = value } 内联表，嵌套值递归处理。
    if isinstance(value, dict):
        entries = (
            f"{json.dumps(key)} = {_toml_value(item)}"
            for key, item in value.items()
        )
        return "{ " + ", ".join(entries) + " }"

    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"

    return json.dumps(value)


def agent_codex_home() -> Path:
    """选择配置来源，不混用不同来源的配置和凭据。"""
    # 来源优先级：本项目专用变量 → CODEX_HOME → 当前用户的 ~/.codex。
    # 选定一个目录后，配置和认证文件都从那里取。
    override = os.environ.get("AGENTICSIMULATOR_CODEX_HOME")
    source = (
        Path(override).expanduser()
        if override
        else Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    )

    if not (source / "config.toml").is_file():
        raise ValueError(f"Missing Codex config: {source / 'config.toml'}")

    return source


def stage_home(root: Path) -> Path:
    """在调用方提供的新临时目录中准备独立 HOME。"""
    # 仅复制运行所需部分，容器对这个临时副本的写入不会修改宿主原配置。
    home = Path(root)
    codex = home / ".codex"
    codex.mkdir(parents=True, exist_ok=True)

    source = agent_codex_home()

    # 如果使用文件认证，复制到临时 HOME，0600 表示只有所属用户可读写。
    # 该临时目录由 runner 管理，不能放进运行归档或 Git。
    auth = source / "auth.json"
    if auth.is_file():
        destination = codex / "auth.json"
        shutil.copy2(auth, destination)
        destination.chmod(0o600)

    data = tomllib.loads(
        (source / "config.toml").read_text(encoding="utf-8")
    )

    # 保留模型和 provider 配置，不复制宿主插件、MCP、历史或额外指令。
    keys = (
        "model",
        "model_provider",
        "model_providers",
        "model_reasoning_effort",
        "model_context_window",
        "model_auto_compact_token_limit",
        "service_tier",
    )

    # 按白名单逐项重写配置，而不是原样复制整个 config.toml。
    lines = [
        f"{key} = {_toml_value(data[key])}"
        for key in keys
        if key in data
    ]
    # 告知 CLI 本次容器内工作目录是可信项目，用于自动执行流程。
    lines += [
        "",
        f'[projects."{CONTAINER_WORKSPACE}"]',
        'trust_level = "trusted"',
        "",
    ]

    config = codex / "config.toml"
    config.write_text("\n".join(lines), encoding="utf-8")
    config.chmod(0o600)
    return home
