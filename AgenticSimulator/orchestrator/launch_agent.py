"""构造 coding-agent 命令，并准备其容器运行环境。

进程启动、停止、session socket 和工作区只读挂载由 session_runner 管理。
本模块不负责仿真执行、版本管理或实验评估。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import tomllib
from pathlib import Path


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
    """沿用外层：显式指定本次运行的模型与推理配置。"""
    if preset == "codex":
        argv = [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
        ]
        if model:
            argv += ["-c", f"model={json.dumps(model)}"]
        if effort:
            argv += ["-c", f"model_reasoning_effort={json.dumps(effort)}"]
        return [*argv, prompt]

    if preset == "claude":
        argv = ["claude", "--print", "--permission-mode", "acceptEdits"]
        if model:
            argv += ["--model", model]
        return [*argv, prompt]

    raise ValueError(f"Unknown agent preset: {preset!r}")


def adapt_for_docker(argv: list[str], *, preset: str) -> list[str]:
    """沿用外层：在 Docker 已提供隔离时调整 CLI 的执行模式。"""
    adapted = list(argv)
    if preset != "codex":
        return adapted

    try:
        index = adapted.index("--sandbox")
    except ValueError as exc:
        raise ValueError("Codex argv is missing --sandbox") from exc

    if index + 1 >= len(adapted):
        raise ValueError("Codex argv has an incomplete --sandbox option")

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
    name = container_name or (
        f"agenticsimulator-agent-{os.getpid()}-{time.time_ns()}"
    )

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

    cpu_args = []
    cpuset = os.environ.get("AGENTICSIMULATOR_AGENT_CPUSET")
    if cpuset:
        cpu_args = ["--cpuset-cpus", cpuset]

    return [
        "docker", "run",
        "--rm",
        "--init",
        "--read-only",
        "--tmpfs", "/tmp:mode=1777",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--workdir", str(CONTAINER_WORKSPACE),

        "--env", f"HOME={CONTAINER_HOME}",
        "--env", f"CODEX_HOME={CONTAINER_HOME / '.codex'}",
        "--env", f"AGENTICSIMULATOR_WORKSPACE={CONTAINER_WORKSPACE}",
        "--env", "CODEX_CI=1",
        "--env", "LANG=C.UTF-8",
        "--env", "LC_ALL=C.UTF-8",
        "--env", f"TERM={os.environ.get('TERM', 'xterm-256color')}",
        *proxy_args,

        "--mount", f"type=bind,src={workspace},dst={CONTAINER_WORKSPACE}",
        "--mount", f"type=bind,src={home},dst={CONTAINER_HOME}",

        "--network", network,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "512",
        "--memory", f"{memory_mb}m",
        "--cpus", str(cpus),
        *cpu_args,

        "--name", name,
        image,
        *inner,
    ]


def _docker(argv: list[str]) -> subprocess.CompletedProcess[str]:
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

    daemon = _docker(["version", "--format", "{{.Server.Version}}"])
    if daemon.returncode != 0 or not daemon.stdout.strip():
        detail = daemon.stderr.strip() or "Docker daemon unavailable"
        raise RuntimeError(f"Cannot reach the Docker daemon: {detail}")

    if _docker(["image", "inspect", image]).returncode != 0:
        raise RuntimeError(f"Agent image {image!r} is not available locally")


def ensure_network(network: str = AGENT_NETWORK) -> None:
    """沿用外层：确保专用 Docker bridge 存在。"""
    if _docker(["network", "inspect", network]).returncode == 0:
        return

    created = _docker(["network", "create", "--driver", "bridge", network])
    if (
        created.returncode != 0
        and _docker(["network", "inspect", network]).returncode != 0
    ):
        raise RuntimeError(
            f"Cannot create Docker network {network!r}: "
            f"{created.stderr.strip()}"
        )


def _toml_value(value):
    """沿用外层的配置值序列化。"""
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
    home = Path(root)
    codex = home / ".codex"
    codex.mkdir(parents=True, exist_ok=True)

    source = agent_codex_home()

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

    lines = [
        f"{key} = {_toml_value(data[key])}"
        for key in keys
        if key in data
    ]
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