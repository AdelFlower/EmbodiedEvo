# 实验、模型、预算和环境配置
"""加载与下游任务解耦的框架配置。

约定：
- 人工配置集中在任务目录的 task.toml。
- 框架路径相对于配置文件所在目录解析。
- 返回值写入运行目录的 JSON 快照，供 session 使用。
- experiment 内嵌实验设置，由执行层结合 config_dir 解析任务路径。

可以把本文件看成“配置检查站”：读取配置 → 补齐可选项 → 检查格式 →
把框架需要的路径变成绝对路径。这里只检查配置，不启动容器或模拟器。
TOML 不支持 null；可选的总时限和预算字段省略时，加载器补为 None。
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 使用 tomli 依赖。

from .launch_agent import AGENT_NETWORK


def _object(value, name: str) -> dict:
    """要求值为字典；name 用于说明配置中的哪个字段填错了。"""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a table")
    return value


def _text(value, name: str) -> str:
    """要求值为非空字符串；strip 只用于检查，不会改写原始值。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _path(
    value,
    name: str,
    base: Path,
    *,
    directory: bool = False,
) -> str:
    """解析明确的文件系统路径，不通过 PATH 查找命令。"""
    # base 是配置文件所在目录，因此换一个终端工作目录不会改变路径含义。
    # 例如配置中的 ./template，始终相对于配置文件寻找。
    path = Path(_text(value, name)).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()

    # directory=True 检查目录，否则检查文件；最终统一返回可写入 JSON 的字符串。
    if directory:
        if not path.is_dir():
            raise ValueError(f"{name} is not a directory: {path}")
    elif not path.is_file():
        raise ValueError(f"{name} is not a file: {path}")

    return str(path)


def _positive(value, name: str) -> None:
    """检查有限的正数，拒绝 bool、NaN、无穷大、零和负数。"""
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


def load_config(path: str | Path) -> dict:
    """读取 task.toml、补默认值并检查框架使用的字段。"""
    # 下面会直接修改读出的 cfg 及其子字典，返回的是补齐后的配置。
    config_path = Path(path).expanduser().resolve()
    cfg = _object(
        tomllib.loads(config_path.read_text(encoding="utf-8")),
        "config",
    )
    base = config_path.parent
    # 随运行快照传给 worker；实验路径始终相对于原 task.toml，非快照目录。
    cfg["config_dir"] = str(base)

    # 模板提供指令、工具等工作区初始内容；source_root 提供待修改的源实现。
    # 两者在 runner 创建工作区时分别复制，并不会在这里执行。
    cfg["workspace_template"] = _path(
        cfg.get("workspace_template"),
        "workspace_template",
        base,
        directory=True,
    )
    cfg["source_root"] = _path(
        cfg.get("source_root"),
        "source_root",
        base,
        directory=True,
    )

    # agent 管“模型在哪个容器里运行”，session 管“实验服务如何运行”。
    agent = _object(cfg.get("agent"), "agent")
    session = _object(cfg.get("session"), "session")

    # Agent 启动设置。
    preset = agent.get("preset")
    if preset not in ("codex", "claude"):
        raise ValueError("agent.preset must be codex or claude")

    _text(agent.get("image"), "agent.image")
    _text(agent.get("model"), "agent.model")

    # setdefault 仅在字段缺失时填默认值，保留用户明确提供的设置。
    agent.setdefault("reasoning_effort", None)
    if agent["reasoning_effort"] is not None:
        _text(agent["reasoning_effort"], "agent.reasoning_effort")

    # cpus 是容器可用 CPU 配额，memory_mb 是内存上限，单位为 MB。
    agent.setdefault("network", AGENT_NETWORK)
    agent.setdefault("cpus", 4)
    agent.setdefault("memory_mb", 8192)

    _text(agent["network"], "agent.network")
    _positive(agent["cpus"], "agent.cpus")

    if type(agent["memory_mb"]) is not int or agent["memory_mb"] <= 0:
        raise ValueError("agent.memory_mb must be a positive integer")

    # 默认沿用启动调度的解释器；仅需要独立环境时才显式配置路径。
    python_path = session.get("python")
    session["python"] = (
        _path(python_path, "session.python", base)
        if python_path is not None else sys.executable
    )

    # 三种时限各管一段流程：startup 等服务就绪；execution 限制单次请求；
    # timeout 从 Agent 启动后限制整段会话，None 表示不设该总时限。
    session.setdefault("startup_timeout_sec", 120)
    session.setdefault("timeout_sec", None)
    session.setdefault("source_budget", None)
    session.setdefault("target_budget", None)

    _positive(
        session["startup_timeout_sec"],
        "session.startup_timeout_sec",
    )
    _positive(
        session.get("execution_timeout_sec"),
        "session.execution_timeout_sec",
    )

    if session["timeout_sec"] is not None:
        _positive(session["timeout_sec"], "session.timeout_sec")

    # 预算按每侧实际启动的实验计数；0 表示不允许启动，None 表示不设上限。
    # type(...) is int 会拒绝 True/False，避免把布尔值误当成次数。
    for name in ("source_budget", "target_budget"):
        value = session[name]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(
                f"session.{name} must be a nonnegative integer or null"
            )

    # 实验设置直接内嵌在 task.toml，不再读取第二份配置文件。
    experiment = _object(cfg.get("experiment"), "experiment")

    # adapter 指定接入方式；这里只检查名字，实际加载工作由执行层负责。
    _text(experiment.get("adapter"), "experiment.adapter")

    # policy、目标物理等路径由 worker 按 config_dir 解析；source.physics
    # 保持相对候选目录的含义。此处不导入 adapter 或解释具体任务配置。
    return cfg
