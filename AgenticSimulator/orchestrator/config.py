# 实验、模型、预算和环境配置
"""加载与下游任务解耦的框架配置。

约定：
- 配置文件使用 JSON。
- 框架路径相对于配置文件所在目录解析。
- 返回值可以直接写入运行目录的配置快照。
- experiment 中的任务专用内容由执行层解释。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .launch_agent import AGENT_NETWORK


def _object(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _text(value, name: str) -> str:
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
    path = Path(_text(value, name)).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()

    if directory:
        if not path.is_dir():
            raise ValueError(f"{name} is not a directory: {path}")
    elif not path.is_file():
        raise ValueError(f"{name} is not a file: {path}")

    return str(path)


def _positive(value, name: str) -> None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


def load_config(path: str | Path) -> dict:
    """读取配置、补默认值并检查框架使用的字段。"""
    config_path = Path(path).expanduser().resolve()
    cfg = _object(
        json.loads(config_path.read_text(encoding="utf-8")),
        "config",
    )
    base = config_path.parent

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

    agent = _object(cfg.get("agent"), "agent")
    session = _object(cfg.get("session"), "session")

    # Agent 启动设置。
    preset = agent.get("preset")
    if preset not in ("codex", "claude"):
        raise ValueError("agent.preset must be codex or claude")

    _text(agent.get("image"), "agent.image")
    _text(agent.get("model"), "agent.model")

    agent.setdefault("reasoning_effort", None)
    if agent["reasoning_effort"] is not None:
        _text(agent["reasoning_effort"], "agent.reasoning_effort")

    agent.setdefault("network", AGENT_NETWORK)
    agent.setdefault("cpus", 4)
    agent.setdefault("memory_mb", 8192)

    _text(agent["network"], "agent.network")
    _positive(agent["cpus"], "agent.cpus")

    if type(agent["memory_mb"]) is not int or agent["memory_mb"] <= 0:
        raise ValueError("agent.memory_mb must be a positive integer")

    # session 服务的解释器，与仿真 worker 的运行环境分开。
    session["python"] = _path(
        session.get("python"),
        "session.python",
        base,
    )

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

    for name in ("source_budget", "target_budget"):
        value = session[name]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(
                f"session.{name} must be a nonnegative integer or null"
            )

    # 预留任务接入信息，不在这里加载或解释具体任务。
    experiment = _object(
        cfg.setdefault("experiment", {}),
        "experiment",
    )

    if "adapter" in experiment:
        _text(experiment["adapter"], "experiment.adapter")

    if "config" in experiment:
        experiment["config"] = _path(
            experiment["config"],
            "experiment.config",
            base,
        )

    return cfg
