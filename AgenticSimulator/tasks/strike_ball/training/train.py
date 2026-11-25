"""击球任务的源侧续训配方，由宿主传入本轮环境、父策略和输出目录。"""

from __future__ import annotations

from pathlib import Path


def configure(env_cfg, agent_cfg) -> None:
    """在创建源环境前应用训练配方中的种子。"""
    env_cfg.seed = agent_cfg.seed


def train(
    env,
    agent_cfg,
    checkpoint: str | Path,
    output_dir: str | Path,
    iterations: int,
) -> dict:
    """从父策略新增指定次数的击球 PPO 更新，返回宿主登记新策略所需的路径。"""
    import torch
    import yaml
    from isaaclab.utils.io import dump_yaml

    from AgenticSimulator.orchestrator.files import write_json
    from AgenticSimulator.tasks.strike_ball.ppo_shot import train_shots

    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    params = output / "params"
    params.mkdir()

    agent_cfg.native_updates = iterations
    # env.yaml 记录本轮候选的实际配置。下一轮训练读取本轮的平铺 agent.yaml。
    dump_yaml(str(params / "env.yaml"), env.unwrapped.cfg)
    with (params / "agent.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(vars(agent_cfg), stream, sort_keys=False)

    env.reset(seed=agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)
    train_shots(env, base=None, args=agent_cfg, log_dir=output, checkpoint=checkpoint)

    result = {
        "parent_checkpoint": str(checkpoint),
        "checkpoint": str(output / "network.pt"),
        "params_dir": str(params),
        "log_dir": str(output),
        "iterations": iterations,
        "num_envs": env.unwrapped.num_envs,
    }
    write_json(output / "training_complete.json", result)
    return result
