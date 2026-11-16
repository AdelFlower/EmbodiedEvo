# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Agent 可编辑的 lift_cube 续训入口，复用已有 RSL-RL PPO 训练流程。

执行端将 configure 传给 adapter.open_training()，再将返回的环境和配置交给
train()。checkpoint 是上一轮的策略；首次训练传入 P0。output_dir 是本次
--run-dir 下新的训练输出目录。仿真应用和环境的生命周期由执行端与 adapter 管理。
"""

from __future__ import annotations

import json
from pathlib import Path


def configure(env_cfg, agent_cfg) -> None:
    """创建环境前调整训练安排；初版沿用已恢复的奖励、课程和 PPO 参数。"""
    # Agent 可在这里修改奖励、课程、学习率等，保持任务目标与成功条件固定。
    env_cfg.seed = agent_cfg.seed
    agent_cfg.logger = "tensorboard"


def train(
    env,
    agent_cfg,
    checkpoint: str | Path,
    output_dir: str | Path,
    iterations: int,
) -> dict:
    """从 Pk-1 继续训练指定次数，保存 Pk 并返回模型、配置与日志的位置。

    iterations 表示本轮新增的 PPO 迭代次数。继承模型、观测归一化和优化器状态，
    学习率使用本轮配置。异常直接交给执行端记录，已产生的日志和中间模型保留。
    """
    import torch
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_arena.numerics import install_numerics_guard
    from rsl_rl.runners import OnPolicyRunner

    if iterations < 1:
        raise ValueError("iterations must be positive")

    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    params = output / "params"
    params.mkdir()
    agent_cfg.max_iterations = iterations

    # 沿用原训练入口的计算设置。
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    dump_yaml(str(params / "env.yaml"), env.unwrapped.cfg)
    dump_yaml(str(params / "agent.yaml"), agent_cfg)
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=str(output), device=agent_cfg.device,
    )
    runner.load(str(checkpoint), load_optimizer=True)
    # optimizer 恢复也会带回旧学习率；保留其动量，让本轮学习率设置生效。
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = agent_cfg.algorithm.learning_rate
    install_numerics_guard(runner, output)

    print(f"[INFO] Resume checkpoint: {checkpoint}", flush=True)
    print(f"[INFO] Training outputs: {output}", flush=True)
    try:
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
        # 显式保存本轮最后状态，供下一轮和 rollout 使用同一个稳定路径。
        model = output / "model.pt"
        runner.save(str(model))
    finally:
        if runner.writer is not None:
            runner.writer.close()

    result = {
        "parent_checkpoint": str(checkpoint),
        "checkpoint": str(model),
        "params_dir": str(params),
        "log_dir": str(output),
        "iterations": iterations,
        "num_envs": env.num_envs,
    }
    (output / "training_complete.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return result
