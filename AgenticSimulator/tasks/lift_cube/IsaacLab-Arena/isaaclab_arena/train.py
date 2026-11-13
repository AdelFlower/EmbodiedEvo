# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Train the cube-lifting task with Isaac Sim 4.5 and RSL-RL 2.3."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
import sys

TASK_NAME = "lift_object"


def register_task() -> None:
    """Register task metadata without importing simulator code before startup."""
    import gymnasium as gym

    assert TASK_NAME not in gym.registry, f"Task name already registered: {TASK_NAME}"
    gym.register(
        id=TASK_NAME,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isaaclab_arena.env_cfg:LiftCubeEnvCfg",
            "rsl_rl_cfg_entry_point": "isaaclab_arena.ppo_cfg:RLPolicyCfg",
        },
    )


def main(argv: list[str] | None = None) -> int:
    # AppLauncher is the only Isaac Lab import permitted before SimulationApp.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env_spacing", type=float, default=30.0)
    parser.add_argument("--disable_fabric", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Resume from this checkpoint in a new run directory.")
    parser.add_argument("--rl_training_mode", action=argparse.BooleanOptionalAction, default=True)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args(argv)
    if args.num_envs is not None and args.num_envs < 1:
        parser.error("--num_envs must be positive")
    if args.max_iterations is not None and args.max_iterations < 1:
        parser.error("--max_iterations must be positive")
    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0], *hydra_args]
    launcher = AppLauncher(args)
    try:
        import gymnasium as gym
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.hydra import hydra_task_config

        register_task()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

        @hydra_task_config(TASK_NAME, "rsl_rl_cfg_entry_point")
        def train(env_cfg, agent_cfg):
            if args.num_envs is not None:
                env_cfg.scene.num_envs = args.num_envs
            if args.max_iterations is not None:
                agent_cfg.max_iterations = args.max_iterations
            env_cfg.seed = agent_cfg.seed = args.seed
            env_cfg.scene.env_spacing = args.env_spacing
            env_cfg.sim.device = agent_cfg.device = args.device
            env_cfg.sim.use_fabric = not args.disable_fabric
            env_cfg.terminations.success.params["rl_training"] = args.rl_training_mode
            log_dir = Path.cwd() / "logs" / "rsl_rl" / agent_cfg.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f"[INFO] Training outputs: {log_dir}", flush=True)
            env = gym.make(TASK_NAME, cfg=env_cfg)
            try:
                env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                # The local RSL-RL runner consumes the policy group for both actor and
                # critic; env_cfg preserves the former [policy, task_obs] term order.
                dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
                dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
                if args.checkpoint is not None:
                    runner.load(str(args.checkpoint.expanduser().resolve()))
                runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
                (Path.cwd() / "training_complete.json").write_text(json.dumps({
                    "iterations": agent_cfg.max_iterations, "num_envs": env_cfg.scene.num_envs,
                    "log_dir": str(log_dir),
                }, indent=2) + "\n")
            finally:
                env.close()

        train()
        return 0
    except BaseException:
        # Kit shutdown can terminate Python before an unhandled exception is
        # printed. Preserve the original traceback before closing the app.
        traceback.print_exc()
        sys.stderr.flush()
        return 1
    finally:
        launcher.app.close()
        sys.argv = saved_argv


if __name__ == "__main__":
    raise SystemExit(main())
