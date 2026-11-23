"""Train or resume the UR10 strike policy with RSL-RL PPO."""

import argparse
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

from . import TASK_ID, enter_run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-repeat", type=int, help="Native control steps per PPO decision")
    parser.add_argument("--checkpoint", type=Path, help="Resume PPO with optimizer and observation statistics")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.max_iterations < 1:
        parser.error("--num_envs and --max_iterations must be positive")
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    enter_run_directory(args.run_dir)
    log_dir = Path.cwd() / "rsl_rl"
    log_dir.mkdir(exist_ok=False)

    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        from isaaclab.utils.io import dump_yaml, load_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        from .env_cfg import StrikeBallEnvCfg
        from .ppo_cfg import StrikeBallPpoCfg

        cfg = StrikeBallEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed, cfg.sim.device = args.seed, args.device
        agent_cfg = StrikeBallPpoCfg()
        if checkpoint:
            agent_cfg.from_dict(load_yaml(str(checkpoint.parent / "params" / "agent.yaml")))
        if args.action_repeat is not None:
            if checkpoint and args.action_repeat != agent_cfg.action_repeat:
                raise ValueError("A resumed policy must keep its trained action repeat")
            agent_cfg.action_repeat = args.action_repeat
        agent_cfg.seed, agent_cfg.device = args.seed, args.device
        agent_cfg.max_iterations = args.max_iterations
        env = gym.make(TASK_ID, cfg=cfg)
        try:
            horizon = env.unwrapped.max_episode_length
            repeat = agent_cfg.action_repeat
            if repeat < 1 or horizon % repeat:
                raise ValueError(f"--action-repeat must divide the {horizon}-step episode")
            env.unwrapped.action_repeat = repeat
            if repeat > 1 and not checkpoint:
                agent_cfg.num_steps_per_env = horizon // repeat
                agent_cfg.algorithm.gamma **= repeat
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            dump_yaml(str(log_dir / "params/env.yaml"), cfg)
            dump_yaml(str(log_dir / "params/agent.yaml"), agent_cfg)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=args.device)
            if checkpoint:
                runner.load(str(checkpoint))
                # RSL-RL 2.3 stores the index of the last completed update.
                runner.current_learning_iteration += 1
            print(f"PPO outputs: {log_dir}", flush=True)
            runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
            runner.save(str(log_dir / "model.pt"))
            if runner.writer is not None:
                runner.writer.close()
        finally:
            env.close()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        return 1
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
