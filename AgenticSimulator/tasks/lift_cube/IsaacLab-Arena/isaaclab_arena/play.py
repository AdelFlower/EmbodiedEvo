# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Display deterministic policy rollouts using a checkpoint's saved configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback


def load_config(path: Path) -> dict:
    """Read Isaac Lab YAML, including its serialized tuple and slice values."""
    import yaml

    class ConfigLoader(yaml.FullLoader):
        pass

    def read_slice(loader, node):
        return slice(*loader.construct_sequence(node))

    ConfigLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", read_slice)
    return yaml.load(path.read_text(), Loader=ConfigLoader)


def restore_config(config, values: dict) -> None:
    """Restore optional values and resolved indexes before Isaac Lab's typed update."""
    def prepare(target, data):
        for key, value in data.items():
            current = target.get(key) if isinstance(target, dict) else getattr(target, key)
            # Isaac Lab 2's updater cannot populate None defaults or replace a
            # default slice with the explicit index list saved after training.
            if current is None or isinstance(current, slice) and isinstance(value, list):
                if isinstance(target, dict):
                    target[key] = value
                else:
                    setattr(target, key, value)
                current = value
            if isinstance(value, dict):
                prepare(current, value)
            elif isinstance(value, (list, tuple)):
                for item, saved in zip(current, value):
                    if isinstance(saved, dict):
                        prepare(item, saved)
    prepare(config, values)
    config.from_dict(values)


def main(argv: list[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps; 0 runs until the window closes.")
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable_fabric", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.steps < 0:
        parser.error("--num_envs must be positive and --steps must be nonnegative")
    launcher = AppLauncher(args)
    try:
        import gymnasium as gym
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_arena.env_cfg import LiftCubeEnvCfg
        from isaaclab_arena.ppo_cfg import RLPolicyCfg
        from isaaclab_arena.train import register_task

        checkpoint = args.checkpoint.resolve()
        env_cfg = LiftCubeEnvCfg()
        restore_config(env_cfg, load_config(checkpoint.parent / "params" / "env.yaml"))
        agent_cfg = RLPolicyCfg()
        restore_config(agent_cfg, load_config(checkpoint.parent / "params" / "agent.yaml"))
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = agent_cfg.device = args.device
        env_cfg.sim.use_fabric = not args.disable_fabric
        register_task()
        env = gym.make("lift_object", cfg=env_cfg)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            observations, _ = env.get_observations()
            step_count = 0
            print(f"[INFO] Playing checkpoint: {checkpoint}", flush=True)
            while launcher.app.is_running() and (args.steps == 0 or step_count < args.steps):
                started = time.perf_counter()
                with torch.inference_mode():
                    actions = policy(observations)
                    observations, _, _, _ = env.step(actions)
                step_count += 1
                delay = env.unwrapped.step_dt - (time.perf_counter() - started)
                if args.real_time and delay > 0:
                    time.sleep(delay)
            (Path.cwd() / "playback_complete.json").write_text(json.dumps({
                "checkpoint": str(checkpoint), "steps": step_count, "num_envs": args.num_envs,
            }, indent=2) + "\n")
        finally:
            env.close()
        return 0
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        return 1
    finally:
        launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
