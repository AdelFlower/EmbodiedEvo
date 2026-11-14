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
    """Read Isaac Lab YAML and resolve relative USD paths under project assets."""
    import yaml
    from AgenticSimulator.scenes.plate_cube import ASSETS

    class ConfigLoader(yaml.FullLoader):
        pass

    def read_slice(loader, node):
        return slice(*loader.construct_sequence(node))

    def resolve_asset_paths(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "usd_path" and isinstance(item, str):
                    # Keep URLs and absolute paths; relative paths use assets/,
                    # independent of the checkpoint location or working directory.
                    if "://" not in item and not Path(item).is_absolute():
                        value[key] = str((ASSETS / item).resolve())
                else:
                    resolve_asset_paths(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                resolve_asset_paths(item)

    ConfigLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", read_slice)
    config = yaml.load(path.read_text(), Loader=ConfigLoader)
    resolve_asset_paths(config)
    return config


def restore_config(config, values: dict) -> None:
    """Restore optional values and resolved indexes before Isaac Lab's typed update."""
    def prepare(target, data):
        if isinstance(target, dict):
            for key in set(target) - set(data):
                del target[key]
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
    parser.add_argument("--episodes-per-env", type=int, default=0, help="Evaluate this many full episodes per environment.")
    parser.add_argument("--seed", type=int, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.steps < 0 or args.episodes_per_env < 0:
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
        saved_env = load_config(checkpoint.parent / "params" / "env.yaml")
        # The lift task excludes the plate, including checkpoints from the
        # temporary plate-inclusive scene configuration.
        saved_env["scene"].pop("plate", None)
        saved_env["events"].pop("reset_plate", None)
        if "scene_config" in saved_env:
            saved_env["scene_config"]["objects"] = [
                item for item in saved_env["scene_config"]["objects"] if item["name"] == "Cube"
            ]
        if "usd_path" in saved_env["scene"]["table"]["spawn"]:
            from isaaclab.sim import UsdFileCfg
            env_cfg.scene.table.spawn = UsdFileCfg(usd_path=saved_env["scene"]["table"]["spawn"]["usd_path"])
        restore_config(env_cfg, saved_env)
        agent_cfg = RLPolicyCfg()
        restore_config(agent_cfg, load_config(checkpoint.parent / "params" / "agent.yaml"))
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = agent_cfg.device = args.device
        env_cfg.sim.use_fabric = not args.disable_fabric
        if args.seed is not None:
            env_cfg.seed = args.seed
        register_task()
        env = gym.make("lift_object", cfg=env_cfg)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            metrics = None
            if args.episodes_per_env:
                from isaaclab_arena.evaluation import attach_evaluation
                metrics = attach_evaluation(env.unwrapped, args.episodes_per_env)
            observations, _ = env.get_observations()
            step_count = 0
            print(f"[INFO] Playing checkpoint: {checkpoint}", flush=True)
            while launcher.app.is_running() and (args.steps == 0 or step_count < args.steps):
                started = time.perf_counter()
                with torch.inference_mode():
                    actions = policy(observations)
                    observations, _, _, _ = env.step(actions)
                step_count += 1
                if metrics is not None and metrics.complete:
                    break
                delay = env.unwrapped.step_dt - (time.perf_counter() - started)
                if args.real_time and delay > 0:
                    time.sleep(delay)
            if metrics is not None:
                report = {"checkpoint": str(checkpoint), "seed": env_cfg.seed, **metrics.summary()}
                (Path.cwd() / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
                if not metrics.complete:
                    raise RuntimeError("Playback ended before all requested evaluation episodes completed")
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
