"""Play a strike policy in Isaac, or score a fixed number of full episodes."""

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from isaaclab.app import AppLauncher

from . import TASK_ID, enter_run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--episodes", type=int, default=0, help="Total episodes to score; 0 plays indefinitely")
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.episodes < 0:
        parser.error("--num_envs must be positive and --episodes must be nonnegative")
    checkpoint = args.checkpoint.expanduser().resolve()
    enter_run_directory(args.run_dir)
    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import torch
        from isaaclab.utils.io import load_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        from .env_cfg import StrikeBallEnvCfg

        cfg = StrikeBallEnvCfg()
        cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
        agent_cfg = load_yaml(str(checkpoint.parent / "params" / "agent.yaml"))
        env = gym.make(TASK_ID, cfg=cfg)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg["clip_actions"])
            runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=args.device)
            runner.load(str(checkpoint), load_optimizer=False)
            policy = runner.get_inference_policy(device=args.device)
            obs, _ = env.get_observations()
            real_time = env.unwrapped.sim.has_gui() if args.real_time is None else args.real_time
            count = hits = passes = crossings = 0
            error_sum = 0.
            with torch.inference_mode():
                while app.is_running() and (not args.episodes or count < args.episodes):
                    start = time.monotonic()
                    obs, _, _, extras = env.step(policy(obs))
                    if "shot" in extras:
                        shot = extras["shot"]
                        n = len(shot["hit"])
                        if args.episodes:
                            n = min(n, args.episodes - count)
                        count += n
                        hits += int(shot["hit"][:n].sum())
                        passes += int(shot["passed"][:n].sum())
                        error = shot["crossing_error"][:n]
                        measured = error[error.isfinite()]
                        crossings += len(measured)
                        error_sum += float(measured.sum())
                        print(f"Episodes {count}, paddle hits {hits}, passes {passes}", flush=True)
                    if real_time:
                        time.sleep(max(0., env.unwrapped.step_dt - (time.monotonic() - start)))
            if args.episodes:
                report = {
                    "checkpoint": str(checkpoint), "seed": args.seed, "episodes": count,
                    "hits": hits, "passes": passes, "hit_rate": hits / count if count else None,
                    "pass_rate": passes / count if count else None,
                    "pass_given_hit": passes / hits if hits else None, "crossings": crossings,
                    "mean_crossing_error_mm": 1000 * error_sum / crossings if crossings else None,
                }
                Path("evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
                if count < args.episodes:
                    raise RuntimeError(f"Playback ended after {count} of {args.episodes} episodes")
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
