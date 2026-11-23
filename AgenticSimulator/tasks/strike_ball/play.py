"""Play a strike policy in Isaac, or score a fixed number of full episodes."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

from . import TASK_ID, enter_run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--episodes", type=int, default=0, help="Total episodes to score (0 plays indefinitely)"
    )
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
        from .scene import HOLE_CENTER, SAMPLE_Z

        cfg = StrikeBallEnvCfg()
        cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
        env = gym.make(TASK_ID, cfg=cfg)
        try:
            has_ppo_config = (checkpoint.parent / "params/agent.yaml").exists()
            if checkpoint.name == "trajectory_policy.pt" or not has_ppo_config:
                bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if bundle.get("format") == "strike-ball-ppo-network-v1":
                    from .ppo_network import FrozenStrikeNetwork

                    policy = FrozenStrikeNetwork(bundle, args.device)
                elif bundle.get("format") in {"strike-ball-trajectory-v1", "strike-ball-trajectory-v2"}:
                    from .train_swing import TrajectoryPolicy

                    policy = TrajectoryPolicy(bundle, args.device)
                else:
                    raise ValueError("Unrecognized policy checkpoint format")
                env = RslRlVecEnvWrapper(env, clip_actions=10.)
            else:
                agent_cfg = load_yaml(str(checkpoint.parent / "params" / "agent.yaml"))
                env.unwrapped.action_repeat = agent_cfg.get("action_repeat", 1)
                env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg["clip_actions"])
                runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=args.device)
                runner.load(str(checkpoint), load_optimizer=False)
                policy = runner.get_inference_policy(device=args.device)
            if args.episodes:
                # Start the evaluation RNG sequence after checkpoint-specific initialization.
                env.seed(args.seed)
                obs, _ = env.reset()
            else:
                obs, _ = env.get_observations()
            real_time = env.unwrapped.sim.has_gui() if args.real_time is None else args.real_time
            count = hits = passes = crossings = 0
            error_sum = 0.
            # Observation 26 is the height of the hole relative to the ball.
            start_height = HOLE_CENTER[2] - obs[:, 26]
            measured_min = float("inf")
            measured_max = float("-inf")
            height_edges = torch.linspace(*SAMPLE_Z, steps=5, device=obs.device)
            band_episodes = torch.zeros(4, device=obs.device, dtype=torch.long)
            band_passes = torch.zeros_like(band_episodes)
            height_groups = {name: {"episodes": 0, "passes": 0}
                             for name in ("below_board", "at_or_above_board")}
            with torch.inference_mode():
                while app.is_running() and (not args.episodes or count < args.episodes):
                    start = time.monotonic()
                    obs, _, done, extras = env.step(policy(obs))
                    if "shot" in extras:
                        shot = extras["shot"]
                        n = len(shot["hit"])
                        if args.episodes:
                            n = min(n, args.episodes - count)
                        finished = done.nonzero().flatten()
                        heights = start_height[finished[:n]]
                        low = heights < HOLE_CENTER[2]
                        if args.episodes:
                            measured_min = min(measured_min, heights.min().item())
                            measured_max = max(measured_max, heights.max().item())
                            bands = torch.bucketize(heights, height_edges[1:-1])
                            band_episodes += torch.bincount(bands, minlength=4)
                            band_passes += torch.bincount(bands[shot["passed"][:n]], minlength=4)
                        for name, mask in (("below_board", low), ("at_or_above_board", ~low)):
                            height_groups[name]["episodes"] += int(mask.sum())
                            height_groups[name]["passes"] += int((shot["passed"][:n] & mask).sum())
                        start_height[finished] = HOLE_CENTER[2] - obs[finished, 26]
                        count += n
                        hits += int(shot["hit"][:n].sum())
                        passes += int(shot["passed"][:n].sum())
                        error = shot["crossing_error"][:n]
                        measured = error[error.isfinite()]
                        crossings += len(measured)
                        error_sum += float(measured.sum())
                        print(f"Episodes {count}, paddle hits {hits}, passes {passes}", flush=True)
                    if real_time:
                        interval = env.unwrapped.step_dt * env.unwrapped.action_repeat
                        time.sleep(max(0., interval - (time.monotonic() - start)))
            if args.episodes:
                report = {
                    "checkpoint": str(checkpoint), "seed": args.seed, "episodes": count,
                    "hits": hits, "passes": passes, "hit_rate": hits / count if count else None,
                    "pass_rate": passes / count if count else None,
                    "pass_given_hit": passes / hits if hits else None, "crossings": crossings,
                    "by_initial_height": height_groups,
                    "configured_height_m": list(SAMPLE_Z),
                    "observed_height_m": [measured_min, measured_max] if count else None,
                    "height_bands": [
                        {"min_m": lo, "max_m": hi, "episodes": total, "passes": successful}
                        for lo, hi, total, successful in zip(height_edges[:-1].tolist(), height_edges[1:].tolist(),
                                                            band_episodes.tolist(), band_passes.tolist())
                    ],
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
