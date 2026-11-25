"""Train a randomly initialized six-joint strike policy with clipped PPO.

From the repository root, run

    conda run --no-capture-output -n isaac-sim python -m AgenticSimulator.tasks.strike_ball.train_network_ppo --headless --run-dir "AgenticSimulator/runs/$(date +%Y%m%d%H%M)_strike_ppo"

The default recipe verifies a native pass-rate target on two batches before stopping
early. It reloads neural_ppo/network.pt and writes the independently seeded replay
counts to neural_ppo/evaluation.json.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from . import TASK_ID, enter_run_directory


def evaluate_saved_network(env, checkpoint, seed, episodes):
    import torch
    from .ppo_network import FrozenStrikeNetwork
    from .scene import HOLE_CENTER, SAMPLE_Z

    policy = FrozenStrikeNetwork(torch.load(checkpoint, map_location="cpu", weights_only=True), env.unwrapped.device)
    obs = env.reset(seed=seed)[0]["policy"]
    count = hits = passes = 0
    measured_min, measured_max = float("inf"), float("-inf")
    with torch.inference_mode():
        while count < episodes:
            start_height = HOLE_CENTER[2] - obs[:, 26]
            for _ in range(env.unwrapped.max_episode_length):
                obs, _, _, _, extras = env.step(policy(obs))
                obs = obs["policy"]
            shot = extras["shot"]
            take = min(len(start_height), episodes - count)
            passed = shot["passed"][:take]
            hits += int(shot["hit"][:take].sum())
            passes += int(passed.sum())
            measured_min = min(measured_min, start_height[:take].min().item())
            measured_max = max(measured_max, start_height[:take].max().item())
            count += take
    report = dict(seed=seed, checkpoint=checkpoint.name, episodes=count, hits=hits, passes=passes,
                  pass_rate=passes / count,
                  configured_height_m=SAMPLE_Z, observed_height_m=[measured_min, measured_max])
    checkpoint.with_name("evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-episodes", type=int, default=4096,
                        help="Replay the saved policy on this many freshly seeded native starts")
    parser.add_argument("--warmup-updates", type=int, default=3600)
    parser.add_argument("--reference-flight", type=float, default=0.70, help="Flight time in the kinematic PPO reward")
    parser.add_argument("--contact-window", type=float, default=0.,
                        help="Width in seconds of the warmup interval for rewarding paddle position and velocity")
    parser.add_argument("--native-updates", type=int, default=420)
    parser.add_argument("--aim-reward", choices=("entrance",), default="entrance")
    parser.add_argument("--native-method", choices=("shot-model",), default="shot-model")
    parser.add_argument("--fine-after", type=int, default=-1,
                        help="Start fine spatial PPO after this many native shot-model updates")
    parser.add_argument("--align-contact", action=argparse.BooleanOptionalAction, default=True,
                        help="Let the shot actor learn a clock offset and reward consistent first contact timing")
    parser.add_argument("--outcome-after", nargs="*", type=int,
                        help="Start a landing-error-conditioned PPO head after each listed update (empty disables)")
    parser.add_argument("--outcome-samples", type=int, default=49152,
                        help="Number of frozen native shots used to fit each landing predictor")
    parser.add_argument("--prioritize-misses", action=argparse.BooleanOptionalAction, default=None,
                        help="During outcome refinement, sample more starts where the frozen predictor expects a miss")
    parser.add_argument("--stop-pass-rate", type=float, default=0.85,
                        help="Finish native updates when two fresh full-range batches meet this rate (0 disables)")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.outcome_after is None:
        args.outcome_after = []
        if args.fine_after < 0:
            args.outcome_after = [update for update in (120, 220, 320) if update < args.native_updates]
    if args.prioritize_misses is None:
        args.prioritize_misses = bool(args.outcome_after)
    if not 0. <= args.stop_pass_rate <= 1.:
        parser.error("--stop-pass-rate must be between 0 and 1")
    if args.num_envs < 32 or args.num_envs % 32 or min(args.warmup_updates, args.native_updates) < 1:
        parser.error("Use a positive number of updates and a positive multiple of 32 environments")
    if args.prioritize_misses and not args.outcome_after:
        parser.error("--prioritize-misses uses an outcome refinement stage")
    if args.outcome_after:
        if args.outcome_samples < 128:
            parser.error("Outcome refinement needs at least 128 outcome samples")
        if sorted(set(args.outcome_after)) != args.outcome_after or not all(
            args.fine_after < v < args.native_updates and v >= 0 for v in args.outcome_after
        ):
            parser.error("Outcome stages must increase, follow any fine stage, and precede the final update")
    if args.reference_flight <= 0:
        parser.error("--reference-flight must be positive")
    if args.contact_window < 0:
        parser.error("--contact-window must be nonnegative")
    if args.evaluation_episodes < 1:
        parser.error("--evaluation-episodes must be positive")
    if args.fine_after < -1:
        parser.error("--fine-after must be -1 or a nonnegative update index")
    enter_run_directory(args.run_dir)
    log_dir = Path.cwd() / "neural_ppo"
    log_dir.mkdir(exist_ok=False)
    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import torch

        from .env_cfg import StrikeBallEnvCfg
        from .ppo_network import StrikeActor
        from .ppo_shot import train_shots
        from .ppo_warmup import warmup
        from .scene import SAMPLE_X, SAMPLE_Y, SAMPLE_Z

        cfg = StrikeBallEnvCfg()
        cfg.seed, cfg.sim.device, cfg.scene.num_envs = args.seed, args.device, args.num_envs
        env = gym.make(TASK_ID, cfg=cfg)
        try:
            env.reset()
            torch.manual_seed(args.seed)
            torch.set_num_threads(3)
            actor = StrikeActor()
            (log_dir / "config.json").write_text(json.dumps({
                "seed": args.seed, "num_envs": args.num_envs, "warmup_updates": args.warmup_updates,
                "native_updates": args.native_updates, "device": args.device,
                "aim_reward": args.aim_reward,
                "native_method": args.native_method,
                "reference_flight": args.reference_flight,
                "contact_window": args.contact_window,
                "sample_x_m": SAMPLE_X, "sample_y_m": SAMPLE_Y, "sample_z_m": SAMPLE_Z,
                "fine_after": args.fine_after,
                "align_contact": args.align_contact,
                "outcome_after": args.outcome_after, "outcome_samples": args.outcome_samples,
                "prioritize_misses": args.prioritize_misses,
                "stop_pass_rate": args.stop_pass_rate,
                "evaluation_seed": args.seed + 10000, "evaluation_episodes": args.evaluation_episodes,
            }, indent=2) + "\n")
            warmup(actor, env.unwrapped, args.warmup_updates, flight_time=args.reference_flight,
                   contact_window=args.contact_window)
            train_shots(env, actor.to(env.unwrapped.device), args, log_dir)
            evaluate_saved_network(env, log_dir / "network.pt", args.seed + 10000, args.evaluation_episodes)
        finally:
            env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
