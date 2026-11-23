"""Train a randomly initialized six-joint strike policy with clipped PPO.

From the repository root, run

    conda run --no-capture-output -n isaac-sim python -m AgenticSimulator.tasks.strike_ball.train_network_ppo --headless --run-dir "AgenticSimulator/runs/$(date +%Y%m%d%H%M)_strike_ppo"

The default recipe verifies a native pass-rate target on two batches before stopping
early. It reloads neural_ppo/network.pt and writes the independently seeded replay
counts to neural_ppo/evaluation.json.
"""

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

from . import TASK_ID, enter_run_directory


def train(env, actor, args, log_dir):
    import torch
    from torch import nn

    from .scene import SAMPLE_X, SAMPLE_Z

    native = env.unwrapped
    n, steps, device = native.num_envs, native.max_episode_length, native.device
    group = 8
    half_range = [(high - low) / 2 for low, high in (SAMPLE_X, SAMPLE_Z)]
    joint_scale = native.cfg.actions.arm.scale
    seconds = torch.arange(steps, device=device).view(1, steps, 1) * native.step_dt
    critic = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 1)).to(device)
    actor_optimizer = torch.optim.Adam([
        {"params": actor.net.parameters(), "lr": 2e-5},
        {"params": actor.refinement.parameters(), "lr": 2e-4},
    ])
    if actor.feedback is not None:
        actor_optimizer.add_param_group({"params": actor.feedback.parameters(), "lr": 2e-4})
    value_optimizer = torch.optim.Adam(critic.parameters(), lr=0.002)
    correlation = 0.97
    innovation = math.sqrt(1. - correlation**2)

    def trajectory(context):
        return actor(context[:, None, :].expand(-1, steps, -1), seconds.expand(len(context), -1, -1))

    def save(name, update):
        torch.save({
            "format": "strike-ball-ppo-network-v1", "actor": actor.state_dict(),
            "spatial_refinement": actor.refinement is not None,
            "feedback": actor.feedback is not None,
            "episode_seconds": steps * native.step_dt, "joint_action_scale": joint_scale,
            "ball_half_range": half_range, "seed": args.seed, "native_updates": update,
        }, log_dir / name)

    save("warmup.pt", 0)
    with (log_dir / "training.jsonl").open("x") as log:
        for iteration in range(args.native_updates):
            with torch.no_grad():
                # Sibling shots share a sampled ball position. They explore independently.
                initial = native.held_state.clone()
                local = (initial[:, :3] - native.scene.env_origins)[::group].repeat_interleave(group, dim=0)
                initial[:, :3] = native.scene.env_origins + local
                native.ball.write_root_state_to_sim(initial)
                native.held_state.copy_(initial)
                native.previous_pos.copy_(initial[:, :3])
                context = (initial[:, :3] - native.hole)[:, [0, 2]] / initial.new_tensor(half_range)
                old_mean = trajectory(context)
                sigma = max(0.00035, 0.035 * math.exp(-iteration / 80))
                white = torch.randn_like(old_mean)
                noise = white.clone()
                for step in range(1, steps):
                    noise[:, step] = correlation * noise[:, step - 1] + innovation * white[:, step]
                observations = None
                if actor.feedback is not None:
                    obs = native.observation_manager.compute()["policy"]
                    observations = obs.new_empty((n, steps, obs.shape[1]))
                error = initial.new_full((n,), 1.5)
                slope = torch.ones_like(error)
                closest = torch.full_like(error, 1.5)
                for step in range(steps):
                    if observations is not None:
                        observations[:, step] = obs
                        old_mean[:, step] += actor.feedback_offset(obs)
                    action = (old_mean[:, step] + sigma * noise[:, step]) / joint_scale
                    next_obs, _, _, _, extras = env.step(action)
                    if observations is not None:
                        obs = next_obs["policy"]
                    if step + 1 < steps:
                        closest = torch.minimum(closest, native.best_distance)
                    measured = (native.launch_entrance_error_this_step if args.aim_reward == "entrance"
                                else native.launch_aim_error_this_step)
                    error = torch.where(measured.isfinite(), measured, error)
                    slope = torch.where(measured.isfinite(), native.launch_descent_slope_this_step, slope)
                hit, passed = extras["shot"]["hit"], extras["shot"]["passed"]
                if args.aim_reward == "entrance":
                    shallow = ((slope - 0.32) / 0.10).clamp_min(0.)
                    reward = -torch.log1p((error / 0.003).square()) - 1.5 * torch.log1p(shallow.square())
                    reward += 2.5 * passed.float()
                else:
                    reward = -torch.log1p((error / 0.004).square()) + 2. * passed.float()
                reward += torch.where(hit, 0., -2. - 4. * closest)
                grouped = reward.reshape(-1, group)
                peers = ((grouped.sum(1, keepdim=True) - grouped) / (group - 1)).flatten()
                advantage = reward - 0.98 * peers - 0.02 * critic(context).flatten()
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-7)

            def whitened_shift(ids):
                mean = trajectory(context[ids])
                if observations is not None:
                    mean = mean + actor.feedback_offset(observations[ids])
                shift = (mean - old_mean[ids]) / sigma
                increments = (shift[:, 1:] - correlation * shift[:, :-1]) / innovation
                return torch.cat((shift[:, :1], increments), dim=1)

            for _ in range(4):
                for ids in torch.randperm(n, device=device).chunk(4):
                    shift = whitened_shift(ids)
                    # This is the joint log likelihood ratio of the Gaussian action sequence.
                    log_ratio = (white[ids] * shift - 0.5 * shift.square()).sum(dim=(1, 2))
                    ratio = log_ratio.clamp(-20., 20.).exp()
                    gain = torch.minimum(ratio * advantage[ids], ratio.clamp(0.8, 1.2) * advantage[ids])
                    actor_optimizer.zero_grad()
                    (-gain.mean()).backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 1.)
                    actor_optimizer.step()
                with torch.no_grad():
                    kl = 0.5 * whitened_shift(slice(None)).square().sum(dim=(1, 2)).mean().item()
                if kl > 0.04:
                    break
            params = actor_optimizer.param_groups[0]
            if kl > 0.04:
                params["lr"] = max(1e-7, 0.65 * params["lr"])
            elif kl < 0.012:
                params["lr"] = min(0.0003, 1.15 * params["lr"])
            for extra_group in actor_optimizer.param_groups[1:]:
                extra_group["lr"] = 10 * params["lr"]
            value_optimizer.zero_grad()
            value_loss = (critic(context).flatten() - reward).square().mean()
            value_loss.backward()
            value_optimizer.step()
            count = iteration + 1
            low = context[:, 1] < 0.
            record = json.dumps({
                "update": count, "episodes": n, "hits": int(hit.sum()), "passes": int(passed.sum()),
                "median_predicted_miss_mm": 1000. * error.median().item(), "noise_std_rad": sigma, "kl": kl,
                "median_descent_slope": slope.median().item(),
                "below_board": int(low.sum()), "below_board_passes": int((low & passed).sum()),
                "mean_return": reward.mean().item(), "value_loss": value_loss.item(),
            })
            print(record, file=log, flush=True)
            print(record, flush=True)
            if count % 25 == 0:
                save(f"network_{count}.pt", count)
        save("network.pt", args.native_updates)


def evaluate_saved_network(env, checkpoint, seed, episodes):
    import torch
    from .ppo_network import FrozenStrikeNetwork
    from .scene import HOLE_CENTER, SAMPLE_Z

    policy = FrozenStrikeNetwork(torch.load(checkpoint, map_location="cpu", weights_only=True), env.unwrapped.device)
    obs = env.reset(seed=seed)[0]["policy"]
    count = hits = passes = 0
    low_count = low_passes = 0
    measured_min, measured_max = float("inf"), float("-inf")
    with torch.inference_mode():
        while count < episodes:
            low = obs[:, 26] > 0
            start_height = HOLE_CENTER[2] - obs[:, 26]
            for _ in range(env.unwrapped.max_episode_length):
                obs, _, _, _, extras = env.step(policy(obs))
                obs = obs["policy"]
            shot = extras["shot"]
            take = min(len(low), episodes - count)
            passed = shot["passed"][:take]
            hits += int(shot["hit"][:take].sum())
            passes += int(passed.sum())
            low_count += int(low[:take].sum())
            low_passes += int((passed & low[:take]).sum())
            measured_min = min(measured_min, start_height[:take].min().item())
            measured_max = max(measured_max, start_height[:take].max().item())
            count += take
    report = dict(seed=seed, checkpoint=checkpoint.name, episodes=count, hits=hits, passes=passes,
                  pass_rate=passes / count, below_board_episodes=low_count, below_board_passes=low_passes,
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
    parser.add_argument("--aim-reward", choices=("entrance", "midplane"), default="entrance")
    parser.add_argument("--native-method", choices=("joint-noise", "shot-model", "launch-model"), default="shot-model")
    parser.add_argument("--fine-after", type=int, default=-1,
                        help="Start fine spatial PPO after this many native shot-model updates")
    parser.add_argument("--align-contact", action=argparse.BooleanOptionalAction, default=None,
                        help="Let the shot actor learn a clock offset and reward consistent first contact timing")
    parser.add_argument("--outcome-after", nargs="*", type=int,
                        help="Start a landing-error-conditioned PPO head after each listed update (empty disables)")
    parser.add_argument("--outcome-samples", type=int, default=49152,
                        help="Number of frozen native shots used to fit each landing predictor")
    parser.add_argument("--prioritize-misses", action=argparse.BooleanOptionalAction, default=None,
                        help="During outcome refinement, sample more starts where the frozen predictor expects a miss")
    parser.add_argument("--stop-pass-rate", type=float,
                        help="Finish native updates when two fresh full-range batches meet this rate (0 disables)")
    parser.add_argument("--feedback", action="store_true", help="Use the current 35-value observation at each native step")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    shot_model = args.native_method == "shot-model"
    if args.align_contact is None:
        args.align_contact = shot_model
    if args.outcome_after is None:
        args.outcome_after = []
        if shot_model and args.fine_after < 0:
            args.outcome_after = [update for update in (120, 220, 320) if update < args.native_updates]
    if args.prioritize_misses is None:
        args.prioritize_misses = bool(args.outcome_after)
    if args.stop_pass_rate is None:
        args.stop_pass_rate = 0.97 if shot_model else 0.
    if not 0. <= args.stop_pass_rate <= 1. or (args.stop_pass_rate and args.native_method == "joint-noise"):
        parser.error("--stop-pass-rate must be between 0 and 1 and uses a shot-level method")
    if args.num_envs < 8 or args.num_envs % 8 or min(args.warmup_updates, args.native_updates) < 1:
        parser.error("Use a positive number of updates and a positive multiple of eight environments")
    if args.native_method != "joint-noise" and (args.num_envs % 32 or args.aim_reward != "entrance"):
        parser.error("Shot-model PPO uses the entrance reward and groups of 32 native environments")
    if args.feedback and args.native_method != "joint-noise":
        parser.error("--feedback uses joint-noise PPO")
    if args.align_contact and args.native_method != "shot-model":
        parser.error("--align-contact uses shot-model PPO")
    if args.prioritize_misses and not args.outcome_after:
        parser.error("--prioritize-misses uses an outcome refinement stage")
    if args.outcome_after:
        if args.native_method != "shot-model" or args.outcome_samples < 128:
            parser.error("Outcome refinement needs shot-model PPO and at least 128 outcome samples")
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
    if args.fine_after < -1 or (args.fine_after >= 0 and args.native_method == "joint-noise"):
        parser.error("--fine-after is available for the two shot-level methods")
    enter_run_directory(args.run_dir)
    log_dir = Path.cwd() / "neural_ppo"
    log_dir.mkdir(exist_ok=False)
    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import torch

        from .env_cfg import StrikeBallEnvCfg
        from .ppo_network import StrikeActor
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
                "feedback": args.feedback,
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
            if args.native_method != "joint-noise":
                from .ppo_shot import train_shots

                train_shots(env, actor.to(env.unwrapped.device), args, log_dir)
            else:
                actor.enable_refinement()
                if args.feedback:
                    actor.enable_feedback()
                train(env, actor.to(env.unwrapped.device), args, log_dir)
            evaluate_saved_network(env, log_dir / "network.pt", args.seed + 10000, args.evaluation_episodes)
        finally:
            env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
