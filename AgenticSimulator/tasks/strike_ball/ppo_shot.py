"""Episodic PPO on native strikes and short rollouts in fitted local forward models."""

from collections import deque
import json
import math

import torch
from torch import nn

from .ppo_network import JointAdjustment
from .scene import SAMPLE_X, SAMPLE_Z


def aim_return(outcome):
    # First three coordinates are the projected entrance miss in metres. The
    # fourth is the horizontal/vertical speed ratio, scaled for model fitting.
    error2 = outcome[..., :3].square().sum(-1)
    shallow = ((outcome[..., 3] / 0.15 - 0.30) / 0.10).clamp_min(0.)
    return -torch.log1p(error2 / 0.003**2) - 1.5 * torch.log1p(shallow.square())


class ShotPPO:
    def __init__(self, device, action_dim):
        self.actor = JointAdjustment(action_dim).to(device)
        self.critic = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 1)).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.002)

    def refine(self):
        self.actor.net.requires_grad_(False)
        self.actor.enable_fine()
        self.actor_optimizer = torch.optim.Adam(self.actor.fine.parameters(), lr=0.0002)

    def update(self, start, action, old, reward, sigma, group):
        with torch.no_grad():
            scores = reward.view(-1, group)
            peers = ((scores.sum(1, keepdim=True) - scores) / (group - 1)).flatten()
            advantage = reward - 0.98 * peers - 0.02 * self.critic(start).flatten()
            if self.actor.fine is not None or len(self.actor.outcome_adjustments):
                # Estimate local variation from the other actions at the same start.
                second = ((scores.square().sum(1, keepdim=True) - scores.square()) / (group - 1)).flatten()
                spread = (second - peers.square()).clamp_min(0.03**2).sqrt()
                advantage = ((reward - peers) / spread).clamp(-5., 5.)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-7)
            old_log = -0.5 * ((action - old) / sigma).square().sum(1)
        for _ in range(4):
            for ids in torch.randperm(len(start), device=start.device).chunk(4):
                log_prob = -0.5 * ((action[ids] - self.actor(start[ids])) / sigma).square().sum(1)
                ratio = (log_prob - old_log[ids]).clamp(-20., 20.).exp()
                gain = torch.minimum(ratio * advantage[ids], ratio.clamp(0.8, 1.2) * advantage[ids])
                self.actor_optimizer.zero_grad()
                (-gain.mean()).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
                self.actor_optimizer.step()
            with torch.no_grad():
                kl = 0.5 * ((self.actor(start) - old) / sigma).square().sum(1).mean().item()
            if kl > 0.035:
                break
        params = self.actor_optimizer.param_groups[0]
        if kl > 0.045:
            params["lr"] = max(5e-7, 0.75 * params["lr"])
        elif kl < 0.011:
            params["lr"] = min(0.004, 1.15 * params["lr"])
        self.critic_optimizer.zero_grad()
        value_loss = (self.critic(start).flatten() - reward).square().mean()
        value_loss.backward()
        self.critic_optimizer.step()
        return kl


def fit_forward_models(noise, measured, released, group):
    """Predict physical shot outcomes from sampled actions at each training position."""
    inputs = noise.view(-1, group, noise.shape[-1])
    design = torch.cat((torch.ones_like(inputs[:, :, :1]), inputs), dim=2)
    outcomes = measured.view(-1, group, measured.shape[-1]).clamp(-1.5, 1.5)
    valid = released.view(-1, group, 1)
    eye = torch.eye(design.shape[-1], device=noise.device)

    def fit(count):
        x = design[:, :count] * valid[:, :count]
        return torch.linalg.solve(x.transpose(1, 2) @ x + 0.1 * eye, x.transpose(1, 2) @ outcomes[:, :count])

    train_count = 3 * group // 4
    heldout = outcomes[:, train_count:]
    predicted = design[:, train_count:] @ fit(train_count)
    residual = (predicted - heldout).norm(dim=2).median(1).values
    miss = heldout[:, :, :3].norm(dim=2).median(1).values
    usable = (valid.float().mean(1).squeeze(-1) > 0.88) & (residual < 0.003 + 0.8 * miss)
    return fit(group), usable, residual


def model_updates(learner, history, sigma, current_start, current_mean, score):
    # The fitted models describe forward dynamics near the sampled policy.
    # Draw fresh on-policy actions for every model update. The fit never supplies
    # an action target, and none of the models is exported for inference.
    start, anchor, spread, coefficients = (torch.cat(column) for column in zip(*history))
    group = 16
    for _ in range(14):
        with torch.no_grad():
            current = learner.actor(start)
            within_fit = ((current - anchor) / spread).square().sum(1) < 6.25
            candidates = within_fit.nonzero().flatten()
            if not len(candidates):
                return
            actor_change = ((learner.actor(current_start) - current_mean) / sigma).square().sum(1).mean()
            if actor_change > 2.:
                return
            ids = candidates[torch.randint(len(candidates), (min(256, len(candidates)),), device=start.device)]
            contexts = start[ids].repeat_interleave(group, dim=0)
            old = current[ids].repeat_interleave(group, dim=0)
            draws = old + sigma * torch.randn_like(old)
            shift = (draws.view(-1, group, draws.shape[-1]) - anchor[ids, None, :]) / spread[ids, None, :]
            design = torch.cat((torch.ones_like(shift[:, :, :1]), shift), dim=2)
            outcome = design @ coefficients[ids]
            reward = score(outcome).flatten()
        learner.update(contexts, draws, old, reward, sigma, group)


def train_shots(env, base, args, log_dir):
    native = env.unwrapped
    device, n, steps = native.device, native.num_envs, native.max_episode_length
    group = 32
    base.eval()
    scales = torch.tensor([0.035] * 6 + [0.10] * 6 + ([0.035] if args.align_contact else []), device=device)
    learner = ShotPPO(device, len(scales))
    target_substep = None
    target_distance = None
    native.record_contact_geometry = args.align_contact

    def score(outcome):
        result = aim_return(outcome)
        if args.align_contact:
            result = result - 1.6 * torch.log1p((outcome[..., 4] / 0.006).square())
            result = result - 0.8 * torch.log1p((outcome[..., 5] / 0.0015).square())
        return result

    half_range = [(high - low) / 2 for low, high in (SAMPLE_X, SAMPLE_Z)]
    joint_scale = native.cfg.actions.arm.scale
    seconds = torch.arange(steps, device=device)[None, :, None] * native.step_dt
    history = deque(maxlen=48)

    def save(name, update):
        torch.save({
            "format": "strike-ball-ppo-network-v1", "actor": base.state_dict(),
            "adjustment": learner.actor.state_dict(), "adjustment_scale": scales.tolist(),
            "adjustment_basis": "joint-gain",
            "fine_adjustment": learner.actor.fine is not None,
            "outcome_adjustments": len(learner.actor.outcome_adjustments),
            "joint_action_scale": joint_scale, "ball_half_range": half_range,
            "episode_seconds": steps * native.step_dt, "seed": args.seed, "native_updates": update,
        }, log_dir / name)

    def rollout(group_size, sigma, measure_outcomes=False):
        nonlocal target_substep, target_distance
        state = native.held_state.clone()
        positions = state[:, :3] - native.scene.env_origins
        selected = positions[::group_size]
        if sigma and args.prioritize_misses and len(learner.actor.outcome_adjustments):
            contexts = (state[:, :3] - native.hole)[:, [0, 2]] / state.new_tensor(half_range)
            predicted = learner.actor.outcome_adjustments[-1].predict(contexts)[0].norm(dim=1)
            difficult = predicted >= predicted.quantile(0.75)
            weights = 0.5 + 2. * difficult.float()
            selected = positions[torch.multinomial(weights, len(selected), replacement=False)]
        local = selected.repeat_interleave(group_size, dim=0)
        state[:, :3] = native.scene.env_origins + local
        native.ball.write_root_state_to_sim(state)
        native.held_state.copy_(state)
        native.previous_pos.copy_(state[:, :3])
        start = (state[:, :3] - native.hole)[:, [0, 2]] / state.new_tensor(half_range)
        mean = learner.actor(start)
        noise = torch.randn_like(mean) if sigma else torch.zeros_like(mean)
        action = mean + sigma * noise
        controls = action * scales
        bias, gain = controls[:, :12].split(6, dim=1)
        clock = seconds.expand(n, -1, -1)
        if args.align_contact:
            clock = (clock + controls[:, None, 12:]).clamp(0., steps * native.step_dt)
        joints = base(start[:, None, :].expand(-1, steps, -1), clock)
        commands = (joints + gain[:, None, :] * joints + bias[:, None, :]) / joint_scale
        measured = state.new_ones((n, 4))
        released = torch.zeros(n, dtype=torch.bool, device=device)
        for step in range(steps):
            _, _, _, _, extras = env.step(commands[:, step])
            offset = native.launch_entrance_offset_this_step
            event = offset.isfinite().all(dim=1)
            measured[:, :3] = torch.where(event[:, None], offset, measured[:, :3])
            measured[:, 3] = torch.where(event, 0.15 * native.launch_descent_slope_this_step, measured[:, 3])
            released |= event
        shot = extras["shot"]
        if measure_outcomes:
            return (start, measured[:, :2], shot["contact_substep"].float(), shot["contact_plane_distance"],
                    shot["passed"], released)
        if args.align_contact:
            contact = shot["contact_substep"]
            distance = shot["contact_plane_distance"]
            if target_substep is None:
                target_substep = contact[shot["hit"]].median().item()
                target_distance = distance[shot["hit"]].median().item()
            timing = native.physics_dt * (contact - target_substep)
            measured = torch.cat((measured, timing[:, None], (distance - target_distance)[:, None]), dim=1)
        return start, mean, noise, action, measured, released, shot["hit"], shot["passed"]

    save("warmup.pt", 0)
    last_outcome_stage = None
    with (log_dir / "training.jsonl").open("x") as log:
        for it in range(args.native_updates):
            forecast_metrics = {}
            if it == args.fine_after:
                learner.refine()
            sigma = max(0.008, 0.8 * math.exp(-it / 32))
            if it in args.outcome_after:
                native.record_contact_geometry = True
                with torch.no_grad():
                    _, duplicate_offset, _, _, _, duplicate_release = rollout(2, 0., measure_outcomes=True)
                    paired = duplicate_release[::2] & duplicate_release[1::2]
                    duplicate_error = (duplicate_offset[::2] - duplicate_offset[1::2])[paired].norm(dim=1)
                    samples = [rollout(1, 0., measure_outcomes=True) for _ in range(math.ceil(args.outcome_samples / n))]
                native.record_contact_geometry = args.align_contact
                learner.actor.requires_grad_(False)
                layer = learner.actor.add_outcome_adjustment()
                *data, released = (torch.cat(part) for part in zip(*samples))
                starts, offsets, contact, distance, passed = (part[released] for part in data)
                torch.save({
                    "start": starts.cpu(), "entrance_offset": offsets.cpu(),
                    "contact_substep": contact.cpu(), "contact_plane_distance": distance.cpu(), "passed": passed.cpu(),
                }, log_dir / f"frozen_shots_{it}.pt")
                forecast_metrics = layer.fit_outcomes(starts, offsets, contact, distance)
                if len(duplicate_error):
                    forecast_metrics["repeat_pairs"] = len(duplicate_error)
                    forecast_metrics["repeat_median_difference_mm"] = 1000. * duplicate_error.median().item()
                    forecast_metrics["repeat_p90_difference_mm"] = 1000. * duplicate_error.quantile(0.9).item()
                print(json.dumps(forecast_metrics), flush=True)
                learner.actor_optimizer = torch.optim.Adam(layer.policy.parameters(), lr=0.0004)
                history.clear()
                last_outcome_stage = it
            if last_outcome_stage is not None:
                sigma = max(0.005, 0.045 * math.exp(-(it - last_outcome_stage) / 30))
            with torch.no_grad():
                start, old, noise, action, measured, released, hit, passed = rollout(group, sigma)
                reward = score(measured) + 3.5 * passed.float() - 2. * (~released)
            kl = learner.update(start, action, old, reward, sigma, group)
            with torch.no_grad():
                fit, usable, residual = fit_forward_models(noise, measured, released, group)
                if usable.any():
                    position = start[::group][usable]
                    history.append((position, old[::group][usable], torch.full_like(position[:, :1], sigma), fit[usable]))
            if history:
                model_updates(learner, history, sigma, start[::group], old[::group], score)
            count = it + 1
            miss, slope = measured[:, :3].norm(dim=1), measured[:, 3] / 0.15
            record = {
                "update": count, "episodes": n, "hits": int(hit.sum()), "passes": int(passed.sum()),
                "mean_shot_reward": reward.mean().item(),
                "median_predicted_miss_mm": 1000 * miss.median().item(),
                "median_descent_slope": slope.median().item(), "kl": kl, "noise_std": sigma,
                "median_model_error_mm": 1000 * residual.median().item(), "usable_models": int(usable.sum()),
            }
            if forecast_metrics:
                record["frozen_outcome_prediction"] = forecast_metrics
            if args.align_contact:
                record.update(target_contact_substep=target_substep,
                              median_contact_error_ms=1000. * measured[hit, 4].abs().median().item(),
                              target_contact_plane_mm=1000. * target_distance,
                              median_contact_plane_error_mm=1000. * measured[hit, 5].abs().median().item())
            target_met = False
            if count % 20 == 0 or count == args.native_updates:
                save(f"network_{count}.pt", count)
                with torch.no_grad():
                    validation = rollout(1, 0.)
                    cleared = validation[-1]
                    record.update(validation_episodes=n, validation_passes=int(cleared.sum()))
                    if args.stop_pass_rate and record["validation_passes"] / n >= args.stop_pass_rate:
                        confirmation = rollout(1, 0.)[-1]
                        record.update(confirmation_episodes=n, confirmation_passes=int(confirmation.sum()))
                        target_met = record["confirmation_passes"] / n >= args.stop_pass_rate
                        record["stop_target_met"] = target_met
            line = json.dumps(record)
            print(line, file=log, flush=True)
            print(line, flush=True)
            if target_met:
                break
        save("network.pt", count)
