"""Episodic PPO on native strikes and short rollouts in fitted local forward models."""

from collections import deque
import json
import math

import torch
from torch import nn

from .ppo_network import FrozenStrikeNetwork, JointAdjustment
from .scene import SAMPLE_X, SAMPLE_Z


def aim_return(outcome):
    # First three coordinates are the projected entrance miss in metres. The
    # fourth is the horizontal/vertical speed ratio, scaled for model fitting.
    error2 = outcome[..., :3].square().sum(-1)
    shallow = ((outcome[..., 3] / 0.15 - 0.30) / 0.10).clamp_min(0.)
    return -torch.log1p(error2 / 0.003**2) - 1.5 * torch.log1p(shallow.square())


class ShotPPO:
    def __init__(self, device, action_dim, *, actor=None):
        self.actor = JointAdjustment(action_dim).to(device) if actor is None else actor
        self.critic = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 1)).to(device)
        self.select_actor_parameters()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.002)

    def select_actor_parameters(self):
        """Use the same trainable branch when entering a stage or restoring it."""
        if self.actor.outcome_adjustments:
            branch = self.actor.outcome_adjustments[-1].policy
            lr = 0.0004
        elif self.actor.fine is not None:
            branch = self.actor.fine
            lr = 0.0002
        else:
            branch = self.actor.net
            lr = 0.001
        self.actor.requires_grad_(False)
        branch.requires_grad_(True)
        self.actor_optimizer = torch.optim.Adam(branch.parameters(), lr=lr)

    def refine(self):
        self.actor.enable_fine()
        self.select_actor_parameters()

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
                nn.utils.clip_grad_norm_(self.actor_optimizer.param_groups[0]["params"], 1.)
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


def train_shots(env, base, args, log_dir, *, checkpoint=None):
    """Run args.native_updates additional updates in the supplied, already reset environment.

    Standalone training supplies its warmup actor. For continuation, pass base=None
    and a shot checkpoint to restore the full swing and the active PPO branch.
    Stage boundaries use cumulative updates. Save intervals use updates in this round.
    """
    native = env.unwrapped
    use_env_rewards = bool(native.reward_manager.active_terms)
    score_contact = args.align_contact and not use_env_rewards
    device, n, steps = native.device, native.num_envs, native.max_episode_length
    group = 32
    if n < group or n % group or args.native_updates < 1:
        raise ValueError("Shot PPO needs a positive multiple of 32 environments and at least one update")
    start_update = 0
    last_outcome_stage = None
    training_state = {}
    if checkpoint is None:
        scales = torch.tensor([0.035] * 6 + [0.10] * 6 + ([0.035] if args.align_contact else []), device=device)
        learner = ShotPPO(device, len(scales))
        half_range = [(high - low) / 2 for low, high in (SAMPLE_X, SAMPLE_Z)]
        joint_scale = native.cfg.actions.arm.scale
        episode_seconds = steps * native.step_dt
        save_interval = 20
    else:
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if bundle.get("format") != "strike-ball-ppo-network-v1" or "adjustment" not in bundle:
            raise ValueError("Shot PPO continuation requires a combined strike checkpoint with joint adjustments")
        policy = FrozenStrikeNetwork(bundle, device)
        if policy.actor.feedback is not None or policy.adjustment_basis != "joint-gain":
            raise ValueError("Expected the open-loop joint-gain policy produced by train_shots")
        base = policy.actor.requires_grad_(False)
        scales = policy.adjustment_scale
        half_range = bundle["ball_half_range"]
        joint_scale = policy.joint_scale
        episode_seconds = policy.episode_seconds
        learner = ShotPPO(device, len(scales), actor=policy.adjustment)
        start_update = bundle["native_updates"]
        save_interval = getattr(args, "save_interval", 100)
        if save_interval < 1:
            raise ValueError("save_interval must be positive")
        if "training_state" in bundle:
            training_state = bundle["training_state"]
            learner.critic.load_state_dict(training_state["critic"])
            learner.actor_optimizer.load_state_dict(training_state["actor_optimizer"])
            learner.critic_optimizer.load_state_dict(training_state["critic_optimizer"])
            last_outcome_stage = training_state["last_outcome_stage"]
        else:
            # Original P0 exports have the policy and update count only. Recover the
            # exploration schedule from the recipe when its past stages match P0.
            num_heads = len(learner.actor.outcome_adjustments)
            if num_heads:
                past_stages = [stage for stage in args.outcome_after if stage < start_update]
                if len(past_stages) == num_heads:
                    last_outcome_stage = past_stages[-1]
                else:
                    last_outcome_stage = start_update
                    print(f"Restarting outcome exploration schedule at update {start_update}", flush=True)
            print(f"Restored policy at update {start_update}; initializing critic and PPO optimizers", flush=True)
    base.eval()
    stop_pass_rate = args.stop_pass_rate if checkpoint is None else 0.
    target_substep = training_state.get("target_substep")
    target_distance = training_state.get("target_distance")
    native.record_contact_geometry = args.align_contact

    def score(outcome):
        result = aim_return(outcome)
        if args.align_contact:
            result = result - 1.6 * torch.log1p((outcome[..., 4] / 0.006).square())
            result = result - 0.8 * torch.log1p((outcome[..., 5] / 0.0015).square())
        return result

    seconds = torch.arange(steps, device=device)[None, :, None] * (episode_seconds / steps)
    # These local dynamics models describe only the current source physics.
    history = deque(maxlen=48)

    def save(name, update):
        path = log_dir / name
        temporary = path.with_suffix(".pt.tmp")
        torch.save({
            "format": "strike-ball-ppo-network-v1", "actor": base.state_dict(),
            "spatial_refinement": base.refinement is not None,
            "adjustment": learner.actor.state_dict(), "adjustment_scale": scales.tolist(),
            "adjustment_basis": "joint-gain",
            "fine_adjustment": learner.actor.fine is not None,
            "outcome_adjustments": len(learner.actor.outcome_adjustments),
            "joint_action_scale": joint_scale, "ball_half_range": half_range,
            "episode_seconds": episode_seconds, "seed": args.seed, "native_updates": update,
            "round_updates": update - start_update,
            "training_state": {
                "critic": learner.critic.state_dict(),
                "actor_optimizer": learner.actor_optimizer.state_dict(),
                "critic_optimizer": learner.critic_optimizer.state_dict(),
                "last_outcome_stage": last_outcome_stage,
                "target_substep": target_substep, "target_distance": target_distance,
            },
        }, temporary)
        temporary.replace(path)

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
        if scales.numel() == 13:
            clock = (clock + controls[:, None, 12:]).clamp(0., episode_seconds)
        joints = base(start[:, None, :].expand(-1, steps, -1), clock)
        commands = (joints + gain[:, None, :] * joints + bias[:, None, :]) / joint_scale
        measured = state.new_ones((n, 4))
        released = torch.zeros(n, dtype=torch.bool, device=device)
        episode_reward = state.new_zeros((n,)) if use_env_rewards else None
        for step in range(steps):
            _, step_reward, _, _, extras = env.step(commands[:, step])
            if use_env_rewards:
                episode_reward.add_(step_reward)
            offset = native.launch_entrance_offset_this_step
            event = offset.isfinite().all(dim=1)
            measured[:, :3] = torch.where(event[:, None], offset, measured[:, :3])
            measured[:, 3] = torch.where(event, 0.15 * native.launch_descent_slope_this_step, measured[:, 3])
            released |= event
        shot = extras["shot"]
        if measure_outcomes:
            return (start, measured[:, :2], shot["contact_substep"].float(), shot["contact_plane_distance"],
                    shot["passed"], released)
        if score_contact:
            contact = shot["contact_substep"]
            distance = shot["contact_plane_distance"]
            if target_substep is None:
                target_substep = contact[shot["hit"]].median().item()
                target_distance = distance[shot["hit"]].median().item()
            timing = native.physics_dt * (contact - target_substep)
            measured = torch.cat((measured, timing[:, None], (distance - target_distance)[:, None]), dim=1)
        if not use_env_rewards:
            episode_reward = score(measured) + 3.5 * shot["passed"].float() - 2. * (~released)
        return start, mean, noise, action, measured, released, episode_reward, shot["hit"], shot["passed"]

    if checkpoint is None:
        save("warmup.pt", 0)
    with (log_dir / "training.jsonl").open("x") as log:
        for round_update in range(1, args.native_updates + 1):
            it = start_update + round_update - 1
            forecast_metrics = {}
            if it == args.fine_after and learner.actor.fine is None and not learner.actor.outcome_adjustments:
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
                learner.select_actor_parameters()
                history.clear()
                last_outcome_stage = it
            if last_outcome_stage is not None:
                sigma = max(0.005, 0.045 * math.exp(-(it - last_outcome_stage) / 30))
            with torch.no_grad():
                start, old, noise, action, measured, released, reward, hit, passed = rollout(group, sigma)
            kl = learner.update(start, action, old, reward, sigma, group)
            count = it + 1
            miss, slope = measured[:, :3].norm(dim=1), measured[:, 3] / 0.15
            record = {
                "update": count, "round_update": round_update, "episodes": n,
                "hits": int(hit.sum()), "passes": int(passed.sum()),
                "mean_shot_reward": reward.mean().item(),
                "median_predicted_miss_mm": 1000 * miss.median().item(),
                "median_descent_slope": slope.median().item(), "kl": kl, "noise_std": sigma,
            }
            # Local models predict shot outcomes; arbitrary environment rewards require a real rollout.
            if not use_env_rewards:
                with torch.no_grad():
                    fit, usable, residual = fit_forward_models(noise, measured, released, group)
                    if usable.any():
                        position = start[::group][usable]
                        history.append((position, old[::group][usable], torch.full_like(position[:, :1], sigma), fit[usable]))
                if history:
                    model_updates(learner, history, sigma, start[::group], old[::group], score)
                record.update(median_model_error_mm=1000 * residual.median().item(), usable_models=int(usable.sum()))
            if forecast_metrics:
                record["frozen_outcome_prediction"] = forecast_metrics
            if score_contact:
                record.update(target_contact_substep=target_substep,
                              median_contact_error_ms=1000. * measured[hit, 4].abs().median().item(),
                              target_contact_plane_mm=1000. * target_distance,
                              median_contact_plane_error_mm=1000. * measured[hit, 5].abs().median().item())
            target_met = False
            final_update = round_update == args.native_updates
            if round_update % save_interval == 0 or final_update:
                save(f"network_{count}.pt", count)
            if count % 20 == 0 or final_update:
                with torch.no_grad():
                    validation = rollout(1, 0.)
                    cleared = validation[-1]
                    record.update(validation_episodes=n, validation_passes=int(cleared.sum()))
                    if stop_pass_rate and record["validation_passes"] / n >= stop_pass_rate:
                        confirmation = rollout(1, 0.)[-1]
                        record.update(confirmation_episodes=n, confirmation_passes=int(confirmation.sum()))
                        target_met = record["confirmation_passes"] / n >= stop_pass_rate
                        record["stop_target_met"] = target_met
            line = json.dumps(record)
            print(line, file=log, flush=True)
            print(line, flush=True)
            if target_met:
                break
        save("network.pt", count)
