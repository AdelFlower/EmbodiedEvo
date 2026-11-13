"""Recurrent actor-critic with the same physical observation/action schema as BC."""

import math

import torch
from torch import nn
from torch.distributions import Bernoulli, Beta, Normal
from torch.nn import functional as F
from .rewards import rotate


class ActorCritic(nn.Module):
    def __init__(self, hidden_dim=256, recurrent_dim=128,
                 position_scale=0.01, rotation_scale=0.1, initial_std=0.3, grip_decimation=1,
                 absolute_targets=False):
        super().__init__()
        self.recurrent_dim = recurrent_dim
        self.grip_decimation = grip_decimation
        self.absolute_targets = absolute_targets
        self.register_buffer("obs_mean", torch.zeros(40))
        self.register_buffer("obs_std", torch.ones(40))
        self.register_buffer("pose_scale", torch.tensor(
            [position_scale] * 3 + [rotation_scale] * 3))
        self.encoder = nn.Sequential(nn.Linear(54, hidden_dim), nn.ELU())
        self.gru = nn.GRU(hidden_dim, recurrent_dim, batch_first=True)
        # 6 pose means, 1 grip logit, 2 Beta parameters for the force scalar.
        # The executed action remains eight-dimensional.
        self.actor = nn.Linear(recurrent_dim, 9)
        # Independent critic: value updates cannot directly move actor features.
        self.critic = nn.Sequential(nn.Linear(54, hidden_dim), nn.ELU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))
        self.log_std = nn.Parameter(torch.as_tensor(initial_std, dtype=torch.float32).expand(6).log().clone())
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        # Start mostly open, with small pose residuals. No BC weights are loaded.
        with torch.no_grad():
            self.actor.bias[6] = -2.0
            # Beta(4,4): initially centered on 10 N, with learnable exploration.
            self.actor.bias[7:] = math.log(math.expm1(3.0))

    def set_normalization(self, stats):
        # A fixed floor prevents off-demonstration deviations in almost constant
        # coordinates from being amplified by tiny expert standard deviations.
        floor = self.obs_std.new_tensor(
            [0.1] * 14 + [0.01] * 2 + [0.05] * 3 + [0.1] * 4
            + [0.05] * 3 + [0.1] * 4 + [0.1] * 3 + [0.05] * 3 + [0.1] * 4)
        self.obs_mean.copy_(stats["obs_mean"][:40])
        self.obs_std.copy_(torch.maximum(stats["obs_std"][:40].to(floor), floor))

    def initial_hidden(self, batch_size):
        return self.obs_mean.new_zeros(1, batch_size, self.recurrent_dim)

    def grip_decision_mask(self, learner_steps):
        return learner_steps.remainder(self.grip_decimation) == 0

    def features(self, observations):
        grip = observations[..., 16:19] + rotate(
            observations[..., 19:23], observations.new_tensor([0., 0., 0.1034]))
        relatives = torch.cat((observations[..., 23:26] - grip,
                               observations[..., 33:36] - observations[..., 23:26]), -1) / 0.2
        return torch.cat(((observations[..., :40] - self.obs_mean) / self.obs_std,
                          observations[..., 40:48], relatives), -1)

    def forward(self, observations, hidden=None, starts=None):
        """Raw [B,48] or [B,T,48]; starts marks resets BEFORE each observation."""
        single = observations.ndim == 2
        if single:
            observations = observations[:, None]
            if starts is not None:
                starts = starts[:, None]
        features = self.features(observations)
        x = self.encoder(features)
        if hidden is None:
            hidden = self.initial_hidden(x.shape[0])
        if starts is None:
            x, hidden = self.gru(x, hidden)
        else:
            # Run contiguous spans between resets, retaining sequence gradients.
            boundaries = starts.any(dim=0).nonzero().flatten().tolist()
            boundaries = sorted(set([0, *boundaries, x.shape[1]]))
            pieces = []
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                hidden = hidden * (~starts[:, left])[None, :, None]
                part, hidden = self.gru(x[:, left:right], hidden)
                pieces.append(part)
            x = torch.cat(pieces, dim=1)
        raw, value = self.actor(x), self.critic(features).squeeze(-1)
        if self.absolute_targets:
            raw = self.target_residual(raw, observations)
        if single:
            raw, value = raw[:, 0], value[:, 0]
        return raw, value, hidden

    def target_residual(self, raw, observations):
        """A learned base-frame goal, expressed as the executor's native residual.

        A constant head output now holds a fixed pose instead of integrating a
        constant displacement forever. Sampling/densities still use residuals.
        """
        target = raw[..., :3] * self.pose_scale[:3] + raw.new_tensor([0.4, 0.0, 0.25])
        vector = raw[..., 3:6] * self.pose_scale[3:]
        angle = vector.norm(dim=-1, keepdim=True)
        xyz = vector * (0.5 * torch.sinc(angle / (2 * math.pi)))
        w = torch.cos(angle / 2)
        # Exp(vector) composed with the fixed downward-facing reference quaternion.
        goal = torch.cat((-xyz[..., 0:1], w, xyz[..., 2:3], -xyz[..., 1:2]), -1)
        current = observations[..., 19:23]
        inverse = torch.cat((current[..., :1], -current[..., 1:]), -1)
        ew = goal[..., :1] * inverse[..., :1] - (goal[..., 1:] * inverse[..., 1:]).sum(-1, keepdim=True)
        ev = (goal[..., :1] * inverse[..., 1:] + inverse[..., :1] * goal[..., 1:]
              + torch.cross(goal[..., 1:], inverse[..., 1:], dim=-1))
        sign = torch.where(ew < 0, -1.0, 1.0)
        ew, ev = ew * sign, ev * sign
        norm = ev.norm(dim=-1, keepdim=True)
        factor = torch.where(norm > 1e-6, 2 * torch.atan2(norm, ew) / norm.clamp_min(1e-6), 2.0)
        return torch.cat(((target - observations[..., 16:19]) / self.pose_scale[:3],
                          ev * factor / self.pose_scale[3:], raw[..., 6:]), -1)

    def distributions(self, raw):
        concentration = F.softplus(raw[..., 7:9]) + 1.0
        return (Normal(raw[..., :6], self.log_std.clamp(-5, 2).exp()),
                Bernoulli(logits=raw[..., 6]),
                Beta(concentration[..., 0], concentration[..., 1]))

    def action_statistics(self, raw, actions, grip_decisions=None):
        pose, grip, force = self.distributions(raw)
        if grip_decisions is None:
            grip_decisions = torch.ones_like(actions[..., 6], dtype=torch.bool)
        # Density in physical coordinates, with the constant scaling Jacobian.
        logp = (pose.log_prob(actions[..., :6] / self.pose_scale).sum(-1)
                - self.pose_scale.log().sum() + grip_decisions * grip.log_prob(actions[..., 6]))
        # Force is conditional on closing. Opening has a deterministic -1 label,
        # not a force sample, so its force density and auxiliary gradient are zero.
        unit_force = ((actions[..., 7] + 1) / 2).clamp(1e-6, 1 - 1e-6)
        logp += actions[..., 6] * (force.log_prob(unit_force) - math.log(2.0))
        close_probability = torch.where(grip_decisions, grip.probs, actions[..., 6])
        entropy = (pose.entropy().sum(-1) + self.pose_scale.log().sum() + grip_decisions * grip.entropy()
                   + close_probability * (force.entropy() + math.log(2.0)))
        return logp, entropy

    @torch.no_grad()
    def act(self, observations, hidden=None, starts=None, deterministic=False, grip_decisions=None):
        raw, value, hidden = self(observations, hidden, starts)
        pose, grip, force = self.distributions(raw)
        delta = pose.mean if deterministic else pose.sample()
        closed = (raw[..., 6] >= 0).float() if deterministic else grip.sample()
        if grip_decisions is not None:
            # Between decisions this component has probability one: retain the
            # actual previous command. Pose and force are still sampled each step.
            closed = torch.where(grip_decisions, closed, observations[..., 46])
        unit_force = force.mean if deterministic else force.sample()
        commanded_force = torch.where(closed.bool(), 2 * unit_force - 1, -1.0)
        actions = torch.cat((delta * self.pose_scale, closed[..., None],
                             commanded_force[..., None]), dim=-1)
        logp, _ = self.action_statistics(raw, actions, grip_decisions)
        return actions, logp, value, hidden

    def imitation_loss(self, observations, actions, mask):
        # Each demonstration starts with its own zero memory, never rollout memory.
        raw, _, _ = self(observations)
        return self.supervision_loss(raw, actions, mask)

    def supervision_loss(self, raw, actions, mask):
        """Teacher targets on the same states and recurrent history as PPO."""
        pose_loss = F.smooth_l1_loss(
            raw[..., :6], actions[..., :6] / self.pose_scale, reduction="none").mean(-1)
        grip_loss = F.binary_cross_entropy_with_logits(
            raw[..., 6], actions[..., 6], reduction="none")
        _, _, force = self.distributions(raw)
        force_loss = ((2 * force.mean - 1 - actions[..., 7]).square()
                      * actions[..., 6])
        return (pose_loss + grip_loss + force_loss)[mask].mean()
