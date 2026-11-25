"""Six-joint neural actor used by the episodic PPO strike trainer and playback."""

import math

import torch
from torch import nn

from .ppo_outcome import OutcomeAdjustment


class StrikeActor(nn.Module):
    """Map the observed start position and the current clock directly to six joint offsets."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(), nn.Linear(96, 6),
        )
        self.refinement = None
        self.feedback = None

    def enable_refinement(self):
        """Add a zero-initialized output branch for position-dependent native collision effects."""
        self.refinement = nn.Sequential(nn.Linear(23, 64), nn.Tanh(), nn.Linear(64, 6))
        nn.init.zeros_(self.refinement[-1].weight)
        nn.init.zeros_(self.refinement[-1].bias)

    def enable_feedback(self):
        """Let PPO adjust the next joint command from the current native observation."""
        self.feedback = nn.Sequential(nn.Linear(35, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 6))
        nn.init.zeros_(self.feedback[-1].weight)
        nn.init.zeros_(self.feedback[-1].bias)
        scale = torch.ones(35)
        scale[6:12] = 0.25  # joint velocities
        scale[18:24] = 0.5  # paddle and ball velocities
        scale[27:33] = 0.5  # previous native joint actions
        self.register_buffer("observation_scale", scale)

    def feedback_offset(self, observation):
        return 0.1 * self.feedback((observation * self.observation_scale).clamp(-5., 5.))

    def forward(self, start, seconds):
        phase = math.pi * seconds
        inputs = torch.cat((
            start, seconds, phase.sin(), phase.cos(), (2 * phase).sin(), (2 * phase).cos(),
        ), dim=-1)
        angles = self.net(inputs)
        if self.refinement is not None:
            spatial = []
            for frequency in (1., 2., 4., 8.):
                coordinate = (math.pi * frequency) * start
                spatial.extend((coordinate.sin(), coordinate.cos()))
            angles = angles + 0.1 * self.refinement(torch.cat([inputs, *spatial], dim=-1))
        return angles


class JointAdjustment(nn.Module):
    """Learn position-dependent biases and gains on the neural actor's six outputs."""

    def __init__(self, output_size=12):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(34, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(), nn.Linear(96, output_size))
        nn.init.normal_(self.net[-1].weight, std=0.001)
        nn.init.zeros_(self.net[-1].bias)
        self.fine = None
        self.outcome_adjustments = nn.ModuleList()

    def add_outcome_adjustment(self):
        layer = OutcomeAdjustment(self.net[-1].out_features).to(self.net[0].weight.device)
        self.outcome_adjustments.append(layer)
        return layer

    def enable_fine(self):
        """A small linear readout of spatial Fourier features for millimetre-scale residuals."""
        device = self.net[0].weight.device
        angles = torch.arange(16, device=device) * (math.pi / 16)
        directions = torch.stack((angles.cos(), angles.sin()), dim=1)
        rates = torch.tensor((0.5, 1., 2., 3., 4., 6., 8., 12., 16.), device=device)
        frequencies = (math.pi * rates[:, None, None] * directions[None, :, :]).reshape(-1, 2)
        self.register_buffer("fine_frequencies", frequencies)
        self.fine = nn.Linear(2 * len(frequencies), self.net[-1].out_features, device=device)
        nn.init.zeros_(self.fine.weight)
        nn.init.zeros_(self.fine.bias)

    def forward(self, start):
        x, z = start.split(1, dim=-1)
        directions = torch.cat((x, z, (x + z) / math.sqrt(2.), (x - z) / math.sqrt(2.)), dim=-1)
        features = [start]
        for frequency in (1., 2., 4., 8.):
            phase = directions * (math.pi * frequency)
            features.extend((phase.sin() / math.sqrt(frequency), phase.cos() / math.sqrt(frequency)))
        mean = self.net(torch.cat(features, dim=-1))
        if self.fine is not None:
            phase = start @ self.fine_frequencies.T
            mean = mean + 0.1 * self.fine(torch.cat((phase.sin(), phase.cos()), dim=-1))
        for layer in self.outcome_adjustments:
            mean = mean + layer(start)
        return mean


class FrozenStrikeNetwork:
    """Retain each episode's observed ball position, including after the ball takes flight."""

    def __init__(self, bundle, device):
        self.actor = StrikeActor()
        if bundle.get("spatial_refinement", False):
            self.actor.enable_refinement()
        if bundle.get("feedback", False):
            self.actor.enable_feedback()
        self.actor.to(device)
        self.actor.load_state_dict(bundle["actor"])
        self.actor.eval()
        self.adjustment = None
        if "adjustment" in bundle:
            self.adjustment = JointAdjustment(len(bundle["adjustment_scale"])).to(device)
            if bundle.get("fine_adjustment", False):
                self.adjustment.enable_fine()
            for _ in range(bundle.get("outcome_adjustments", 0)):
                self.adjustment.add_outcome_adjustment()
            self.adjustment.load_state_dict(bundle["adjustment"])
            self.adjustment.eval()
            self.adjustment_scale = torch.tensor(bundle["adjustment_scale"], device=device)
            self.adjustment_basis = bundle.get("adjustment_basis", "joint-gain")
        self.half_range = torch.tensor(bundle["ball_half_range"], device=device)
        self.joint_scale = bundle["joint_action_scale"]
        self.episode_seconds = bundle["episode_seconds"]
        self.start = None
        self.controls = None

    @torch.no_grad()
    def __call__(self, obs):
        remaining = obs[:, -1:]
        position = -obs[:, [24, 26]] / self.half_range
        if self.start is None:
            self.start = position
            if self.adjustment is not None:
                self.controls = self.adjustment(position) * self.adjustment_scale
        else:
            reset_ids = (remaining[:, 0] == 1.).nonzero().flatten()
            if len(reset_ids):
                self.start[reset_ids] = position[reset_ids]
                if self.adjustment is not None:
                    self.controls[reset_ids] = self.adjustment(position[reset_ids]) * self.adjustment_scale
        seconds = (1. - remaining) * self.episode_seconds
        controls = self.controls
        if controls is not None and controls.shape[-1] == 13:
            seconds = (seconds + controls[:, 12:]).clamp(0., self.episode_seconds)
        joints = self.actor(self.start, seconds)
        if self.actor.feedback is not None:
            joints = joints + self.actor.feedback_offset(obs)
        if controls is not None:
            bias, weight = controls[:, :12].split(6, dim=-1)
            basis = (math.pi * seconds).sin() if self.adjustment_basis == "clock" else joints
            joints = joints + weight * basis + bias
        return joints / self.joint_scale
