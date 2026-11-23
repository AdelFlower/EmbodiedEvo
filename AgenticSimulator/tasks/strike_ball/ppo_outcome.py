"""Predict a frozen neural swing's outcome and learn an additional PPO action head."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class OutcomeAdjustment(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        angles = torch.arange(8) * math.pi / 8
        directions = torch.stack((angles.cos(), angles.sin()), dim=1)
        rates = torch.tensor((0.5, 1., 2., 4., 8., 16., 32.))
        self.register_buffer("frequencies", (math.pi * rates[:, None, None] * directions).reshape(-1, 2))
        self.register_buffer("clock_center", torch.zeros(()))
        self.clock = nn.Sequential(nn.Linear(18, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 1))
        self.outcome = nn.Sequential(nn.Linear(119, 160), nn.SiLU(), nn.Linear(160, 160), nn.SiLU(), nn.Linear(160, 2))
        self.policy = nn.Sequential(nn.Linear(25, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(), nn.Linear(96, action_dim))
        nn.init.zeros_(self.policy[-1].weight)
        nn.init.zeros_(self.policy[-1].bias)

    def predict(self, start):
        phase = start @ self.frequencies.T
        axial = phase[:, 8:40:4]
        low = torch.cat((start, axial.sin(), axial.cos()), dim=1)
        clock = self.clock_center + 10. * self.clock(low)
        tick = clock.detach().round()
        detail = torch.cat(((tick - self.clock_center) / 10., clock.detach() - tick,
                            (math.pi / 2 * tick).sin(), (math.pi / 2 * tick).cos(),
                            (2 * math.pi * clock.detach()).sin()), dim=1)
        offset = self.outcome(torch.cat((start, phase.sin(), phase.cos(), detail), dim=1))
        return offset, clock, torch.cat((low, offset.clamp(-6., 6.), detail), dim=1)

    def forward(self, start):
        with torch.no_grad():
            _, _, features = self.predict(start)
        return 0.25 * self.policy(features)

    def fit_outcomes(self, start, offset, contact, distance):
        """Use only measured outcomes as supervision. PPO alone trains the action-producing layers."""
        order = torch.randperm(len(start), device=start.device)
        held = order[:len(order) // 10]
        train = order[len(held):]
        width = torch.quantile(distance[train], 0.95) - torch.quantile(distance[train], 0.05)
        substep = contact + (0.9 * (distance - distance[train].median()) / width.clamp_min(0.001)).clamp(-0.49, 0.49)
        self.clock_center.copy_(contact[train].median())
        target = (offset / 0.01).clamp(-20., 20.)
        optim = torch.optim.Adam([*self.clock.parameters(), *self.outcome.parameters()], lr=0.001)
        for it in range(4000):
            ids = train[torch.randint(len(train), (768,), device=start.device)]
            predicted, clock, _ = self.predict(start[ids])
            clock_loss = F.smooth_l1_loss(clock[:, 0], substep[ids], beta=0.15)
            loss = clock_loss + F.smooth_l1_loss(predicted, target[ids], beta=0.2)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if it in (2499, 3499):
                optim.param_groups[0]["lr"] *= 0.35
        self.clock.requires_grad_(False)
        self.outcome.requires_grad_(False)
        with torch.no_grad():
            pred, clock, _ = self.predict(start[held])
            error = (0.01 * pred - offset[held]).norm(dim=1)
            return dict(samples=len(start), heldout=len(held),
                        baseline_median_miss_mm=1000. * offset[held].norm(dim=1).median().item(),
                        prediction_error_mm=1000. * error.median().item(),
                        prediction_error_p90_mm=1000. * error.quantile(0.9).item(),
                        contact_substep_accuracy=(clock[:, 0].round() == contact[held]).float().mean().item())
