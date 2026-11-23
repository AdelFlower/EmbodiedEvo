"""PPO curriculum on forward kinematics. Only scalar rewards supervise the neural actor."""

import math

import torch
from torch import nn

from .scene import HOLE_CENTER, SAMPLE_X, SAMPLE_Y, SAMPLE_Z
from .mdp import paddle_normal


class PaddleModel:
    """Forward rotations about the six measured UR10 axes at reset."""

    def __init__(self, env, device, flight_time):
        robot = env.robot
        links = [robot.body_names.index(name) for name in (
            "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link",
        )]
        origin = env.scene.env_origins[0]
        axes = robot.root_physx_view.get_jacobians()[0, env.wrist - 1, 3:].T.to(device)
        self.anchors = (robot.data.body_link_pos_w[0, links] - origin).to(device)
        self.center = (env.paddle_center()[0] - origin).to(device)
        self.half_range = self.center.new_tensor([(high - low) / 2 for low, high in (SAMPLE_X, SAMPLE_Z)])
        # Native PPO expresses starts as offsets from the hole divided by the half range.
        lower_offset = self.center.new_tensor((SAMPLE_X[0] - HOLE_CENTER[0], SAMPLE_Z[0] - HOLE_CENTER[2]))
        self.start_min = lower_offset / self.half_range
        self.normal = paddle_normal(env)[0].to(device)
        x, y, z = axes.T
        zero = torch.zeros_like(x)
        self.cross = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=1).reshape(6, 3, 3)
        self.identity = torch.eye(3, device=device)
        self.perpendicular = self.identity - axes[:, :, None] * axes[:, None, :]
        self.gravity = env.gravity.to(device)
        self.physics_dt = env.physics_dt
        self.flight_time = flight_time

    def sample_starts(self, count):
        return self.start_min + 2 * torch.rand(count, 2, device=self.start_min.device)

    def pose(self, angles):
        theta = angles[:, :, None, None]
        rotations = self.identity + theta.sin() * self.cross + (theta.cos() - 1.) * self.perpendicular
        transform = rotations[:, 0]
        origin = self.anchors[0].expand(len(angles), -1)
        for j in range(1, 6):
            origin = origin + transform @ (self.anchors[j] - self.anchors[j - 1])
            transform = transform @ rotations[:, j]
        center = origin + transform @ (self.center - self.anchors[-1])
        return center, transform @ self.normal

    def rewards(self, start, actions, contact_time):
        hole = start.new_tensor(HOLE_CENTER)
        ball = hole.expand(len(start), 3).clone()
        ball[:, [0, 2]] += start * self.half_range
        ball[:, 1] = SAMPLE_Y
        flight = self.flight_time
        velocity = (hole - ball) / flight - self.gravity * (flight + self.physics_dt) / 2
        direction = nn.functional.normalize(velocity, dim=1)
        rest, preparation, impact = actions
        center, normal = self.pose(preparation)
        prep_error = (center - ball + 0.16 * direction).norm(dim=1)
        prep_tilt = torch.linalg.cross(normal, direction).norm(dim=1)
        joints, rate = impact.split(6, dim=1)
        center, normal = self.pose(joints)
        ahead, _ = self.pose(joints + 0.005 * rate)
        behind, _ = self.pose(joints - 0.005 * rate)
        paddle_speed = 0.56 * velocity
        wanted = ball - 0.023 * direction + (contact_time - 0.80) * paddle_speed
        position_error = (center - wanted).norm(dim=1)
        tilt = torch.linalg.cross(normal, direction).norm(dim=1)
        speed_error = ((ahead - behind) / 0.01 - paddle_speed).norm(dim=1)
        return (
            -torch.log1p((rest.norm(dim=1) / 0.025).square()),
            -torch.log1p((prep_error / 0.015).square()) - 0.35 * torch.log1p((prep_tilt / 0.06).square()),
            -torch.log1p((position_error / 0.006).square()) - 0.65 * torch.log1p((tilt / 0.035).square())
            - 0.65 * torch.log1p((speed_error / 0.12).square()),
        )


def endpoint_means(actor, start, rest, preparation, contact):
    width = 0.015
    times = torch.cat((rest, preparation, contact, contact - width, contact + width))
    home, pre, hit, left, right = actor(start.repeat(5, 1), times).chunk(5)
    return home, pre, torch.cat((hit, (right - left) / (2 * width)), dim=1)


def relative_advantage(reward, group_size):
    grouped = reward.view(-1, group_size)
    peers = (grouped.sum(1, keepdim=True) - grouped) / (group_size - 1)
    advantage = reward - peers.flatten()
    return (advantage - advantage.mean()) / (advantage.std() + 1e-7)


def warmup(actor, env, updates, flight_time=0.70, contact_window=0.):
    """Sample joint commands, score their paddle motion, and optimize clipped probability ratios."""
    device = next(actor.parameters()).device
    arm = PaddleModel(env, device, flight_time)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.0008)
    validation = arm.sample_starts(512)
    contexts, group = 192, 8
    for iteration in range(updates):
        start = arm.sample_starts(contexts).repeat_interleave(group, dim=0)
        rest = (0.06 * torch.rand(contexts, 1, device=device)).repeat_interleave(group, dim=0)
        pre = (0.52 + 0.04 * torch.rand(contexts, 1, device=device)).repeat_interleave(group, dim=0)
        contact = torch.full_like(pre, 0.80)
        if contact_window:
            contact += contact_window * (torch.rand(contexts, 1, device=device) - 0.5).repeat_interleave(group, dim=0)
        noise = max(0.0035, 0.28 * math.exp(-iteration / 550))
        sigma = start.new_full((6,), noise)
        scales = (sigma, sigma, torch.cat((sigma, 3 * sigma)))
        with torch.no_grad():
            old = endpoint_means(actor, start, rest, pre, contact)
            actions = [mu + sd * torch.randn_like(mu) for mu, sd in zip(old, scales)]
            old_logs = [-0.5 * ((a - mu) / sd).square().sum(1) for a, mu, sd in zip(actions, old, scales)]
            rewards = arm.rewards(start, actions, contact)
            advantages = [relative_advantage(reward, group) for reward in rewards]
        for _ in range(3):
            for ids in torch.randperm(len(start), device=device).chunk(4):
                means = endpoint_means(actor, start[ids], rest[ids], pre[ids], contact[ids])
                gain = 0.
                for weight, mu, a, sd, old_log, adv in zip((0.15, 0.45, 1.), means, actions, scales, old_logs, advantages):
                    log_prob = -0.5 * ((a[ids] - mu) / sd).square().sum(1)
                    ratio = (log_prob - old_log[ids]).clamp(-20, 20).exp()
                    gain += weight * torch.minimum(ratio * adv[ids], ratio.clamp(0.8, 1.2) * adv[ids]).mean()
                optimizer.zero_grad()
                (-gain).backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.)
                optimizer.step()
            with torch.no_grad():
                current = endpoint_means(actor, start, rest, pre, contact)
                kl = max(0.5 * ((mu - prev) / sd).square().sum(1).mean().item()
                         for mu, prev, sd in zip(current, old, scales))
            if kl > 0.035:
                break
        params = optimizer.param_groups[0]
        if kl > 0.04:
            params["lr"] = max(4e-7, 0.8 * params["lr"])
        elif kl < 0.01:
            params["lr"] = min(0.001, 1.1 * params["lr"])
        if iteration % 250 == 0 or iteration + 1 == updates:
            with torch.no_grad():
                time = torch.zeros_like(validation[:, :1])
                test_actions = endpoint_means(actor, validation, time, time + 0.54, time + 0.80)
                impact_reward = arm.rewards(validation, test_actions, time + 0.80)[2].mean().item()
            print(f"Kinematic PPO update {iteration + 1}/{updates}, validation impact reward {impact_reward:.3f}", flush=True)
