"""Physical metrics, BC-compatible placement monitor and training-only shaping."""

import math

import torch


def rotate(quaternion, vector):
    xyz = quaternion[..., 1:]
    uv = 2 * torch.cross(xyz, vector.expand_as(xyz), dim=-1)
    return vector + quaternion[..., :1] * uv + torch.cross(xyz, uv, dim=-1)


def physical_metrics(observation, initial_cube_z):
    hand, cube, plate = observation[:, 16:23], observation[:, 23:30], observation[:, 33:40]
    grip = hand[:, :3] + rotate(hand[:, 3:], hand.new_tensor([0., 0., 0.1034]))
    cube_bottom = cube[:, :3] + rotate(cube[:, 3:], cube.new_tensor([0., 0., -0.0125]))
    plate_bottom = plate[:, :3] + rotate(plate[:, 3:], plate.new_tensor([0., 0., -0.0178308]))
    difference = cube_bottom - plate_bottom
    w, x, y, z = cube[:, 3:].unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    # A cube permits four equivalent top-down grasps; q and -q are equivalent.
    angle = yaw[:, None] / 2 + torch.arange(4, device=yaw.device) * (math.pi / 4)
    alignment = (hand[:, 4:5] * angle.cos() + hand[:, 5:6] * angle.sin()).square().amax(-1)
    return dict(cube_rise=cube[:, 2] - initial_cube_z,
                grasp_distance=(cube[:, :3] - grip).norm(dim=-1),
                xy_distance=difference[:, :2].norm(dim=-1),
                height_error=difference[:, 2].abs(), alignment=alignment)


class TaskReward:
    def __init__(self, num_envs, device, config, gamma):
        self.config, self.gamma = config, gamma
        self.seen_lift = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.seen_release = self.seen_lift.clone()
        self.success = self.seen_lift.clone()
        self.stable_seconds = torch.zeros(num_envs, device=device)
        self.potential = torch.zeros_like(self.stable_seconds)

    def reset(self, ids, metrics):
        self.seen_lift[ids] = False
        self.seen_release[ids] = False
        self.success[ids] = False
        self.stable_seconds[ids] = 0
        self.potential[ids] = self.compute_potential(metrics, self.seen_lift)[ids]

    def observe(self, metrics, closed, active, dt):
        # Same geometry/time thresholds and observed lift/release gating as eval_bc.
        self.seen_lift |= active & closed & (metrics["cube_rise"] >= 0.06) & (metrics["grasp_distance"] <= 0.05)
        self.seen_release |= active & self.seen_lift & ~closed
        placed = (metrics["xy_distance"] < 0.03) & (metrics["height_error"] < 0.01)
        verifying = active & self.seen_release & ~closed & placed
        self.stable_seconds = torch.where(verifying, self.stable_seconds + dt, 0.0)
        self.success |= active & (self.stable_seconds > 0.5)

    def compute_potential(self, m, seen_lift):
        c = self.config
        near = torch.exp(-m["grasp_distance"] / 0.1)
        held = (m["grasp_distance"] <= 0.05).float()
        # Plateau inside the native placement tolerances. Continuing to reward
        # smaller errors there encourages pressing/knocking an already placed
        # cube instead of leaving it stable for the success check.
        placed = seen_lift * torch.exp(
            -(m["xy_distance"] - 0.03).clamp_min(0) / 0.03
            -(m["height_error"] - 0.01).clamp_min(0) / 0.02)
        approaching = 1 - placed
        return ((c["reach"] * near + c["alignment"] * m["alignment"] * torch.sqrt(near)
                 + c["lift"] * (m["cube_rise"] / 0.1).clamp(0, 1) * held) * approaching
                + c["transport"] * seen_lift * torch.exp(-(m["xy_distance"] - 0.03).clamp_min(0) / 0.1)
                + c["place"] * placed
                + c.get("retreat", 0.0) * placed * (1 - torch.exp(-m["grasp_distance"] / 0.08)))

    def reward(self, metrics, active, terminated, previous_lift, dt):
        potential = self.compute_potential(metrics, self.seen_lift)
        # True terminals have zero potential. Time-limit truncations retain it.
        potential = torch.where(terminated, 0.0, potential)
        reward = self.gamma * potential - self.potential
        # Progress-only shaping supplies little signal once the critic learns
        # V ~= -potential on uniformly failing rollouts. Also reward maintaining
        # useful physical states (per second), with completion worth more than
        # indefinitely holding the highest-scoring state.
        reward += self.config.get("dense_scale", 0.0) * dt * potential
        reward += self.config["lift_bonus"] * (self.seen_lift & ~previous_lift)
        reward += self.config["success_bonus"] * self.success
        reward -= self.config["failure_penalty"] * (terminated & ~self.success)
        reward -= self.config["time_cost"] * dt
        self.potential.copy_(potential)
        return torch.where(active, reward, 0.0)
