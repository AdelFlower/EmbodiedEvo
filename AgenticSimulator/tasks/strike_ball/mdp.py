"""Task-specific Isaac Lab terms. The reward manager integrates rates over step_dt."""

import torch
from isaaclab.envs.mdp import JointPositionAction
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply

from .scene import BALL_RADIUS, FLIGHT_TIME


class LimitedJointPositionAction(JointPositionAction):
    """Apply the UR10's soft joint limits after Isaac Lab scales and offsets the action."""

    def process_actions(self, actions):
        super().process_actions(actions)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self.processed_actions.clamp_(limits[..., 0], limits[..., 1])


def ball_from_paddle(env):
    return env.ball.data.root_pos_w - env.paddle_center()


def paddle_normal(env):
    return quat_apply(env.robot.data.body_link_quat_w[:, env.wrist], env.local_normal)


def paddle_velocity(env):
    robot = env.robot.data
    lever = env.paddle_center() - robot.body_com_pos_w[:, env.wrist]
    return robot.body_com_lin_vel_w[:, env.wrist] + torch.linalg.cross(robot.body_com_ang_vel_w[:, env.wrist], lever)


def hole_from_ball(env):
    return env.hole - env.ball.data.root_pos_w


def has_hit(env):
    return env.hit[:, None].float()


def remaining_time(env):
    return (1. - env.episode_length_buf / env.max_episode_length)[:, None]


def approach_reward(env, distance_scale: float):
    """Improvement over the pre-hit closest approach, summed over this policy step."""
    previous = torch.exp(-env.previous_best_distance / distance_scale)
    current = torch.exp(-env.best_distance / distance_scale)
    return (current - previous) / env.step_dt


def paddle_hit_reward(env):
    return env.first_hit_this_step.float() / env.step_dt


def reference_velocity(env, start, env_ids=None):
    hole = env.hole if env_ids is None else env.hole[env_ids]
    return (hole - start) / FLIGHT_TIME - 0.5 * env.gravity * (FLIGHT_TIME + env.physics_dt)


class SwingPreparation(ManagerTermBase):
    """Pay only for improving the best pre-impact combination of reach, tilt and speed."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.best = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.best[env_ids if env_ids is not None else slice(None)] = 0.

    def __call__(self, env):
        ball = env.held_state[:, :3]
        target = reference_velocity(env, ball)
        direction = torch.nn.functional.normalize(target, dim=1)
        center_error = (ball - BALL_RADIUS * direction - env.paddle_center()).norm(dim=1)
        near = torch.exp(-center_error / 0.12)
        aligned = (paddle_normal(env) * direction).sum(dim=1).abs().clamp_max(1.).pow(6)
        speed_error = (paddle_velocity(env) - 0.55 * target).norm(dim=1)
        quality = near * (0.25 + 0.75 * aligned) * (0.35 + 0.65 * torch.exp(-speed_error))
        gain = (quality - self.best).clamp_min(0.) * (~env.hit | env.first_hit_this_step)
        self.best += gain
        return gain / env.step_dt


def launch_arc_reward(env):
    """A release with too little upward speed still gets directional feedback."""
    return torch.exp(-env.launch_speed_error_this_step / 1.5) / env.step_dt


def predict_launch_aim_error(offset, velocity, gravity, dt, remaining_steps):
    """Predict a free ball's descending board crossing, apex if too low, or deadline."""
    return predict_launch_offset(offset, velocity, gravity, dt, remaining_steps).norm(dim=-1)


def predict_launch_offset(offset, velocity, gravity, dt, remaining_steps):
    """Signed displacement from the goal at the crossing, apex, or deadline."""
    # Semi-implicit gravity gives p(t) = p0 + (v0 + g*(t+dt)/2)*t at physics samples.
    down_accel = -gravity[2]
    height, vz = offset[:, 2], velocity[:, 2]
    apex_step = (vz / (down_accel * dt)).floor().clamp_min(0)
    apex_time = apex_step * dt
    apex_height = height + (vz - 0.5 * down_accel * (apex_time + dt)) * apex_time

    vertical_speed = vz - 0.5 * down_accel * dt
    discriminant = vertical_speed.square() + 2 * down_accel * height
    down_time = (vertical_speed + discriminant.clamp_min(0).sqrt()) / down_accel
    reaches_plane = apex_height >= 0
    sample = torch.where(reaches_plane, (down_time / dt).floor().clamp_min(0), apex_step)
    sample = torch.minimum(sample, remaining_steps)
    t = (sample * dt).unsqueeze(1)
    position = offset + (velocity + 0.5 * gravity * (t + dt)) * t

    # Match check_passage's straight-line interpolation between neighboring physics samples.
    step_delta = (velocity + gravity * (t + dt)) * dt
    fraction = (position[:, 2] / (-step_delta[:, 2]).clamp_min(1e-12)).clamp(0., 1.)
    fraction = torch.where(reaches_plane & (sample < remaining_steps), fraction, 0.)
    return position + fraction.unsqueeze(1) * step_delta


def launch_aim_reward(env, coarse_scale: float, fine_scale: float):
    error = env.launch_aim_error_this_step
    return 0.5 * (torch.exp(-error / coarse_scale) + torch.exp(-error / fine_scale)) / env.step_dt


def hole_distance_reward(env, distance_scale: float):
    """Reward proximity to the entrance on each policy step of the shot."""
    distance = (env.ball.data.root_pos_w - env.hole_top).norm(dim=-1)
    return torch.exp(-distance / distance_scale) * (env.hit & ~env.passed)


def board_contact_penalty(env):
    """Count distinct ball/board impacts accumulated during this policy step."""
    return env.board_impacts_this_step / env.step_dt


def hole_pass_reward(env):
    return env.first_pass_this_step.float() / env.step_dt


def reset_shot(env, env_ids):
    env.reset_shot(env_ids)
