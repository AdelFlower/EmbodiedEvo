"""Strike observations, action limits, reset and shot trajectory measurements."""

import torch
from isaaclab.envs.mdp import JointPositionAction
from isaaclab.utils.math import quat_apply

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


def reset_shot(env, env_ids):
    env.reset_shot(env_ids)
