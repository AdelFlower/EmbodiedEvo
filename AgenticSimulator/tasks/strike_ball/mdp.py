"""Task-specific Isaac Lab terms. The reward manager integrates rates over step_dt."""

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


def approach_reward(env, distance_scale: float):
    """Improvement over the pre-hit closest approach, summed over this policy step."""
    previous = torch.exp(-env.previous_best_distance / distance_scale)
    current = torch.exp(-env.best_distance / distance_scale)
    return (current - previous) / env.step_dt


def paddle_hit_reward(env):
    return env.first_hit_this_step.float() / env.step_dt


def launch_velocity_reward(env, velocity_scale: float):
    return torch.exp(-env.launch_error_this_step / velocity_scale) / env.step_dt


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
