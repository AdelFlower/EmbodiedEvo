# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object_height_w = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    return torch.where(object_height_w > minimal_height, 1.0, 0.0)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    T_W_B = env.scene[robot_cfg.name].data.root_state_w[:, :7]
    object_position_w = env.scene[object_cfg.name].data.root_pos_w

    command = env.command_manager.get_command(command_name)
    desired_position_b = command[:, :3]
    desired_position_w, _ = combine_frame_transforms(
        T_W_B[:, :3],
        T_W_B[:, 3:],
        desired_position_b,
    )
    distance_to_goal = torch.norm(desired_position_w - object_position_w, dim=1)
    object_is_above_minimal_height = object_position_w[:, 2] > minimal_height
    return object_is_above_minimal_height * (1 - torch.tanh(distance_to_goal / std))


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    object_position_w = env.scene[object_cfg.name].data.root_pos_w

    ee_frame = env.scene[ee_frame_cfg.name]
    end_effector_position_w = ee_frame.data.target_pos_w[..., 0, :]
    distance_to_object = torch.norm(object_position_w - end_effector_position_w, dim=1)

    return 1 - torch.tanh(distance_to_object / std)


def object_position_in_frame(
    env: ManagerBasedRLEnv,
    root_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the object's position in the requested root frame."""
    T_W_R = env.scene[root_frame_cfg.name].data.root_state_w[:, :7]
    object_position_w = env.scene[object_cfg.name].data.root_pos_w
    object_position_in_root_frame, _ = subtract_frame_transforms(T_W_R[:, :3], T_W_R[:, 3:], object_position_w)
    return object_position_in_root_frame


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)
    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def lift_object_rl_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rl_training: bool = False,
    command_name: str = "object_pose",
    position_tolerance: float = 0.05,
) -> torch.Tensor:
    """Dynamic success termination for lift object task.

    Supports multiple modes:
    - RL training: Always returns False (no early termination)
    - RL evaluation: Uses goal from command manager

    Args:
        env: The RL environment instance.
        object_cfg: The configuration of the object to track.
        robot_cfg: The robot configuration (needed to transform the goal to world frame).
        rl_training: If True, always returns False (disables success termination for RL training).
        command_name: The name of the command that is used to control the object.
        position_tolerance: Distance tolerance for success (m).

    Returns:
        A boolean tensor of shape (num_envs,) indicating success.
    """
    if rl_training:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    T_W_B = env.scene[robot_cfg.name].data.root_state_w[:, :7]
    object_position_w = env.scene[object_cfg.name].data.root_pos_w

    command = env.command_manager.get_command(command_name)
    desired_position_b = command[:, :3]

    # Transform goal from robot-base frame to world frame
    desired_position_w, _ = combine_frame_transforms(T_W_B[:, :3], T_W_B[:, 3:], desired_position_b)

    distance = torch.linalg.norm(desired_position_w - object_position_w, dim=1)
    return distance < position_tolerance


def reset_cube(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Restore cube pose and zero velocity at each episode reset."""
    cube = env.scene["dex_cube"]
    pose = torch.tensor((0.5, 0.0, 0.055, 1.0, 0.0, 0.0, 0.0), device=env.device).repeat(len(env_ids), 1)
    pose[:, :3] += env.scene.env_origins[env_ids]
    cube.write_root_pose_to_sim(pose, env_ids=env_ids)
    cube.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


def generated_object_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Keep the policy command quaternion in XYZW while Sim 4.5 uses WXYZ."""
    command = env.command_manager.get_command("object_pose")
    return torch.cat((command[:, :3], command[:, 4:7], command[:, 3:4]), dim=1)


def ee_frame_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return end-effector position relative to its environment origin."""
    return env.scene["ee_frame"].data.target_pos_w[:, 0, :] - env.scene.env_origins


def ee_frame_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Keep the end-effector observation quaternion in XYZW."""
    quaternion = env.scene["ee_frame"].data.target_quat_w[:, 0, :]
    return torch.cat((quaternion[:, 1:], quaternion[:, :1]), dim=1)
