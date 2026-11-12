"""50 Hz 专家决策及可复用的 200 Hz 位姿动作执行器。

仅在 AppLauncher 启动后导入本模块。动作和观测均使用 schema 的物理单位。
"""

import torch
from isaaclab.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    compute_pose_error,
    quat_apply_inverse,
    subtract_frame_transforms,
)

from .expert import BatchExpert
from .schema import ACTION_DIM, INITIAL_PREVIOUS_ACTION, normalize_grip_force


def canonical_quaternion(quaternion):
    """单位化并固定 q/-q 的表示；最大绝对值分量取正。"""
    norm = quaternion.norm(dim=-1, keepdim=True)
    if not torch.isfinite(norm).all().item() or (norm < 1e-8).any().item():
        raise ValueError("观测包含无效四元数")
    quaternion = quaternion / norm
    largest = quaternion.gather(-1, quaternion.abs().argmax(-1, keepdim=True))
    return torch.where(largest < 0, -quaternion, quaternion)


class BCExpertController(BatchExpert):
    """专家状态机只在 decide() 中推进，physics_step() 只跟踪锁存目标。

    set_action()/physics_step() 也可接收 BC 动作；力度按任务约定固定 10 N/指。
    保留原控制器的成功判据、阶段超时、IK 和关节目标限速。
    """

    def __init__(self, robot, cube, plate, device):
        super().__init__(robot, cube, plate, device, grip_force=10.0)
        self.previous_action = torch.tensor(
            INITIAL_PREVIOUS_ACTION, device=device, dtype=torch.float32,
        ).repeat(self.n, 1)
        self._held_position = None
        self._held_quaternion = None
        self._tracking = torch.zeros(self.n, dtype=torch.bool, device=device)
        self._has_closed = self._tracking.clone()

    def observation(self):
        robot = self.robot
        root = robot.data.root_state_w[:, :7]

        def pose_in_base(pose):
            position, quaternion = subtract_frame_transforms(
                root[:, :3], root[:, 3:7], pose[:, :3], pose[:, 3:7],
            )
            return torch.cat((position, canonical_quaternion(quaternion)), dim=-1)

        observation = torch.cat((
            robot.data.joint_pos[:, self.arm_ids],
            robot.data.joint_vel[:, self.arm_ids],
            robot.data.joint_pos[:, self.finger_ids],
            pose_in_base(robot.data.body_state_w[:, self.hand_id, :7]),
            pose_in_base(self.cube.data.root_state_w[:, :7]),
            quat_apply_inverse(root[:, 3:7], self.cube.data.root_lin_vel_w),
            pose_in_base(self.plate.data.root_state_w[:, :7]),
            self.previous_action,
        ), dim=-1)
        if not torch.isfinite(observation).all().item():
            raise ValueError("观测包含 NaN/Inf")
        return observation.clone()

    def decide(self, control_dt):
        # 这里不计算或发送低层关节目标，也不提前移动夹爪。
        super().step(control_dt, decision_only=True)
        target_position, target_quaternion, moving = self._target_pose()
        root = self.robot.data.root_state_w[:, :7]
        hand = self.robot.data.body_state_w[:, self.hand_id, :7]
        current_position, current_quaternion = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], hand[:, :3], hand[:, 3:7],
        )
        target_position, target_quaternion = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], target_position, target_quaternion,
        )
        delta_position, delta_rotation = compute_pose_error(
            current_position, current_quaternion, target_position, target_quaternion,
            rot_error_type="axis_angle",
        )
        force = torch.where(
            self.finger_force_mode, normalize_grip_force(self.grip_force), -1.0,
        )
        actions = torch.cat((
            delta_position, delta_rotation,
            self.finger_force_mode[:, None].float(), force[:, None],
        ), dim=-1)
        self.set_action(actions, tracking=moving)
        return actions.clone()

    def set_action(self, actions, *, tracking=None):
        """以调用时的实际手掌位姿为基准生成目标，仅在新动作到来时调用。"""
        if actions.shape != (self.n, ACTION_DIM) or not torch.isfinite(actions).all().item():
            raise ValueError(f"动作必须为有限值数组 ({self.n}, {ACTION_DIM})")
        if not ((actions[:, 6] == 0) | (actions[:, 6] == 1)).all().item():
            raise ValueError("夹爪动作必须是 0/1")
        if (actions[:, 7].abs() > 1).any().item():
            raise ValueError("力度标签必须位于 [-1, 1]")
        # 执行力度固定为 10 N，拒绝与实际执行不一致的闭合标签。
        if ((actions[:, 6] == 1) & (actions[:, 7].abs() > 1e-6)).any().item():
            raise ValueError("固定 10 N 时闭合力度标签必须为 0")
        root = self.robot.data.root_state_w[:, :7]
        hand = self.robot.data.body_state_w[:, self.hand_id, :7]
        position, quaternion = subtract_frame_transforms(
            root[:, :3], root[:, 3:7], hand[:, :3], hand[:, 3:7],
        )
        position, quaternion = apply_delta_pose(position, quaternion, actions[:, :6])
        self._held_position, self._held_quaternion = combine_frame_transforms(
            root[:, :3], root[:, 3:7], position, quaternion,
        )
        enabled = ~self.terminal
        close = actions[:, 6] == 1
        close = torch.where(enabled, close, self.finger_force_mode)
        self._gripper_modes_dirty |= (close != self.finger_force_mode).any()
        self.finger_force_mode = close
        self._has_closed |= close
        self._tracking = enabled if tracking is None else enabled & tracking
        self.previous_action = torch.where(enabled[:, None], actions, self.previous_action).clone()

    def physics_step(self, physics_dt):
        if self._held_position is None:
            raise RuntimeError("physics_step 前必须先设置动作")
        active = ~self.terminal
        opening_limit = torch.where(self._has_closed, 0.03, 0.04)
        width = torch.where(
            self.finger_force_mode,
            (self.width - 0.05 * physics_dt).clamp_min(0.0),
            torch.minimum(self.width + 0.05 * physics_dt, opening_limit),
        )
        self.width = torch.where(active, width, self.width)
        self._track_pose(
            physics_dt, self._held_position, self._held_quaternion,
            self._tracking & active,
        )
        self._send()
