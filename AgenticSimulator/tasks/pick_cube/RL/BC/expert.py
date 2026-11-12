"""BC 采集私有的批量专家适配。

基于 pick_cube/batch_controller.py 的专家实现，拆分决策与跟踪。
原控制器保持不变；本模块仅在 AppLauncher 启动后导入。
"""

import math

import torch
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_quat,
    quat_apply,
    subtract_frame_transforms,
)


class BatchExpert:
    WAIT, CLEAR, ABOVE, DESCEND, CLOSE, LIFT, TRANSFER, LOWER, \
        OPEN, RETREAT, VERIFY, DONE, FAILED = range(13)

    STATE_NAMES = (
        "wait", "clearance", "above_cube", "descend", "close",
        "lift", "transfer", "lower", "open", "retreat", "verify",
        "done", "failed",
    )
    REASON_NAMES = (
        "none",
        "stage_timeout",
        "invalid_ik",
        "cube_slipped",
        "grasp_verification_failed",
        "placement_verification_failed",
        "success",
    )

    def __init__(self, robot, cube, plate, device, grip_force=20.0):
        self.robot = robot
        self.cube = cube
        self.plate = plate
        self.device = device
        self.n = robot.data.joint_pos.shape[0]
        self.grip_force = float(grip_force)
        if not 0.0 < self.grip_force <= 200.0:
            raise ValueError("grip_force 必须大于 0，且不超过当前配置的 200 N 上限")

        if not robot.is_fixed_base:
            raise ValueError("要求机器人固定底座")
        if cube.data.root_pos_w.shape[0] != self.n:
            raise ValueError("方块数量与机器人数量不一致")
        if plate.data.root_pos_w.shape[0] != self.n:
            raise ValueError("盘子数量与机器人数量不一致")

        self.arm_ids = [
            robot.find_joints(f"panda_joint{i}")[0][0]
            for i in range(1, 8)
        ]
        self.finger_ids = [
            robot.find_joints(f"panda_finger_joint{i}")[0][0]
            for i in (1, 2)
        ]
        self.finger_stiffness = robot.data.joint_stiffness[
            :, self.finger_ids
        ].clone()
        self.finger_damping = robot.data.joint_damping[
            :, self.finger_ids
        ].clone()
        # 每个环境独立保留模式；失败后沿用结束时的模式，与单环境一致。
        self.finger_force_mode = torch.zeros(
            self.n, dtype=torch.bool, device=device
        )
        self._gripper_modes_dirty = torch.zeros(
            (), dtype=torch.bool, device=device
        )
        self.hand_id = robot.find_bodies("panda_hand")[0][0]
        self.jacobian_id = self.hand_id - 1

        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
            ),
            num_envs=self.n,
            device=device,
        )

        self.state = torch.full(
            (self.n,), self.WAIT, dtype=torch.long, device=device
        )
        self.last_state = self.state.clone()
        self.reason = torch.zeros_like(self.state)

        self.elapsed = torch.zeros(self.n, device=device)
        self.stable = torch.zeros(self.n, device=device)
        self.total_time = torch.zeros(self.n, device=device)

        self.width = torch.full((self.n,), 0.04, device=device)
        self.last_arm = robot.data.default_joint_pos[
            :, self.arm_ids
        ].clone()

        self.offset = torch.tensor(
            [[0.0, 0.0, 0.1034]], device=device
        ).expand(self.n, -1)

        self.cube_start = torch.zeros(self.n, 3, device=device)
        self.goals = torch.zeros(self.n, 13, 3, device=device)
        self.goal_quat = torch.zeros(self.n, 4, device=device)
        self.goal_quat[:, 0] = 1.0

        # 只累计尚未结束的环境，避免把结束后的保持姿态算作活跃任务。
        self.active_env_steps = torch.zeros(
            (), dtype=torch.long, device=device
        )

    @property
    def terminal(self):
        return self.state >= self.DONE

    def _enter(self, mask, new_state):
        mask = mask & ~self.terminal
        if new_state in (self.CLOSE, self.OPEN):
            force_mode = new_state == self.CLOSE
            changed = mask & (self.finger_force_mode != force_mode)
            self._gripper_modes_dirty |= changed.any()
            self.finger_force_mode = torch.where(
                mask, force_mode, self.finger_force_mode
            )
        self.state = torch.where(mask, new_state, self.state)
        self.elapsed = torch.where(mask, 0.0, self.elapsed)
        self.stable = torch.where(mask, 0.0, self.stable)

    def _finish(self, mask, success, reason):
        mask = mask & ~self.terminal

        self.last_state = torch.where(mask, self.state, self.last_state)
        self.reason = torch.where(mask, reason, self.reason)
        self.last_arm = torch.where(
            mask[:, None],
            self.robot.data.joint_pos[:, self.arm_ids],
            self.last_arm,
        )
        self.state = torch.where(
            mask,
            self.DONE if success else self.FAILED,
            self.state,
        )

    def _grasp_position(self):
        hand = self.robot.data.body_state_w[:, self.hand_id, :7]
        return hand[:, :3] + quat_apply(hand[:, 3:7], self.offset)

    def _prepare(self, mask):
        cube = self.cube.data.root_pos_w
        plate = self.plate.data.root_pos_w
        grip = self._grasp_position()

        # 每个环境独立计算方块朝向。
        w, x, y, z = self.cube.data.root_quat_w.unbind(dim=-1)
        yaw = torch.atan2(
            2 * (w * z + x * y),
            1 - 2 * (y * y + z * z),
        )
        yaw = torch.remainder(
            yaw + math.pi / 4, math.pi / 2
        ) - math.pi / 4

        zeros = torch.zeros_like(yaw)
        quat = torch.stack(
            (zeros, torch.cos(yaw / 2), torch.sin(yaw / 2), zeros),
            dim=-1,
        )

        safe_z = torch.maximum(cube[:, 2], plate[:, 2]) + 0.15

        def at_z(position, height):
            return torch.cat(
                (position[:, :2], height[:, None]), dim=-1
            )

        clearance = at_z(grip, safe_z)
        above = at_z(cube, safe_z)
        transfer = at_z(plate, safe_z)
        lower = at_z(plate, plate[:, 2] + 0.01)

        goals = torch.stack(
            [
                grip, clearance, above, cube, cube,
                above, transfer, lower, lower,
                transfer, transfer, grip, grip,
            ],
            dim=1,
        )

        self.goals = torch.where(
            mask[:, None, None], goals, self.goals
        )
        self.goal_quat = torch.where(
            mask[:, None], quat, self.goal_quat
        )
        self.cube_start = torch.where(
            mask[:, None], cube, self.cube_start
        )

    def _target_pose(self):
        """当前专家阶段的手掌目标，世界坐标；不推进控制或物理。"""
        hand = self.robot.data.body_state_w[:, self.hand_id, :7]
        moving = (
            (self.state >= self.CLEAR)
            & (self.state <= self.VERIFY)
        )

        # 每个环境按自己的状态选目标。
        index = self.state[:, None, None].expand(-1, 1, 3)
        grip_target = self.goals.gather(1, index).squeeze(1)

        target_quat = torch.where(
            moving[:, None], self.goal_quat, hand[:, 3:7]
        )
        target_hand = grip_target - quat_apply(
            target_quat, self.offset
        )
        target_hand = torch.where(
            moving[:, None], target_hand, hand[:, :3]
        )

        return target_hand, target_quat, moving

    def _target_reached(self, target_hand, target_quat, moving):
        hand = self.robot.data.body_state_w[:, self.hand_id, :7]
        pos_error, rot_error = compute_pose_error(
            hand[:, :3], hand[:, 3:7], target_hand, target_quat,
        )
        tolerance = torch.where(self.state == self.DESCEND, 0.003, 0.008)
        return (
            moving
            & (pos_error.norm(dim=-1) < tolerance)
            & (rot_error.norm(dim=-1) < 0.10)
        )

    def _track(self, dt):
        return self._track_pose(dt, *self._target_pose())

    def _track_pose(self, dt, target_hand, target_quat, moving):
        """跟踪给定世界坐标目标；同时供原控制器和 BC 执行器使用。"""
        robot = self.robot
        hand = robot.data.body_state_w[:, self.hand_id, :7]
        root = robot.data.root_state_w[:, :7]

        current_pos, current_quat = subtract_frame_transforms(
            root[:, :3], root[:, 3:7],
            hand[:, :3], hand[:, 3:7],
        )
        target_pos, target_quat_b = subtract_frame_transforms(
            root[:, :3], root[:, 3:7],
            target_hand, target_quat,
        )

        jacobian = robot.root_physx_view.get_jacobians()[
            :, self.jacobian_id
        ][:, :, self.arm_ids].clone()

        # panda_hand link 原点指向 COM 的世界坐标偏移。
        r_com_w = quat_apply(
            hand[:, 3:7],
            robot.data.com_pos_b[:, self.hand_id, :],
        )

        # J_link_linear = J_com_linear + r_com × J_angular
        jacobian[:, :3, :] += torch.cross(
            r_com_w.unsqueeze(-1).expand_as(jacobian[:, 3:, :]),
            jacobian[:, 3:, :],
            dim=1,
        )

        rotation = matrix_from_quat(root[:, 3:7]).transpose(1, 2)
        jacobian[:, :3] = rotation @ jacobian[:, :3]
        jacobian[:, 3:] = rotation @ jacobian[:, 3:]

        valid_jacobian = torch.isfinite(jacobian).flatten(1).all(dim=1)
        jacobian = torch.nan_to_num(
            jacobian, nan=0.0, posinf=0.0, neginf=0.0
        )

        delta_pos, delta_rot = compute_pose_error(
            current_pos, current_quat,
            target_pos, target_quat_b,
            rot_error_type="axis_angle",
        )
        self.ik.set_command(
            torch.cat((delta_pos, delta_rot), dim=-1),
            ee_pos=current_pos,
            ee_quat=current_quat,
        )

        joint_pos = robot.data.joint_pos[:, self.arm_ids]
        desired = self.ik.compute(
            current_pos, current_quat, jacobian, joint_pos
        )

        finite = torch.isfinite(desired).all(dim=1) & valid_jacobian
        self._finish(moving & ~finite, False, 2)

        desired = torch.where(
            finite[:, None], desired, self.last_arm
        )

        # 核心：每台机械臂自己的限速因子，形状为 (N, 1)。
        delta = desired - self.last_arm
        max_delta = delta.abs().amax(dim=-1, keepdim=True)
        factor = (
            0.8 * dt / max_delta.clamp_min(1e-8)
        ).clamp(max=1.0)
        desired = self.last_arm + delta * factor

        limits = robot.data.soft_joint_pos_limits[:, self.arm_ids]
        desired = torch.maximum(
            torch.minimum(desired, limits[..., 1]), limits[..., 0]
        )

        moving = moving & ~self.terminal
        self.last_arm = torch.where(
            moving[:, None], desired, self.last_arm
        )

        return self._target_reached(target_hand, target_quat, moving)

    def _sync_gripper_modes(self):
        # 当前 Isaac Lab 的增益写入接口会传输到 CPU。每步只检查一次，
        # 将该步所有环境的切换合并提交；没有切换时不重复写入增益。
        if not self._gripper_modes_dirty.item():
            return

        stiffness = torch.where(
            self.finger_force_mode[:, None], 0.0, self.finger_stiffness
        )
        damping = torch.where(
            self.finger_force_mode[:, None], 0.0, self.finger_damping
        )
        self.robot.write_joint_stiffness_to_sim(
            stiffness, joint_ids=self.finger_ids
        )
        self.robot.write_joint_damping_to_sim(
            damping, joint_ids=self.finger_ids
        )
        hand = self.robot.actuators["panda_hand"]
        hand.stiffness.copy_(stiffness)
        hand.damping.copy_(damping)
        self._gripper_modes_dirty.zero_()

    def _send(self):
        gravity = (
            self.robot.root_physx_view
            .get_gravity_compensation_forces()
        )
        self.robot.set_joint_effort_target(
            gravity[:, self.arm_ids],
            joint_ids=self.arm_ids,
        )
        self.robot.set_joint_position_target(
            self.last_arm,
            joint_ids=self.arm_ids,
        )
        self._sync_gripper_modes()
        self.robot.set_joint_position_target(
            self.width[:, None].expand(-1, 2),
            joint_ids=self.finger_ids,
        )
        # 指定的是每根手指关节的驱动力，负方向闭合；非力控环境发零力，
        # 避免释放时残留闭合力。位置目标在零增益阶段不起驱动作用。
        finger_effort = torch.where(
            self.finger_force_mode, -self.grip_force, 0.0
        )
        self.robot.set_joint_effort_target(
            finger_effort[:, None].expand(-1, 2),
            joint_ids=self.finger_ids,
        )

    def step(self, dt, *, decision_only=False):
        """默认每物理步执行；decision_only 仅更新专家阶段，不发送控制。"""
        active = ~self.terminal
        self.active_env_steps += active.sum()
        self.elapsed += active * dt
        self.total_time += active * dt

        self._finish(active & (self.elapsed > 20.0), False, 1)

        waiting = self.state == self.WAIT
        speed = torch.maximum(
            self.cube.data.root_lin_vel_w.norm(dim=-1),
            self.plate.data.root_lin_vel_w.norm(dim=-1),
        )
        settled = (self.elapsed > 0.3) & (speed < 0.02)
        self.stable = torch.where(
            waiting,
            torch.where(settled, self.stable + dt, 0.0),
            self.stable,
        )

        ready = waiting & (self.stable > 0.3)
        self._prepare(ready)
        self._enter(ready, self.CLEAR)

        # 冻结本轮状态，防止一轮内连续跳过多个阶段。
        state = self.state.clone()

        if not decision_only:
            self._update_gripper_width(dt, state)

        reached = self._target_reached(*self._target_pose()) if decision_only else self._track(dt)
        motion = (
            (state >= self.CLEAR)
            & (state <= self.RETREAT)
        )
        self.stable = torch.where(
            motion,
            torch.where(reached, self.stable + dt, 0.0),
            self.stable,
        )

        self._advance_stages(state, dt)
        if not decision_only:
            self._send()

    def _update_gripper_width(self, dt, state):
        self.width = torch.where(
            state == self.CLOSE,
            (self.width - 0.05 * dt).clamp_min(0.0),
            self.width,
        )
        opening = (
            (state == self.OPEN)
            | (state == self.RETREAT)
            | (state == self.VERIFY)
        )
        self.width = torch.where(
            opening,
            (self.width + 0.05 * dt).clamp(max=0.03),
            self.width,
        )

    def _advance_stages(self, state, dt):
        grip_distance = (
            self.cube.data.root_pos_w - self._grasp_position()
        ).norm(dim=-1)

        carrying = (
            (state == self.TRANSFER) | (state == self.LOWER)
        )
        self._finish(carrying & (grip_distance > 0.04), False, 3)

        lift_arrived = (
            (state == self.LIFT) & (self.stable > 0.15)
        )
        rise = (
            self.cube.data.root_pos_w[:, 2] - self.cube_start[:, 2]
        )
        bad_grasp = (rise < 0.06) | (grip_distance > 0.05)
        self._finish(lift_arrived & bad_grasp, False, 4)

        # 沿用当前单环境控制器的底部中心近似判定。
        cube_offset = self.offset.new_tensor(
            [[0.0, 0.0, -0.0125]]
        ).expand(self.n, -1)
        plate_offset = self.offset.new_tensor(
            [[0.0, 0.0, -0.0178308]]
        ).expand(self.n, -1)

        cube_bottom = self.cube.data.root_pos_w + quat_apply(
            self.cube.data.root_quat_w, cube_offset
        )
        plate_bottom = self.plate.data.root_pos_w + quat_apply(
            self.plate.data.root_quat_w, plate_offset
        )
        difference = cube_bottom - plate_bottom
        placed = (
            (difference[:, :2].norm(dim=-1) < 0.03)
            & (difference[:, 2].abs() < 0.01)
        )

        verifying = state == self.VERIFY
        self.stable = torch.where(
            verifying,
            torch.where(placed, self.stable + dt, 0.0),
            self.stable,
        )
        self._finish(
            verifying & (self.stable > 0.5), True, 6
        )
        self._finish(
            verifying & (self.elapsed > 4.0), False, 5
        )

        # 这里循环的是固定数量的阶段，不是循环 N 个环境。
        transitions = (
            (self.CLEAR, self.ABOVE),
            (self.ABOVE, self.DESCEND),
            (self.DESCEND, self.CLOSE),
            (self.LIFT, self.TRANSFER),
            (self.TRANSFER, self.LOWER),
            (self.LOWER, self.OPEN),
            (self.RETREAT, self.VERIFY),
        )
        for old, new in transitions:
            mask = (
                (state == old)
                & (self.state == old)
                & (self.stable > 0.15)
            )
            self._enter(mask, new)

        self._enter(
            (state == self.CLOSE)
            & (self.state == self.CLOSE)
            & (self.elapsed > 1.5),
            self.LIFT,
        )
        self._enter(
            (state == self.OPEN)
            & (self.state == self.OPEN)
            & (self.elapsed > 1.5),
            self.RETREAT,
        )

    def outcomes(self):
        # 只在测试结束后搬到 CPU，不放进每步控制中。
        states = self.state.cpu().tolist()
        previous = self.last_state.cpu().tolist()
        reasons = self.reason.cpu().tolist()
        times = self.total_time.cpu().tolist()

        return [
            {
                "env_id": i,
                "state": self.STATE_NAMES[states[i]],
                "last_state": self.STATE_NAMES[previous[i]],
                "success": (
                    states[i] == self.DONE
                    if states[i] >= self.DONE else None
                ),
                "reason": self.REASON_NAMES[reasons[i]],
                "simulation_seconds": times[i],
                "grip_force_per_finger_n": self.grip_force,
            }
            for i in range(self.n)
        ]

    def summary(self):
        rows = self.outcomes()
        succeeded = sum(row["success"] is True for row in rows)
        failed = sum(row["success"] is False for row in rows)
        unfinished = self.n - succeeded - failed

        return {
            "grip_force_per_finger_n": self.grip_force,
            "succeeded": succeeded,
            "failed": failed,
            "unfinished": unfinished,
            "success_rate_over_all_envs": succeeded / self.n,
            "state_counts": {
                name: sum(row["state"] == name for row in rows)
                for name in self.STATE_NAMES
            },
            "active_env_steps": int(self.active_env_steps.item()),
        }
