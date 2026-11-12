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


class PickCubeController:
    """只适用于当前单环境、固定底座 Franka 的抓放控制器。"""

    def __init__(self, robot, cube, plate, device):
        self.robot = robot
        self.cube = cube
        self.plate = plate
        self.device = device

        if not robot.is_fixed_base:
            raise ValueError("这个控制器要求机器人底座固定")

        # 明确按 joint1～joint7 排列机械臂关节。
        self.arm_ids = [
            robot.find_joints(f"panda_joint{i}")[0][0]
            for i in range(1, 8)
        ]
        self.finger_ids = [
            robot.find_joints(f"panda_finger_joint{i}")[0][0]
            for i in (1, 2)
        ]
        self.hand_id = robot.find_bodies("panda_hand")[0][0]

        # 固定底座在 PhysX Jacobian 中不占一行。
        self.jacobian_id = self.hand_id - 1

        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
            ),
            num_envs=1,
            device=device,
        )

        # panda_hand 到两指抓取中心的局部坐标偏移。
        # 这是初始标定值，需要在当前 USD 模型上确认。
        self.grasp_offset = torch.tensor(
            [[0.0, 0.0, 0.1034]],
            device=device,
        )

        self.state = "wait"
        self.elapsed = 0.0
        self.stable_time = 0.0
        self.result = None

        self.width = 0.04
        self.last_arm = robot.data.default_joint_pos[
            :, self.arm_ids
        ].clone()

        self.targets = {}
        self.goal_quat = None
        self.cube_start = None

    def _enter(self, state):
        self.state = state
        self.elapsed = 0.0
        self.stable_time = 0.0
        print(f"[pick_cube] -> {state}", flush=True)

    def _finish(self, success, reason):
        # 结束后保持当前关节位置，不继续追赶原来的运动目标。
        self.last_arm = self.robot.data.joint_pos[
            :, self.arm_ids
        ].clone()

        self.result = {
            "success": success,
            "reason": reason,
            "last_state": self.state,
        }
        self.state = "done" if success else "failed"
        print(f"[pick_cube] {self.result}", flush=True)

    def _send_targets(self):
        # 在位置控制之外，叠加机械臂自身的重力补偿力矩。
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
        fingers = torch.full(
            (1, 2),
            self.width,
            device=self.device,
        )
        self.robot.set_joint_position_target(
            fingers,
            joint_ids=self.finger_ids,
        )

    def _grasp_position(self):
        hand = self.robot.data.body_state_w[:, self.hand_id, :7]
        return hand[:, :3] + quat_apply(
            hand[:, 3:7], self.grasp_offset
        )

    def _prepare_targets(self):
        cube = self.cube.data.root_pos_w.clone()
        plate = self.plate.data.root_pos_w.clone()
        self.cube_start = cube.clone()

        # 根据方块朝向选择一个向下抓取姿态。
        # 正方体绕 Z 轴转 90° 后等价，缩短腕部转动角度。
        w, x, y, z = self.cube.data.root_quat_w[0].tolist()
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4

        # roll=180°，使手掌局部 +Z 朝向世界 -Z。
        self.goal_quat = torch.tensor(
            [[0.0, math.cos(yaw / 2), math.sin(yaw / 2), 0.0]],
            device=self.device,
        )

        safe_z = max(cube[0, 2].item(), plate[0, 2].item()) + 0.15

        def at_height(position, height):
            target = position.clone()
            target[:, 2] = height
            return target

        self.targets = {
            "clearance": at_height(self._grasp_position(), safe_z),
            "above_cube": at_height(cube, safe_z),
            "descend": cube.clone(),
            "lift": at_height(cube, safe_z),
            "transfer": at_height(plate, safe_z),
            # 在盘子上方低处释放，留出指尖与盘子的间隙。
            "lower": at_height(plate, plate[0, 2].item() + 0.01),
            "retreat": at_height(plate, safe_z),
        }
        self.targets["close"] = self.targets["descend"].clone()
        self.targets["open"] = self.targets["lower"].clone()
        self.targets["verify"] = self.targets["retreat"].clone()

    def _track(self, grasp_target, dt):
        """控制 panda_hand 原点，让两指抓取中心到达指定世界坐标。"""
        robot = self.robot
        hand = robot.data.body_state_w[:, self.hand_id, :7]
        root = robot.data.root_state_w[:, :7]

        # 抓取中心目标 -> 手掌原点目标。
        hand_target = grasp_target - quat_apply(
            self.goal_quat, self.grasp_offset
        )

        # 当前位姿与目标位姿统一转换到机器人根坐标系。
        current_pos, current_quat = subtract_frame_transforms(
            root[:, :3], root[:, 3:7],
            hand[:, :3], hand[:, 3:7],
        )
        target_pos, target_quat = subtract_frame_transforms(
            root[:, :3], root[:, 3:7],
            hand_target, self.goal_quat,
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

        # PhysX 的世界坐标 Jacobian -> 根坐标系。
        rotation = matrix_from_quat(root[:, 3:7]).transpose(1, 2)
        jacobian[:, :3, :] = rotation @ jacobian[:, :3, :]
        jacobian[:, 3:, :] = rotation @ jacobian[:, 3:, :]

        joint_pos = robot.data.joint_pos[:, self.arm_ids]

        # 计算当前手掌位姿到目标位姿的增量。
        delta_pos, delta_rot = compute_pose_error(
            current_pos,
            current_quat,
            target_pos,
            target_quat,
            rot_error_type="axis_angle",
        )

        # 相对模式：输入 6 维增量，并提供当前位姿。
        self.ik.set_command(
            torch.cat((delta_pos, delta_rot), dim=-1),
            ee_pos=current_pos,
            ee_quat=current_quat,
        )
        desired = self.ik.compute(
            current_pos, current_quat, jacobian, joint_pos
        )

        if not torch.isfinite(desired).all().item():
            self._finish(False, "IK produced non-finite joint targets")
            return False

        # # 限制每一步的关节目标变化，最大约 0.8 rad/s。
        # delta = desired - joint_pos
        # factor = min(
        #     1.0,
        #     0.8 * dt / max(delta.abs().max().item(), 1e-8),
        # )
        # desired = joint_pos + delta * factor
        # 限制目标相对于上一轮目标的变化。

        # 让控制参考逐渐移动，而不是始终紧贴实际关节位置。
        delta = desired - self.last_arm

        max_step = 0.8 * dt
        factor = min(
            1.0,
            max_step / max(delta.abs().max().item(), 1e-8),
        )

        desired = self.last_arm + delta * factor

        limits = robot.data.soft_joint_pos_limits[:, self.arm_ids]
        self.last_arm = torch.maximum(
            torch.minimum(desired, limits[..., 1]),
            limits[..., 0],
        )

        position_error, rotation_error = compute_pose_error(
            hand[:, :3],
            hand[:, 3:7],
            hand_target,
            self.goal_quat,
        )

        # debug_key = (self.state, int(self.elapsed))

        # if getattr(self, "_last_debug_key", None) != debug_key:
        #     self._last_debug_key = debug_key

        #     pos_mm = position_error.norm().item() * 1000
        #     rot_deg = math.degrees(rotation_error.norm().item())
        #     joint_error = (
        #         self.last_arm - joint_pos
        #     ).abs().max().item()
        #     limit_margin = torch.minimum(
        #         joint_pos - limits[..., 0],
        #         limits[..., 1] - joint_pos,
        #     ).min().item()

        #     print(
        #         f"[debug] state={self.state} "
        #         f"t={self.elapsed:.1f}s "
        #         f"pos_error={pos_mm:.1f}mm "
        #         f"rot_error={rot_deg:.1f}deg "
        #         f"joint_error={joint_error:.4f}rad "
        #         f"limit_margin={limit_margin:.4f}rad\n"
        #         f"  hand={hand[0, :3].tolist()}\n"
        #         f"  hand_target={hand_target[0].tolist()}",
        #         flush=True,
        #     )


        tolerance = 0.003 if self.state == "descend" else 0.008
        return (
            position_error.norm().item() < tolerance
            and rotation_error.norm().item() < 0.10
        )

    def step(self, dt):
        """每个物理步进前调用一次。"""
        if self.result is not None:
            self._send_targets()
            return

        self.elapsed += dt

        if self.elapsed > 20.0:
            self._finish(False, f"timeout in {self.state}")
            self._send_targets()
            return

        if self.state == "wait":
            # 等待物体落稳，同时保持机器人初始姿态。
            speed = max(
                self.cube.data.root_lin_vel_w.norm().item(),
                self.plate.data.root_lin_vel_w.norm().item(),
            )
            if self.elapsed > 0.5 and speed < 0.02:
                self.stable_time += dt
            else:
                self.stable_time = 0.0

            if self.stable_time > 0.3:
                self._prepare_targets()
                self._enter("clearance")

            self._send_targets()
            return

        # 逐渐开合夹爪；width 是每根手指的关节位置。
        if self.state == "close":
            self.width = max(0.0, self.width - 0.05 * dt)
        elif self.state in ("open", "retreat", "verify"):
            self.width = min(0.03, self.width + 0.05 * dt)

        reached = self._track(self.targets[self.state], dt)

        if self.result is not None:
            self._send_targets()
            return

        if self.state != "verify":
            self.stable_time = self.stable_time + dt if reached else 0.0

        # 搬运中检查方块有没有从夹爪中掉落。
        if self.state in ("transfer", "lower"):
            distance = (
                self.cube.data.root_pos_w - self._grasp_position()
            ).norm().item()
            if distance > 0.04:
                self._finish(False, "cube slipped from gripper")
                self._send_targets()
                return

        if self.state == "close":
            if self.elapsed > 1.5:
                self._enter("lift")

        elif self.state == "open":
            if self.elapsed > 1.5:
                self._enter("retreat")

        elif self.state == "verify":
            # 当前资产底部中心的近似局部坐标，单位为米。
            cube_offset = self.cube.data.root_pos_w.new_tensor(
                [[0.0, 0.0, -0.0125]]
            )
            plate_offset = self.plate.data.root_pos_w.new_tensor(
                [[0.0, 0.0, -0.0178308]]
            )
            cube_bottom = self.cube.data.root_pos_w + quat_apply(
                self.cube.data.root_quat_w, cube_offset
            )
            plate_bottom = self.plate.data.root_pos_w + quat_apply(
                self.plate.data.root_quat_w, plate_offset
            )
            bottom_delta = cube_bottom - plate_bottom
            xy_distance = bottom_delta[:, :2].norm().item()
            height_error = bottom_delta[:, 2].abs().item()
            placed = xy_distance < 0.03 and height_error < 0.01

            self.stable_time = (
                self.stable_time + dt if placed else 0.0
            )
            if placed and self.stable_time > 0.5:
                self._finish(
                    True, "bottom centers: XY < 3 cm and height difference < 1 cm"
                )
            elif self.elapsed > 4.0:
                self._finish(False, "placement verification failed")

        elif self.stable_time > 0.15:
            if self.state == "lift":
                rise = (
                    self.cube.data.root_pos_w[:, 2]
                    - self.cube_start[:, 2]
                ).item()
                distance = (
                    self.cube.data.root_pos_w
                    - self._grasp_position()
                ).norm().item()

                if rise < 0.06 or distance > 0.05:
                    self._finish(False, "grasp verification failed")
                    self._send_targets()
                    return

            next_state = {
                "clearance": "above_cube",
                "above_cube": "descend",
                "descend": "close",
                "lift": "transfer",
                "transfer": "lower",
                "lower": "open",
                "retreat": "verify",
            }
            self._enter(next_state[self.state])

        self._send_targets()
