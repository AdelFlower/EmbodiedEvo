"""Reuse BC's pose/IK executor without advancing its expert state machine."""

import torch
from isaaclab.utils.math import quat_apply, compute_pose_error

from ..BC.control import BCExpertController
from ..BC.schema import INITIAL_PREVIOUS_ACTION


class GuidanceController(BCExpertController):
    """Training-only expert. decide() computes actions but never writes motors."""

    def feedback_labels(self, observations, seen_lift, initial_cube_z):
        """Training oracle following current objects and actual finger state."""
        hand, cube, plate = observations[:, 16:23], observations[:, 23:26], observations[:, 33:36]
        grip = hand[:, :3] + quat_apply(hand[:, 3:], self.offset)
        gap = observations[:, 14:16].sum(-1)
        actual_closed = observations[:, 46].bool()
        distance = (cube - grip).norm(dim=-1)
        finger_speed = self.robot.data.joint_vel[:, self.finger_ids].abs().sum(-1)
        contact = actual_closed & (distance < 0.05) & (gap > 0.015) & (gap < 0.045)
        # Force-controlled contact can keep the finger joints vibrating. Once
        # the object actually rises, that motion must not erase grasp evidence.
        lifted = seen_lift | (cube[:, 2] > initial_cube_z + 0.005)
        held = contact & ((finger_speed < 0.02) | lifted)
        plate_distance = (cube[:, :2] - plate[:, :2]).norm(dim=-1)
        above_plate = plate_distance < 0.025
        release = seen_lift & (plate_distance < 0.015) & (cube[:, 2] < plate[:, 2] + 0.02)
        w, x, y, z = hand[:, 3:].unbind(-1)
        hand_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        w, x, y, z = observations[:, 26:30].unbind(-1)
        cube_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        cube_yaw = torch.remainder(cube_yaw + torch.pi / 4, torch.pi / 2) - torch.pi / 4
        yaw = torch.where(held, hand_yaw, cube_yaw)
        zeros = torch.zeros_like(yaw)
        target_quat = torch.stack((zeros, (yaw / 2).cos(), (yaw / 2).sin(), zeros), -1)
        inverse = torch.cat((hand[:, 3:4], -hand[:, 4:]), -1)
        local_offset = quat_apply(inverse, cube - grip)
        target_cube = cube.clone()
        aligned = (cube[:, :2] - grip[:, :2]).norm(dim=-1) < 0.015
        target_cube[:, 2] += torch.where(aligned, 0.0, 0.15)
        lifting = held & ~seen_lift
        target_cube[lifting, 2] = torch.maximum(initial_cube_z, plate[:, 2])[lifting] + 0.15
        carrying = held & seen_lift
        target_cube[carrying] = plate[carrying]
        target_cube[carrying, 2] += torch.where(above_plate, 0.01, 0.15)[carrying]
        offset = self.offset + torch.where(held[:, None], local_offset, 0.0)
        target_hand = target_cube - quat_apply(target_quat, offset)
        dp, dr = compute_pose_error(hand[:, :3], hand[:, 3:], target_hand, target_quat,
                                    rot_error_type="axis_angle")
        # Continue lowering while actually closed; hold only after opening.
        # Real task success, not assumed teacher execution, ends these labels.
        pose = torch.where((release & ~actual_closed)[:, None], 0.0, torch.cat((dp, dr), -1))
        closing = (distance < 0.015) & ((gap > 0.03) | (actual_closed & (gap > 0.015)))
        closed = (held | closing) & ~release
        labels = torch.cat((pose, closed[:, None].float(),
                            torch.where(closed, 0.0, -1.0)[:, None]), -1)
        return labels

    def _target_pose(self):
        target_hand, target_quat, moving = super()._target_pose()
        # Equal finger forces need not center a cube between independent joints.
        # Center the actual carried cube, rather than relying on the old opening
        # position drive to forcibly recenter it at release.
        hand_quat = self.robot.data.body_state_w[:, self.hand_id, 3:7]
        inverse = torch.cat((hand_quat[:, :1], -hand_quat[:, 1:]), dim=-1)
        local_offset = quat_apply(inverse, self.cube.data.root_pos_w - self._grasp_position())
        world_offset = quat_apply(target_quat, local_offset)
        carrying = (self.state == self.TRANSFER) | (self.state == self.LOWER)
        target_hand = target_hand - torch.where(carrying[:, None], world_offset, 0.0)
        return target_hand, target_quat, moving

    def reset_memory(self, env_ids):
        self.state[env_ids] = self.WAIT
        self.last_state[env_ids] = self.WAIT
        self.reason[env_ids] = 0
        self.elapsed[env_ids] = 0
        self.stable[env_ids] = 0
        self.total_time[env_ids] = 0
        self.cube_start[env_ids] = 0
        self.goals[env_ids] = 0
        self.goal_quat[env_ids] = self.goal_quat.new_tensor([1., 0., 0., 0.])
        self.width[env_ids] = 0.04
        self.last_arm[env_ids] = self.robot.data.default_joint_pos[env_ids][:, self.arm_ids]
        self.finger_force_mode[env_ids] = False
        self._gripper_modes_dirty.fill_(True)
        self._has_closed[env_ids] = False
        self._tracking[env_ids] = False
        self.previous_action[env_ids] = self.previous_action.new_tensor(INITIAL_PREVIOUS_ACTION)
        self.ik.reset(env_ids)


class PoseExecutor(GuidanceController):
    def __init__(self, robot, cube, plate, device):
        super().__init__(robot, cube, plate, device)
        self.opening_widths = robot.data.default_joint_pos[:, self.finger_ids].clone()
        self.opening_limits = torch.full_like(self.opening_widths, 0.04)

    def reset(self, env_ids):
        self.reset_memory(env_ids)
        self.opening_widths[env_ids] = self.robot.data.default_joint_pos[env_ids][:, self.finger_ids]
        self.opening_limits[env_ids] = 0.04
        # The next set_action replaces the latched pose before physics resumes.
        self._sync_gripper_modes()

    def set_action(self, actions, *, tracking=None):
        opening = self.finger_force_mode & (actions[:, 6] == 0) & ~self.terminal
        # Force-controlled fingers can be asymmetric around the cube. Starting
        # both position targets at zero squeezes/recenters it during "open".
        # Each finger instead starts at its measured position and moves out.
        measured = self.robot.data.joint_pos[:, self.finger_ids].clamp(0.0, 0.04)
        self.opening_widths[opening] = measured[opening]
        self.opening_limits[opening] = measured[opening].clamp_min(0.03)
        super().set_action(actions, tracking=tracking)

    def physics_step(self, physics_dt):
        opening = ~self.finger_force_mode & ~self.terminal
        target = torch.minimum(self.opening_widths + 0.05 * physics_dt, self.opening_limits)
        self.opening_widths = torch.where(opening[:, None], target, self.opening_widths)
        super().physics_step(physics_dt)

    def _send(self, grip_force=None):
        super()._send(grip_force=grip_force)
        target = torch.where(self.finger_force_mode[:, None], self.width[:, None], self.opening_widths)
        self.robot.set_joint_position_target(target, joint_ids=self.finger_ids)

    def _target_reached(self, target_hand, target_quat, moving):
        # _track_pose's return value is unused; no expert stage/tolerance needed.
        return torch.zeros_like(moving)
