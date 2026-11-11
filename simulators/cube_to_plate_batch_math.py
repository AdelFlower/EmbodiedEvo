"""Simulator-free, scalar-first quaternion and fixed task control arithmetic."""

import torch

from integrations.friction_rl.control import ACTION_LOW, ACTION_HIGH, REFERENCE_QUATERNION
from integrations.friction_rl.network import APPROACH_STEPS, STEP, PAD_FLOOR, F_MID, F_HALF
from integrations.friction_rl.primitive import GRASP_OFFSET


def unit(q):
    q = q / q.norm(dim=-1, keepdim=True)
    return torch.where(q[..., :1] < 0, -q, q)


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def product(a, b):
    a, b = torch.broadcast_tensors(a, b)
    return torch.cat((a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True),
                      a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:]
                      + torch.linalg.cross(a[..., 1:], b[..., 1:])), -1)


def matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def rotvec(q):
    q = unit(q)
    length = q[..., 1:].norm(dim=-1, keepdim=True)
    scale = 2 * torch.atan2(length, q[..., :1]) / length.clamp_min(1e-12)
    return q[..., 1:] * torch.where(length > 1e-12, scale, torch.full_like(scale, 2))


def axis_quaternion(axis, angle):
    return torch.cat((torch.cos(angle[..., None]/2), axis * torch.sin(angle[..., None]/2)), -1)


def grasp(pose):
    return pose[..., :3] + GRASP_OFFSET * matrix(pose[..., 3:])[..., :, 0]


def slerp(a, b, fraction):
    a, b = torch.broadcast_tensors(a, b)
    dot = (a*b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    dot = dot.abs().clamp(max=1)
    angle = torch.acos(dot)
    curved = (torch.sin((1-fraction)*angle)*a + torch.sin(fraction*angle)*b) / torch.sin(angle).clamp_min(1e-12)
    return torch.where(dot > .9995, unit(a + fraction*(b-a)), curved)


class BatchDecoder:
    def __init__(self, poses):
        self.commanded_pose = poses.clone()
        self.pad_cmd = grasp(poses)
        self.q0 = unit(poses[:, 3:])

    def reset(self, ids, poses):
        self.commanded_pose[ids] = poses
        self.pad_cmd[ids] = grasp(poses)
        self.q0[ids] = unit(poses[:, 3:])

    def step(self, t, actions, active):
        a = actions.clamp(-1, 1)
        tau = ((t.to(a.dtype)+1) / APPROACH_STEPS).clamp(max=1)[:, None]
        q = slerp(self.q0, a.new_tensor(REFERENCE_QUATERNION), 10*tau**3-15*tau**4+6*tau**5)
        point = self.pad_cmd + STEP*a[:, :3]
        point[:, 2].clamp_(min=PAD_FLOOR)
        ee = (point-GRASP_OFFSET*matrix(q)[:, :, 0]).clamp(a.new_tensor(ACTION_LOW), a.new_tensor(ACTION_HIGH))
        pose = torch.cat((ee, q), -1)
        self.commanded_pose = torch.where(active[:, None], pose, self.commanded_pose)
        self.pad_cmd = grasp(self.commanded_pose)
        closed = a[:, 3] > 0
        force = torch.where(closed, F_MID+F_HALF*a[:, 4], 0)
        # Retain the native transport round trip: encode_action emits float32.
        delta = product(unit(q), conjugate(a.new_tensor(REFERENCE_QUATERNION)))
        native = torch.cat((2*(ee-a.new_tensor(ACTION_LOW))/a.new_tensor(ACTION_HIGH-ACTION_LOW)-1,
                            rotvec(delta)/torch.pi, torch.where(closed, -1., 1.)[:, None]), -1).clamp(-1, 1).float()
        xyz = native[:, :3].to(a.dtype)
        rv = native[:, 3:6].to(a.dtype)*torch.pi
        angle = rv.norm(dim=-1, keepdim=True)
        dq = torch.cat((torch.cos(angle/2), rv*torch.sinc(angle/(2*torch.pi))/2), -1)
        applied = torch.cat((a.new_tensor(ACTION_LOW)+(xyz+1)*a.new_tensor(ACTION_HIGH-ACTION_LOW)/2,
                             unit(product(dq, a.new_tensor(REFERENCE_QUATERNION)))), -1)
        return applied, force, closed, native


def interpolation(current, target):
    """process_control_info: alpha=1/9,...,8/9,1,1 (not linspace(0,1))."""
    alpha = current.new_tensor([1/9, 2/9, 3/9, 4/9, 5/9, 6/9, 7/9, 8/9, 1, 1])
    return current[None] + alpha[:, None, None]*(target-current)[None]


def gripper_profile(current, closed):
    """Primary joint mapping and mimic [joint8,1,0], gripper_scale=1."""
    target = torch.where(closed, -.01, .044).to(current.dtype)[:, None]
    primary = interpolation(current[:, :1], target).clamp(-.01, .044)
    return primary.expand(-1, -1, 2)


def pad_normals(finger_quaternions):
    """link7 inward = -Y, link8 inward = +Y in their URDF link frames."""
    return matrix(finger_quaternions)[..., :, 1] * finger_quaternions.new_tensor([-1, 1])[None, :, None]


def gripper_drive_target(current, profiled):
    """MetaControl's 20%-of-span slew limit applied to the primary encoder."""
    primary = current[:, :1] + (profiled[:, :1]-current[:, :1]).clamp(-.054*.2, .054*.2)
    return primary.expand(-1, 2)


def arm_control_profile(current, solution, previous_target, success, active):
    target = torch.where((active & success)[:, None], solution, previous_target)
    profile = interpolation(current, target)
    profile = torch.where((active & success)[None, :, None], profile, target[None])
    return target, profile
