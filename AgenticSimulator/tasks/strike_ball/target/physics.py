"""击球任务的固定目标物理，由宿主在启动 Isaac Sim 后加载。"""

import math

import torch
from isaaclab.utils.math import quat_rotate_inverse

from AgenticSimulator.tasks.strike_ball.scene import BALL_RADIUS


AIR_DENSITY = 1.225  # kg/m³
DRAG_COEFFICIENT = 0.47
DRAG_FACTOR = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * math.pi * BALL_RADIUS**2


def configure(env_cfg):
    """创建环境时沿用击球任务的物体属性。"""
    pass


def before_step(env, dt):
    """每个物理子步保留未击中球的支撑力，对已击中的球施加空气阻力。"""
    velocity_w = env.ball.data.root_lin_vel_w
    speed = velocity_w.norm(dim=-1, keepdim=True)
    drag_w = -DRAG_FACTOR * speed * velocity_w
    force_w = torch.where(env.hit[:, None], drag_w, env.support[:, 0])
    # Isaac Lab 的外力接口使用球体坐标系。
    force_b = quat_rotate_inverse(env.ball.data.root_quat_w, force_w)
    env.ball.set_external_force_and_torque(force_b[:, None, :], env.zero_torque)
