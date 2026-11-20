"""目标环境从零训练时可选的连续抬升奖励。

在原训练命令末尾加上这个 Hydra 参数即可启用：
env.rewards.lifting_object.func=AgenticSimulator.scripts.lift_cube_dense_reward:lift_with_progress
"""

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab_arena.mdp import object_is_lifted

from AgenticSimulator.scenes.cube import TABLE_HEIGHT


# 方块边长 0.025 m，平放在桌上时中心比桌面高半个边长。
REST_HEIGHT = TABLE_HEIGHT + 0.0125
SETTLE_TOLERANCE = 0.0001  # 0.1 mm，容纳接触求解时的高度误差
PROGRESS_WEIGHT = 5.0


class lift_with_progress(ManagerTermBase):
    """保留原生抬升奖励，落桌后按抬升进度叠加引导奖励。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.reached_table = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.reached_table.zero_()
        else:
            self.reached_table[env_ids] = False

    def __call__(self, env, minimal_height, object_cfg):
        height = env.scene[object_cfg.name].data.root_pos_w[:, 2]
        near_table = height <= REST_HEIGHT + SETTLE_TOLERANCE
        self.reached_table.logical_or_(near_table)

        progress = ((height - REST_HEIGHT) / (minimal_height - REST_HEIGHT)).clamp(0.0, 1.0)
        bonus = torch.where(self.reached_table & ~near_table, progress, 0.0)
        native = object_is_lifted(env, minimal_height, object_cfg)
        # RewardManager 随后统一乘以 weight 和 dt，原生项保留其权重 15。
        return native + PROGRESS_WEIGHT * bonus / self.cfg.weight
