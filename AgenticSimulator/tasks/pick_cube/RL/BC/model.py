"""Pick-cube 的行为克隆策略网络。"""

import torch
from torch import nn

from .schema import (
    ACTION_DIM,
    GRIP_CLOSE,
    GRIP_COMMAND_INDEX,
    GRIP_FORCE_INDEX,
    GRIP_OPEN,
    OBS_DIM,
)


# class BCPolicy(nn.Module):
#     """48 → 256 → 256 → 8 的 MLP。

#     输入：
#         已由外部预处理的观测，形状 (..., 48)。

#     forward 输出：
#         前 6 维：末端位姿增量的回归预测。
#         第 7 维：夹爪闭合 logit。
#         第 8 维：归一化力度的回归预测。

#     网络不负责观测归一化、动作单位换算或模拟器控制。
#     """

#     def __init__(
#         self,
#         hidden_dim: int = 256,
#         position_scale: float = 0.01,
#         rotation_scale: float = 0.1,
#     ):
#         super().__init__()

#         if position_scale <= 0 or rotation_scale <= 0:
#             raise ValueError("动作缩放尺度必须大于零")

#         # buffer 不参与梯度更新，但随模型保存、加载和移动设备。
#         self.register_buffer(
#             "pose_scale",
#             torch.tensor(
#                 [position_scale] * 3 + [rotation_scale] * 3,
#                 dtype=torch.float32,
#             ),
#         )

#         self.backbone = nn.Sequential(
#             nn.Linear(OBS_DIM, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#         )
#         self.action_head = nn.Linear(hidden_dim, ACTION_DIM)

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         """返回训练使用的原始预测，形状 (..., 8)。"""
#         if observations.ndim < 1 or observations.shape[-1] != OBS_DIM:
#             raise ValueError(
#                 f"观测最后一维应为 {OBS_DIM}，"
#                 f"实际形状为 {tuple(observations.shape)}"
#             )

#         features = self.backbone(observations)
#         return self.action_head(features)

#     @torch.no_grad()
#     def predict(self, observations: torch.Tensor) -> torch.Tensor:
#         """将原始预测转换为策略动作表示。

#         开合转换为 0/1，力度裁剪到 [-1, 1]。
#         前六维已还原为物理单位：平移为 m，旋转为 rad，调用方无需再次缩放。
#         """
#         predictions = self(observations)
#         actions = predictions.clone()
#         # 将前 6 维还原为 m 和 rad。
#         actions[..., :6] = predictions[..., :6] * self.pose_scale

#         # logit >= 0 等价于闭合概率 >= 0.5。
#         close = predictions[..., GRIP_COMMAND_INDEX] >= 0.0
#         actions[..., GRIP_COMMAND_INDEX] = torch.where(
#             close,
#             float(GRIP_CLOSE),
#             float(GRIP_OPEN),
#         )

#         actions[..., GRIP_FORCE_INDEX] = predictions[
#             ..., GRIP_FORCE_INDEX
#         ].clamp(-1.0, 1.0)

#         return actions
class GRUBCPolicy(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        recurrent_dim=128,
        position_scale=0.01,
        rotation_scale=0.1,
    ):
        super().__init__()
        if min(position_scale, rotation_scale) <= 0:
            raise ValueError("动作缩放尺度必须大于零")

        self.register_buffer(
            "pose_scale",
            torch.tensor(
                [position_scale] * 3 + [rotation_scale] * 3,
                dtype=torch.float32,
            ),
        )

        self.encoder = nn.Sequential(
            nn.Linear(OBS_DIM, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=recurrent_dim,
            num_layers=1,
            batch_first=True,
        )
        self.action_head = nn.Sequential(
            nn.Linear(recurrent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )

    def forward(self, observations, hidden=None):
        """
        observations: 已归一化的 [B, T, 48] 或单步 [B, 48]
        hidden: [1, B, recurrent_dim]；None 表示从零记忆开始
        返回：原始预测、更新后的 hidden
        """
        if observations.ndim not in (2, 3):
            raise ValueError("观测必须为 [B, 48] 或 [B, T, 48]")
        if observations.shape[-1] != OBS_DIM:
            raise ValueError(f"观测最后一维必须为 {OBS_DIM}")

        single_step = observations.ndim == 2
        if single_step:
            observations = observations.unsqueeze(1)

        features = self.encoder(observations)
        features, next_hidden = self.gru(features, hidden)
        predictions = self.action_head(features)

        if single_step:
            predictions = predictions[:, 0]

        return predictions, next_hidden

    @torch.no_grad()
    def predict(self, observations, hidden=None):
        predictions, next_hidden = self(observations, hidden)

        actions = predictions.clone()
        actions[..., :6] *= self.pose_scale
        actions[..., GRIP_COMMAND_INDEX] = (
            predictions[..., GRIP_COMMAND_INDEX] >= 0
        ).to(actions.dtype)
        actions[..., GRIP_FORCE_INDEX] = predictions[
            ..., GRIP_FORCE_INDEX
        ].clamp(-1, 1)

        return actions, next_hidden