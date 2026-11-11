"""Pure tensor versions of native substep motion measurements and reset stability."""

import torch

from simulators.cube_to_plate_batch_math import product, conjugate, rotvec


class BatchPoseMotion:
    def __init__(self, pose):
        self.previous = pose.clone()
        self.linear = torch.zeros_like(pose[:, :3])
        self.angular = torch.zeros_like(self.linear)

    def record(self, pose):
        linear = (pose[:, :3]-self.previous[:, :3])/.004
        angular = rotvec(product(pose[:, 3:], conjugate(self.previous[:, 3:])))/.004
        self.linear = torch.where((linear.square().sum(-1) > self.linear.square().sum(-1))[:, None], linear, self.linear)
        self.angular = torch.where((angular.square().sum(-1) > self.angular.square().sum(-1))[:, None], angular, self.angular)
        self.previous = pose.clone()


class BatchStability:
    def __init__(self, pose, table_height):
        self.first = pose.clone()
        self.window = None
        self.table_height = table_height
        self.unstable = torch.zeros(len(pose), device=pose.device, dtype=torch.bool)

    def update(self, step, pose):
        # Native check_layout_stability takes its first pose AFTER substep 0.
        if step == 0:
            self.first = pose.clone()
        angle = rotvec(product(pose[:, 3:], conjugate(self.first[:, 3:]))).norm(dim=-1)
        self.unstable |= ((pose[:, :3]-self.first[:, :3]).abs() > .04).any(-1) | (angle > torch.pi/6)
        self.unstable |= pose[:, 2] < self.table_height-.05
        if step == 200:
            self.window = pose.clone()
        if step >= 200:
            angle = rotvec(product(pose[:, 3:], conjugate(self.window[:, 3:]))).norm(dim=-1)
            self.unstable |= ((pose[:, :3]-self.window[:, :3]).abs() > .04).any(-1) | (angle > torch.pi/18)
        return self.unstable
