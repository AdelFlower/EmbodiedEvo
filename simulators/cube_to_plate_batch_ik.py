"""URDF FK, geometric Jacobian and one warm-started batched DLS solve; no Kit imports."""

from pathlib import Path
import math
import xml.etree.ElementTree as ET

import torch

from simulators.cube_to_plate_batch_math import axis_quaternion, matrix, product, conjugate, rotvec, unit

URDF = Path(__file__).resolve().parents[1] / 'vendor/robodojo/Assets/Robots/x5/X5A.urdf'
ARM_NAMES = tuple(f'joint{i}' for i in range(1, 7))
POSITION_TOLERANCE = .001
ORIENTATION_TOLERANCE = .02
IK_ITERATIONS = 64


def _rpy_quaternion(rpy):
    r, p, y = (v/2 for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]


class ArmKinematics:
    def __init__(self, urdf=URDF, *, device='cpu', dtype=torch.float32):
        tree = ET.parse(urdf)
        children = {j.find('child').get('link'): j for j in tree.findall('joint')}
        chain, link = [], 'link6'
        while link != 'base_link':
            j = children[link]
            chain.append(j)
            link = j.find('parent').get('link')
        chain.reverse()
        if tuple(j.get('name') for j in chain) != ARM_NAMES or any(j.get('type') != 'revolute' for j in chain):
            raise ValueError('X5 base_link -> link6 must contain the six declared revolute joints')
        def tensor(v):
            return torch.tensor(v, device=device, dtype=dtype)
        self.xyz = tensor([[float(v) for v in j.find('origin').get('xyz').split()] for j in chain])
        self.origins = tensor([_rpy_quaternion([float(v) for v in j.find('origin').get('rpy').split()]) for j in chain])
        self.axes = tensor([[float(v) for v in j.find('axis').get('xyz').split()] for j in chain])
        self.lower = tensor([float(j.find('limit').get('lower')) for j in chain])
        self.upper = tensor([float(j.find('limit').get('upper')) for j in chain])

    def forward(self, joints, root):
        """root and result are env-local link frames, not COM frames."""
        p, q = root[:, :3], unit(root[:, 3:])
        points, axes = [], []
        for i in range(6):
            p = p + (matrix(q) @ self.xyz[i, :, None]).squeeze(-1)
            q = product(q, self.origins[i])
            points.append(p)
            axes.append((matrix(q) @ self.axes[i, :, None]).squeeze(-1))
            q = product(q, axis_quaternion(self.axes[i].expand(joints.shape[0], -1), joints[:, i]))
        axes = torch.stack(axes, -1)
        displacement = p[:, :, None] - torch.stack(points, -1)
        jacobian = torch.cat((torch.linalg.cross(axes, displacement, dim=1), axes), dim=1)
        return torch.cat((p, unit(q)), -1), jacobian

    def solve(self, current, target, root):
        """64 DLS iterations, lambda=.01, per-iteration joint increment limited to .2 rad.

        Success uses the legacy configured pose thresholds. There is no collision
        optimizer or seed search here; controller parity must measure that difference.
        """
        q = current.clone()
        eye = torch.eye(6, dtype=q.dtype, device=q.device)[None]
        for _ in range(IK_ITERATIONS):
            pose, jac = self.forward(q, root)
            err = torch.cat((target[:, :3]-pose[:, :3], rotvec(product(target[:, 3:], conjugate(pose[:, 3:])))), -1)
            dq = (jac.transpose(1, 2) @ torch.linalg.solve(jac @ jac.transpose(1, 2)+.01**2*eye, err[:, :, None])).squeeze(-1)
            converged = (err[:, :3].norm(dim=-1) < 1e-5) & (err[:, 3:].norm(dim=-1) < 1e-4)
            q = torch.where(converged[:, None], q, (q+dq.clamp(-.2, .2)).clamp(self.lower, self.upper))
        pose, _ = self.forward(q, root)
        position_error = (pose[:, :3]-target[:, :3]).norm(dim=-1)
        orientation_error = rotvec(product(target[:, 3:], conjugate(pose[:, 3:]))).norm(dim=-1)
        success = (position_error < POSITION_TOLERANCE) & (orientation_error < ORIENTATION_TOLERANCE)
        return q, success, position_error, orientation_error
