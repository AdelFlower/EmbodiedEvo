"""DP actor and privileged critic arithmetic with the M2 layouts on a torch device."""

import torch

from simulators.cube_to_plate_batch_math import grasp, unit
from simulators.cube_to_plate_batch_scoring import vertices


def policy_inputs(step, pad, opening, detection, pose, sample, plate):
    t = step.to(pad.dtype)
    angles = 2*torch.pi*t[:, None]*pad.new_tensor([1, 2, 4, 8, 16])/300
    clock = torch.cat((t[:, None]/300, torch.stack((angles.sin(), angles.cos()), -1).flatten(1)), -1)
    cmd = (pad-pad.new_tensor([.15, -.15, .85]))/pad.new_tensor([.2, .2, .1])
    heights = pad.new_tensor([.7775, .7828]).expand(len(pad), -1)[..., None]
    detected = torch.cat((detection, heights), -1)
    actor = torch.cat((clock, cmd, (detected[:, 0]-pad)/.05, (detected[:, 1]-pad)/.3, opening[:, None]), -1)
    observed_pad = grasp(pose)
    clear = vertices(sample['cube_position'], sample['cube_quaternion'])[:, :, 2].amin(-1)-.765
    privileged = torch.cat((clock, cmd, (sample['cube_position']-observed_pad)/.05,
                            (plate-observed_pad)/.3, clear[:, None]/.05, unit(sample['cube_quaternion']),
                            sample['cube_linear_velocity']/.2, opening[:, None], sample['fingers'][:, 1].to(pad.dtype),
                            sample['table'][:, None].to(pad.dtype), sample['plate'][:, None].to(pad.dtype)), -1)
    return actor.float(), torch.cat((actor, privileged), -1).float()
