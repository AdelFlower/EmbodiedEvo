"""Batched PlacementScorer and network episode termination; tensors stay on device."""

from itertools import product as cartesian

import torch

from integrations.friction_rl.episode import PLATE_REST_RADIUS, REST_STEPS
from integrations.friction_rl.network import HORIZON, NO_LIFT_STEP
from simulators.dynamics_scoring import Criteria
from simulators.cube_to_plate_batch_math import matrix, grasp

STOP_REASONS = (None, 'success', 'native_end', 'dropped', 'no_lift', 'timeout')


def vertices(position, quaternion):
    corners = position.new_tensor(list(cartesian((-.0125, .0125), repeat=3)))
    return corners[None] @ matrix(quaternion).transpose(-1, -2) + position[:, None]


def hand_offset(pose, cube):
    return (matrix(pose[:, 3:]).transpose(1, 2) @ (cube-grasp(pose))[:, :, None]).squeeze(-1)


class BatchPlacementScorer:
    def __init__(self, plate):
        self.plate = plate.clone()
        n, device = len(plate), plate.device
        self.grasp_arm = torch.full((n,), -1, device=device, dtype=torch.long)
        self.transport_arm = self.grasp_arm.clone()
        self.start = torch.zeros_like(plate)
        self.count = torch.zeros_like(self.grasp_arm)
        self.stable_count = self.count.clone()
        self.last_step = torch.full_like(self.count, -1)
        self.transported = torch.zeros(n, device=device, dtype=torch.bool)
        self.released = self.transported.clone()
        self.success = self.transported.clone()

    def reset(self, ids, plate):
        self.plate[ids] = plate
        for name in ('grasp_arm', 'transport_arm', 'last_step'):
            getattr(self, name)[ids] = -1
        for name in ('start', 'count', 'stable_count', 'transported', 'released', 'success'):
            getattr(self, name)[ids] = 0

    def update(self, s, active):
        c = Criteria()
        update = active & (s['step'] != self.last_step) & ~self.success
        self.last_step = torch.where(update, s['step'], self.last_step)
        v = vertices(s['cube_position'], s['cube_quaternion'])
        lowest = v[:, :, 2].amin(-1)
        inside = ((v[:, :, :2]-self.plate[:, None, :2]).norm(dim=-1) <= c.plate_inner_radius).all(-1)
        both = s['fingers'].all(-1)  # [N, left/right]
        prior = self.grasp_arm.clamp_min(0)
        prior_present = both.gather(1, prior[:, None]).squeeze(1) & (self.grasp_arm >= 0)
        first = torch.where(both[:, 0], 0, torch.where(both[:, 1], 1, -1))
        arm = torch.where(prior_present, self.grasp_arm, first)
        clear = (lowest >= c.table_height+c.lift_clearance) & ~s['table'] & ~s['plate']
        counting = (arm >= 0) & clear
        restart = (arm != self.grasp_arm) | (self.count == 0)
        self.start = torch.where((update & counting & restart)[:, None], s['cube_position'], self.start)
        count = torch.where(counting, torch.where(restart, 1, self.count+1), 0)
        self.count = torch.where(update, count, self.count)
        self.grasp_arm = torch.where(update, arm, self.grasp_arm)
        moved = (s['cube_position'][:, :2]-self.start[:, :2]).norm(dim=-1) >= c.transport_distance
        carried = update & counting & moved & (self.count >= c.grasp_steps)
        self.transported |= carried
        self.transport_arm = torch.where(carried, arm, self.transport_arm)
        transport_contacts = s['fingers'].any(-1).gather(1, self.transport_arm.clamp_min(0)[:, None]).squeeze(1)
        release_check = update & self.transported & ~self.released & ~transport_contacts
        release_ok = inside & ~s['table'] & (lowest >= c.plate_bottom-.002) & (lowest <= c.plate_top+c.release_height)
        self.released |= release_check & release_ok
        cancel = update & (s['table'] | (release_check & ~release_ok))
        self.transported &= ~cancel
        self.transport_arm = torch.where(cancel, -1, self.transport_arm)
        self.released &= ~(update & s['table'])
        idx = self.transport_arm.clamp_min(0)
        opening = s['opening'].gather(1, idx[:, None]).squeeze(1)
        pad = s['finger_position'].gather(1, idx[:, None, None].expand(-1, 1, 3)).squeeze(1)
        released = ((self.transport_arm >= 0) & ~s['robot'] & ~s['fingers'].any(-1).any(-1)
                    & (opening >= c.release_opening) & ((pad-s['cube_position']).norm(dim=-1) >= c.hand_clearance))
        stable = (self.released & inside & (lowest >= c.plate_bottom-.002) & (lowest <= c.plate_top+.003)
                  & released & s['plate'] & ~s['table'])
        self.stable_count = torch.where(update, torch.where(stable, self.stable_count+1, 0), self.stable_count)
        self.success |= update & (self.stable_count >= c.stable_steps)
        return self.success


class BatchEpisodeState:
    def __init__(self, plate, initial_pose, initial_cube, initial_fingers):
        n, d = len(plate), plate.device
        self.placement = BatchPlacementScorer(plate)
        self.steps = torch.zeros(n, dtype=torch.long, device=d)
        self.reason = self.steps.clone()
        self.rest = self.steps.clone()
        self.max_clear = torch.zeros(n, device=d, dtype=plate.dtype)
        self.was_closed = torch.zeros(n, device=d, dtype=torch.bool)
        self.prev_closed = self.was_closed.clone()
        self.released = self.was_closed.clone()
        self.lifted = self.was_closed.clone()
        self.hold_intact = torch.ones_like(self.was_closed)
        self.held_to_release = self.was_closed.clone()
        self.release_checked = self.was_closed.clone()
        self.done = self.was_closed.clone()
        self.error = self.was_closed.clone()
        self.previous_offset = hand_offset(initial_pose, initial_cube)
        self.previous_both = initial_fingers.all(-1)
        self.slip_step = self.max_clear.clone()
        self.lift_height = self.max_clear.clone()

    @property
    def active(self):
        return ~(self.done | self.error)

    def reset(self, ids, plate, pose, cube, fingers):
        self.placement.reset(ids, plate)
        for name in ('steps', 'reason', 'rest', 'max_clear', 'was_closed', 'prev_closed', 'released',
                     'lifted', 'held_to_release', 'release_checked', 'done', 'error', 'slip_step', 'lift_height'):
            getattr(self, name)[ids] = 0
        self.hold_intact[ids] = True
        self.previous_offset[ids] = hand_offset(pose, cube)
        self.previous_both[ids] = fingers.all(-1)

    def update(self, s, pose, closed, *, native_terminated, native_truncated, native_success, execution_error):
        active = self.active.clone()
        self.error |= active & execution_error
        valid = active & ~execution_error
        self.steps += valid.long()
        self.placement.update(s, valid)
        native_success = native_success | self.placement.success
        native_terminated = native_terminated | self.placement.success
        v = vertices(s['cube_position'], s['cube_quaternion'])
        clear = v[:, :, 2].amin(-1)-Criteria().table_height
        both = s['fingers'][:, 1].all(-1)
        held = both & ~s['table'] & ~s['plate']
        self.max_clear = torch.where(valid & held, torch.maximum(self.max_clear, clear), self.max_clear)
        self.lift_height = torch.where(valid, clear, self.lift_height)
        # held_to_release uses the first lift and the first closed->open transition AFTER it.
        opening_after_lift = valid & self.lifted & self.prev_closed & ~closed & ~self.release_checked
        self.held_to_release |= opening_after_lift & self.hold_intact
        self.release_checked |= opening_after_lift
        self.hold_intact &= ~(valid & self.lifted & ~opening_after_lift & ~self.held_to_release & ~both)
        self.lifted |= valid & held & (clear > .002)
        self.released |= valid & self.was_closed & ~closed
        self.was_closed |= valid & closed
        self.prev_closed = torch.where(valid, closed, self.prev_closed)
        lowest = clear+Criteria().table_height
        rest = (self.released & s['plate'] & ~s['table'] & ~s['robot'] & ~s['fingers'][:, 1].any(-1)
                & (s['opening'][:, 1] >= .8) & (lowest >= Criteria().plate_bottom-.002)
                & (lowest <= Criteria().plate_top+.003)
                & ((s['cube_position'][:, :2]-self.placement.plate[:, :2]).norm(dim=-1) <= PLATE_REST_RADIUS))
        self.rest = torch.where(valid, torch.where(rest, self.rest+1, 0), self.rest)
        reason = torch.where((self.rest >= REST_STEPS) | (native_terminated & native_success), 1,
                 torch.where(native_terminated | native_truncated, 2,
                 torch.where(s['table'] & (self.max_clear >= .005), 3,
                 torch.where((self.steps >= NO_LIFT_STEP) & (self.max_clear < .005), 4,
                 torch.where(self.steps == HORIZON, 5, 0)))))
        self.reason = torch.where(valid, reason, self.reason)
        self.done |= valid & (reason != 0)
        offset = hand_offset(pose, s['cube_position'])
        self.slip_step = torch.where(valid & both & self.previous_both,
                                     (offset-self.previous_offset).norm(dim=-1), 0)
        self.previous_offset = torch.where(valid[:, None], offset, self.previous_offset)
        self.previous_both = torch.where(valid, both, self.previous_both)
        return valid
