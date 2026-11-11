"""Episode-chunk host records, shared with the M3 gate runner (host-private)."""

import numpy as np

from integrations.friction_rl.episode import hand_offsets, motion_peaks
from integrations.friction_rl.network import held_to_release


def episode_record(trace, reason, native_success, execution_error):
    if execution_error:
        return {'success': False, 'strict_success': False, 'held_to_release': False,
                'stop_reason': None, 'steps': int(trace['valid'].sum()), 'slip': None,
                'max_clear': None, 'force_mean_held': None, 'release_dist': None,
                'native_success': bool(native_success), 'execution_status': 'execution_error',
                'final_cube_position': None}
    both = trace['right_finger_contacts'].all(-1)
    held = both & ~trace['contacts'][:, :2].any(-1)
    closed = trace['closed']
    force_rows = closed & both
    offsets = hand_offsets(trace['observed_ee_pose'], trace['cube_position'])
    slip = 0.
    if held.any() and closed.any():
        first, last = np.flatnonzero(held)[0], np.flatnonzero(closed)[-1]
        slip = float(np.linalg.norm(offsets[first:last+1]-offsets[first], axis=1).max(initial=0.))
    forces = trace['grip_force_command'][force_rows]
    speed, accel = motion_peaks(trace['commanded_ee_pose'][closed])
    hold = held_to_release(trace)
    success = reason == 'success' and not execution_error
    return {'success': success, 'strict_success': success and hold, 'held_to_release': hold,
            'stop_reason': reason, 'steps': len(closed), 'slip': slip,
            'max_clear': float(trace['max_clear'][-1]),
            'force_mean_held': float(forces.mean()) if len(forces) else 0.,
            'release_dist': float(np.linalg.norm(trace['cube_position'][-1, :2]-trace['plate_position'][:2])),
            'cmd_peak_speed': speed, 'cmd_peak_accel': accel, 'native_success': bool(native_success),
            'execution_status': 'execution_error' if execution_error else 'complete',
            'final_cube_position': trace['cube_position'][-1].tolist()}
