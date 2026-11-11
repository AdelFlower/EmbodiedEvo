"""Frozen HARNESS_SPEC §15.2 criteria and reference readers; CPU-testable."""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_REFERENCE = ROOT / 'runs/202611110150_probe_parity_cpu_source'


def reference_records(directory):
    records = [json.loads(line) for line in (Path(directory)/'episodes.jsonl').read_text().splitlines()]
    records.sort(key=lambda row: row['k'])
    if len(records) != 20 or [r['k'] for r in records] != list(range(20)) or any(r['stream'] != 37 for r in records):
        raise ValueError('frozen reference requires exactly the twenty DP stream-37 episodes')
    for row in records:
        with np.load(Path(row['episode_dir'])/'trace.npz') as trace:
            if len(trace['closed']) != row['steps'] or str(trace['stop_reason']) != row['stop_reason']:
                raise ValueError('reference episode and trace disagree')
            row['final_cube_position'] = trace['cube_position'][-1].tolist()
        # Read the reset transition; trace.npz contains poses, not joint states.
        with (Path(row['episode_dir'])/'transitions.jsonl').open() as stream:
            first = json.loads(next(stream))
            if not first['reset']:
                raise ValueError('reference trace must start at reset')
            row['initial_pose'] = first['state']['public_state']['right_ee_pose']
    return records


def criterion(value, rule, passed):
    return {'value': value, 'criterion': rule, 'pass': bool(passed)}


def task_metrics(records, reference, target=False):
    strict = sum(r['strict_success'] for r in records)
    lenient = sum(r['success'] for r in records)
    complete = len(records) == 20 and all(r['execution_status'] == 'complete' for r in records)
    if target:
        bad = sum(r['stop_reason'] in ('dropped', 'no_lift') for r in records)
        criteria = {'strict': criterion(strict, '<= 2/20', strict <= 2),
                    'lenient': criterion(lenient, '<= 2/20', lenient <= 2),
                    'dropped_plus_no_lift': criterion(bad, '>= 18/20', bad >= 18)}
    else:
        equal = sum(a['stop_reason'] == b['stop_reason'] for a, b in zip(records, reference, strict=True))
        length = sum(abs(a['steps']-b['steps']) <= 10 for a, b in zip(records, reference, strict=True))
        means = {k: math.fsum(r[k] for r in records)/len(records) if all(r[k] is not None for r in records) else None for k in ('max_clear', 'slip', 'force_mean_held', 'release_dist')}
        criteria = {'strict': criterion(strict, '>= 19/20', strict >= 19),
                    'lenient': criterion(lenient, '>= 19/20', lenient >= 19),
                    'stop_reason_equal': criterion(equal, '>= 19/20', equal >= 19),
                    'length_within_10': criterion(length, '>= 18/20', length >= 18),
                    'mean_max_clearance_m': criterion(means['max_clear'], '0.0504 +/- 0.003 m', means['max_clear'] is not None and .0474 <= means['max_clear'] <= .0534),
                    'mean_slip_m': criterion(means['slip'], '<= 0.001 m', means['slip'] is not None and means['slip'] <= .001),
                    'mean_held_force_N': criterion(means['force_mean_held'], '15.0 +/- 1.0 N', means['force_mean_held'] is not None and abs(means['force_mean_held']-15) <= 1),
                    'mean_release_distance_m': criterion(means['release_dist'], '<= 0.0098 m', means['release_dist'] is not None and means['release_dist'] <= .0098)}
    return {'planned_episodes': 20, 'completed_episodes': sum(r['execution_status'] == 'complete' for r in records),
            'criteria': criteria, 'pass': complete and all(c['pass'] for c in criteria.values())}


def controller_metrics(pairs, outside):
    agreement = sum(p['dls_success'] == p['curobo_success'] for p in pairs)/len(pairs) if pairs else None
    both = [p for p in pairs if p['dls_success'] and p['curobo_success']]
    dq = float(np.percentile(np.abs([p['delta_q'] for p in both]), 99)) if both else None
    pe = float(np.percentile([p['position_error_m'] for p in both], 99)) if both else None
    re = float(np.percentile([p['orientation_error_rad'] for p in both], 99)) if both else None
    outside_ok = all(not p['dls_success'] and not p['curobo_success'] for p in outside)
    criteria = {'success_agreement': criterion(agreement, '>= 0.995', agreement is not None and agreement >= .995),
        'p99_abs_delta_q_rad': criterion(dq, '<= 0.01 rad', dq is not None and dq <= .01),
        'p99_fk_position_error_m': criterion(pe, '<= 0.001 m', pe is not None and pe <= .001),
        'p99_fk_orientation_error_rad': criterion(re, '<= 0.01 rad', re is not None and re <= .01),
        'outside_workspace_fail_both': criterion(outside_ok, 'all outside-workspace commands fail in both', outside_ok)}
    return {'visited_pairs': len(pairs), 'both_success': len(both), 'outside_pairs': len(outside),
            'criteria': criteria, 'pass': bool(pairs) and bool(outside) and all(c['pass'] for c in criteria.values())}


def independence_metrics(single, batch, cross_contacts):
    equal = all((a['strict_success'], a['success'], a['stop_reason']) ==
                (b['strict_success'], b['success'], b['stop_reason']) for a, b in zip(single, batch[:20], strict=True))
    length = max(abs(a['steps']-b['steps']) for a, b in zip(single, batch[:20], strict=True))
    position = (max(float(np.linalg.norm(np.array(a['final_cube_position'])-b['final_cube_position']))
                   for a, b in zip(single, batch[:20], strict=True))
                if all(r['final_cube_position'] is not None for r in single+batch[:20]) else None)
    criteria = {'identical_outcomes_and_stops': criterion(equal, 'identical for all 20', equal),
        'max_episode_length_difference_steps': criterion(length, '<= 2 steps for all 20', length <= 2),
        'max_final_cube_position_difference_m': criterion(position, '<= 0.002 m for all 20', position is not None and position <= .002),
        'cross_replica_contacts': criterion(cross_contacts, '0 at every step', cross_contacts == 0)}
    return {'criteria': criteria, 'pass': all(c['pass'] for c in criteria.values())
            and all(r['execution_status'] == 'complete' for r in single+batch)}


def unavailable_metrics(gate):
    """An interrupted/failed gate retains every criterion as unmeasured and failed."""
    if gate in ('source', 'target'):
        row = {'strict_success': False, 'success': False, 'stop_reason': None, 'steps': 0,
               'max_clear': None, 'slip': None, 'force_mean_held': None, 'release_dist': None,
               'execution_status': 'execution_error'}
        result = task_metrics([row]*20, [row]*20, gate == 'target')
    elif gate == 'controller':
        result = controller_metrics([], [])
    elif gate == 'independence':
        row = {'strict_success': False, 'success': False, 'stop_reason': None, 'steps': 0,
               'execution_status': 'execution_error', 'final_cube_position': None}
        result = independence_metrics([row]*20, [row]*512, 0)
    else:
        result = {'criteria': {'env_steps_per_s': criterion(None, '>= 2000 env-steps/s over all 300 control steps', False)}}
    result['criteria'] = {name: {**value, 'value': None, 'pass': False} for name, value in result['criteria'].items()}
    result['pass'] = False
    return result
