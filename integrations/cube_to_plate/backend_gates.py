"""Overseer-only M3a GPU gates. The default CLI launches the existing simulator wrapper.

No gate is executed by importing this module. All records here are host-private.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np

from integrations.cube_to_plate.backend_gate_metrics import (
    ROOT, SOURCE_REFERENCE, controller_metrics, criterion, independence_metrics, reference_records, task_metrics, unavailable_metrics,
)
from integrations.cube_to_plate.layouts import make_layouts
from integrations.deployable.layout import build_scene
from orchestrator.deployable import episode_seed
from orchestrator.files import write_json, append_jsonl
from orchestrator.resources import sim_env


def dp_layouts(cfg):
    """Gate layouts are DP uint32 stream-37 seeds, NOT M2 named-stream seeds."""
    nominal = json.loads((ROOT/'tasks/DynamicsAdapt/scene.json').read_text())
    rows = []
    for k in range(20):
        seed = episode_seed(cfg, 37, k)
        scene = build_scene(seed, cfg)
        row = {'stream': 'dp_dev_37', 'index': k, 'scene_seed': seed}
        for name in ('cube', 'plate'):
            row[f'{name}_offset'] = (np.array(scene['Rigid'][name][0]['default_pos'][:2])-
                                    nominal['Rigid'][name][0]['default_pos'][:2]).tolist()
        rows.append(row)
    return rows


def _memory():
    import torch
    free, total = torch.cuda.mem_get_info()
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    return {'vram_used_bytes': total-free, 'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'host_rss_bytes': int(status['VmRSS'].split()[0])*1024,
            'host_peak_rss_bytes': int(status['VmHWM'].split()[0])*1024}


def collect(directory, layouts, seeds, policy, target, *, full_iteration=False, audit=False, controls=False):
    import torch
    from simulators.cube_to_plate_batch import BatchRuntime
    from simulators.cube_to_plate_batch_scoring import STOP_REASONS
    from simulators.cube_to_plate_batch_scene import source_build_report
    from integrations.cube_to_plate.batch_records import episode_record
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/'planned_episodes.json', [{'k': i, 'layout': l, 'detector_seed': seed,
        'execution_status': 'planned'} for i, (l, seed) in enumerate(zip(layouts, seeds, strict=True))])
    for i in range(len(layouts)):
        episode_dir = directory/f'episode_{i:04d}'
        episode_dir.mkdir()
        write_json(episode_dir/'episode.json', {'k': i, 'execution_status': 'incomplete', 'steps': 0})
    runtime = BatchRuntime({'target': target, 'calibration': policy.calibration}, len(layouts), False)
    policy.model.to(runtime.device)
    if audit:
        runtime.enable_contact_audit()
    batch = runtime.reset(layouts)
    runtime.set_detection(list(range(len(layouts))), seeds)
    # Re-encode the reset actor with the exact frozen DP noise draws.
    batch = runtime.reset_observation()
    initial = {k: v.cpu().numpy() for k, v in runtime.initial.items()}
    control_rows = []
    torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(300):
        with torch.no_grad():
            actions = policy.model.actor(batch.actor)
        batch = runtime.step(actions)
        if controls:
            control_rows.append({k: v.clone() for k, v in runtime.last_control.items()})
        if not full_iteration and not bool(runtime.episodes.active.any()):
            break
    chunk = runtime.episode_chunk()  # evidence copy is inside the measured interval
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    timing = {'num_envs': len(layouts), 'control_steps': len(chunk['valid']),
              'valid_env_steps': int(chunk['valid'].sum()), 'elapsed_s': elapsed,
              'env_steps_per_s': float(chunk['valid'].sum()/elapsed), **_memory()}
    np.savez(directory/'evidence.npz', **chunk, detection=runtime.detection_host, **{f'initial_{k}': v for k, v in initial.items()})
    if controls:
        np.savez(directory/'controller_pairs.npz', **{
            k: torch.stack([r[k] for r in control_rows]).cpu().numpy() for k in control_rows[0]})
    records = []
    reasons = runtime.episodes.reason.cpu().tolist()
    errors = runtime.episodes.error.cpu().tolist()
    native = runtime.episodes.placement.success.cpu().tolist()
    for i, layout in enumerate(layouts):
        # Include the attempted failing transition as partial evidence; it is
        # never labeled done or counted as a valid learning transition.
        mask = chunk['valid'][:, i] | chunk['execution_error'][:, i]
        if errors[i]:
            first_error = np.flatnonzero(chunk['execution_error'][:, i])
            if len(first_error):
                mask[first_error[0]+1:] = False
        trace = {k: v[mask, i] for k, v in chunk.items()}
        trace['observed_ee_pose'] = trace.pop('ee_pose')
        trace['action'] = trace['executed_action']
        trace['plate_position'] = initial['plate_position'][i]
        for dest, src in (('initial_observed_ee_pose', 'ee_pose'), ('initial_cube_position', 'cube_position'),
                          ('initial_cube_quaternion', 'cube_quaternion'), ('initial_right_finger_contacts', 'right_finger_contacts')):
            trace[dest] = initial[src][i]
        reason = STOP_REASONS[reasons[i]]
        trace['stop_reason'] = reason if reason is not None else 'execution_error'
        record = episode_record(trace, reason, native[i], errors[i])
        truth = np.array([initial['cube_position'][i, :2], initial['plate_position'][i, :2]])
        record.update(k=i, stream=layout['stream'], layout_index=layout['index'],
            observer={'detector': 'simulated', 'detection': runtime.detection_host[i].tolist(),
                      'truth': truth.tolist(), 'error': (runtime.detection_host[i]-truth).tolist()})
        episode_dir = directory/f'episode_{i:04d}'
        np.savez(episode_dir/'trace.npz', **trace)
        write_json(episode_dir/'episode.json', record)
        append_jsonl(directory/'episodes.jsonl', {**record, 'episode_dir': str(episode_dir.resolve())})
        records.append(record)
    if target is None:
        write_json(directory/'source_build.json', source_build_report(runtime.sim, runtime.scene, runtime.ids, runtime.paths))
    cross = runtime.cross_replica_contacts
    write_json(directory/'performance.json', timing)
    runtime.close()
    return records, timing, cross


def controller_gate(args):
    import torch
    from omegaconf import OmegaConf
    from utils.load_file import load_yaml
    from env.robot_manager.robot_manager import RobotManager
    from simulators.robodojo import ROOT as ROBO
    from simulators.cube_to_plate_batch_ik import ArmKinematics
    from integrations.friction_rl.control import ACTION_LOW, ACTION_HIGH
    # Gate 2 writes the exact pre-IK state and decoded command for every visited pair.
    with np.load(args.source_run/'source/controller_pairs.npz') as data:
        valid = data['valid']
        locations = np.argwhere(valid)
        current, command, root = (torch.as_tensor(data[k][valid], device='cuda:0') for k in ('current', 'command', 'root'))
    cfg = OmegaConf.create(load_yaml(str(ROBO/'env_cfg/robot/dual_x5.yml')))
    manager = RobotManager(num_envs=1, env_spacing=7, config=cfg, dt=.004, device='cuda:0')
    robot = manager.robot_list[1]
    solver = manager.ik_solver[robot.robot_name]
    model = ArmKinematics(device='cuda:0')
    solution, success, pe, re_ = model.solve(current, command, root)
    q, cmd, solved, ok, pe, re_ = (v.cpu().numpy() for v in (current, command, solution, success, pe, re_))
    pairs = []
    for k in range(len(q)):
        legacy = solver.solve_ik_to_joint(q[k], cmd[k], real_robot_pose=robot.entity_origin_pose)
        legacy_ok = legacy['status'] == 'Success'
        row = {'pair': k, 'episode': int(locations[k, 1]), 'control_step': int(locations[k, 0]), 'dls_success': bool(ok[k]), 'curobo_success': legacy_ok,
               'delta_q': (solved[k]-legacy['joint_value']).tolist() if legacy_ok and ok[k] else None,
               'position_error_m': float(pe[k]), 'orientation_error_rad': float(re_[k])}
        pairs.append(row)
        append_jsonl(args.run_dir/'controller_pairs.jsonl', row)
    outside = []
    for axis in range(3):
        for sign in (-1, 1):
            target = cmd[0].copy()
            target[axis] = (ACTION_LOW[axis]-2 if sign < 0 else ACTION_HIGH[axis]+2)
            result = model.solve(current[:1], torch.as_tensor(target[None], device='cuda:0'), root[:1])
            legacy = solver.solve_ik_to_joint(q[0], target, real_robot_pose=robot.entity_origin_pose)
            row = {'axis': axis, 'sign': sign, 'dls_success': bool(result[1][0]),
                   'curobo_success': legacy['status'] == 'Success'}
            outside.append(row)
            append_jsonl(args.run_dir/'outside_workspace.jsonl', row)
    for episode in range(20):
        visited = [p for p in pairs if p['episode'] == episode]
        record = {'k': episode, **controller_metrics(visited, outside)}
        folder = args.run_dir/f'episode_{episode:04d}'
        folder.mkdir()
        write_json(folder/'episode.json', record)
        append_jsonl(args.run_dir/'episodes.jsonl', record)
    return controller_metrics(pairs, outside)


def run_gate(args):
    if args.gate == 'controller':
        return controller_gate(args)
    import torch
    from integrations.cube_to_plate.policy import PolicyAdapter
    from integrations.cube_to_plate.target import load_target_spec
    from orchestrator.harness.contracts import PolicyBundle
    cfg = json.loads((ROOT/'tasks/Deployable/config.json').read_text())
    policy = PolicyAdapter().load(PolicyBundle('m3a_P0', args.bundle, False))
    policy.model.to('cuda:0')
    layouts = dp_layouts(cfg)
    seeds = [[cfg['seed'], 37, k, 1] for k in range(20)]
    if args.gate in ('source', 'target'):
        reference_dir = SOURCE_REFERENCE if args.gate == 'source' else args.target_reference
        reference = reference_records(reference_dir)
        target = load_target_spec(args.target_spec) if args.gate == 'target' else None
        records, timing, _ = collect(args.run_dir/args.gate, layouts, seeds, policy, target,
                                     controls=args.gate == 'source')
        return {**task_metrics(records, reference, args.gate == 'target'), 'performance': timing,
                'reference': str(reference_dir),
                'reference_outcomes': {'strict': sum(r['strict_success'] for r in reference),
                                       'lenient': sum(r['success'] for r in reference),
                                       'dropped_plus_no_lift': sum(r['stop_reason'] in ('dropped', 'no_lift') for r in reference)}}
    evaluation = make_layouts('m3_source_evaluation', cfg['seed'], 512)
    if args.gate == 'independence':
        singles, scaling, cross = [], [], 0
        single_timing = []
        for k in range(20):
            rows, timing, audit = collect(args.run_dir/f'single_{k:02d}', layouts[k:k+1], seeds[k:k+1],
                                          policy, None, audit=True)
            singles.extend(rows)
            single_timing.append(timing)
            cross += audit
        seconds = sum(p['elapsed_s'] for p in single_timing)
        steps = sum(p['valid_env_steps'] for p in single_timing)
        scaling.append({'num_envs': 1, 'episodes': 20, 'elapsed_s': seconds, 'valid_env_steps': steps,
            'env_steps_per_s': steps/seconds,
            **{key: max(p[key] for p in single_timing) for key in
               ('vram_used_bytes', 'torch_peak_allocated_bytes', 'host_rss_bytes', 'host_peak_rss_bytes')}})
        for n in (16, 256, 512):
            chosen = layouts[:n] if n == 16 else layouts+evaluation[20:n]
            noise = seeds[:n] if n == 16 else seeds+[[int(l['scene_seed']), 1] for l in evaluation[20:n]]
            rows, timing, audit = collect(args.run_dir/f'batch_{n}', chosen, noise, policy, None, audit=True)
            scaling.append(timing)
            cross += audit
            if n == 512:
                batch = rows
        result = independence_metrics(singles, batch, cross)
        for k, (single, replica) in enumerate(zip(singles, batch[:20], strict=True)):
            append_jsonl(args.run_dir/'comparisons.jsonl', {'k': k,
                **independence_metrics([single], [replica], cross)})
        result.update(scaling=scaling)
        # The ungated 512-layout evaluation uses the entire frozen evaluation table.
        rows, timing, _ = collect(args.run_dir/'source_evaluation_512', evaluation,
            [[int(l['scene_seed']), 1] for l in evaluation], policy, None)
        result['source_evaluation_strict_rate_512'] = sum(r['strict_success'] for r in rows)/512
        return result
    rows, timing, _ = collect(args.run_dir/'throughput', evaluation,
        [[int(l['scene_seed']), 1] for l in evaluation], policy, None, full_iteration=True)
    c = criterion(timing['env_steps_per_s'], '>= 2000 env-steps/s over all 300 control steps', timing['env_steps_per_s'] >= 2000)
    return {'criteria': {'env_steps_per_s': c}, 'pass': c['pass'] and all(r['execution_status'] == 'complete' for r in rows),
            'performance': timing, 'source_evaluation_strict_rate_512': sum(r['strict_success'] for r in rows)/512}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gate', choices=('controller', 'source', 'target', 'independence', 'throughput'), required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--target-spec', type=Path)
    parser.add_argument('--target-reference', type=Path)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--physics-device', choices=('cuda:0', 'cpu'), default='cuda:0',
                        help='host diagnostic only; formal gates use cuda:0')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not re.match(r'^\d{12}(?:_|$)', args.run_dir.name) or args.gpu < 0:
        parser.error('require YYYYMMDDHHMM run prefix and a nonnegative GPU')
    if args.gate == 'target' and (args.target_spec is None or args.target_reference is None):
        parser.error('target gate requires --target-spec and --target-reference')
    if args.gate == 'controller' and args.source_run is None:
        parser.error('controller gate requires --source-run from gate 2')
    args.run_dir = args.run_dir.resolve()
    args.bundle = args.bundle.resolve()
    for key in ('source_run', 'target_spec', 'target_reference'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if not args._worker:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.run_dir/'planned.json', {'gate': args.gate, 'status': 'planned',
            'planned_episodes': 20 if args.gate in ('source', 'target', 'controller') else 512 if args.gate == 'throughput' else 1316 if args.gate == 'independence' else 0})
        command = [str(ROOT/'.venv-robodojo/bin/python'), '-m', 'integrations.cube_to_plate.backend_gates',
                   '--run-dir', str(args.run_dir), '--gate', args.gate, '--bundle', str(args.bundle), '--physics-device', args.physics_device, '--_worker']
        for key in ('source_run', 'target_spec', 'target_reference'):
            if getattr(args, key) is not None:
                command.extend(['--'+key.replace('_', '-'), str(getattr(args, key))])
        with (args.run_dir/'simulator.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=sim_env(str(args.gpu), 'robodojo'), stdout=log, stderr=subprocess.STDOUT)
        # SimulationApp.close() may exit 0 after an exception; summary is authoritative.
        summary_path = args.run_dir/'summary.json'
        if not summary_path.is_file():
            write_json(summary_path, {**unavailable_metrics(args.gate), 'gate': args.gate, 'status': 'interrupted',
                                      'worker_returncode': result.returncode, 'planned_denominator_retained': True})
        return 0 if json.loads(summary_path.read_text())['pass'] else 1
    from simulators.robodojo import robodojo_cwd
    from simulators.cube_to_plate_batch_scene import launch_app
    import simulators.cube_to_plate_batch_scene as batch_scene
    batch_scene.PHYSICS_DEVICE = args.physics_device
    import faulthandler
    import signal
    # Operator diagnostic: `kill -USR1 <pid>` appends every Python thread's stack.
    faulthandler.register(signal.SIGUSR1, file=(args.run_dir/'stacks.txt').open('a'), all_threads=True)
    app = None
    # The only broad boundary: persist failures BEFORE Kit close can exit(0).
    try:
        with robodojo_cwd():
            app = launch_app(args.run_dir)
            summary = run_gate(args)
            write_json(args.run_dir/'summary.json', {'gate': args.gate, 'status': 'complete', **summary})
    except BaseException as exc:
        detail = (''.join(traceback.format_tb(exc.__traceback__))+type(exc).__name__+'\n'
                  if args.gate == 'target' else traceback.format_exc())
        (args.run_dir/'error.txt').write_text(detail)
        write_json(args.run_dir/'summary.json', {**unavailable_metrics(args.gate), 'gate': args.gate, 'status': 'execution_error',
                                                'planned_denominator_retained': True, 'error': 'error.txt'})
    finally:
        if app is not None:
            app.close()
    return 0 if json.loads((args.run_dir/'summary.json').read_text())['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
