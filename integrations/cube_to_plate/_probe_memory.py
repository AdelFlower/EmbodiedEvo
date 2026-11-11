"""TEMPORARY probe (delete after M3a): host RSS growth of the batch runtime vs N and steps.

.venv-robodojo/bin/python -m integrations.cube_to_plate._probe_memory --run-dir RUN_DIR N STEPS
"""
import json, os, sys, time, traceback
from pathlib import Path

run = Path(sys.argv[sys.argv.index("--run-dir") + 1]).resolve(); n, steps = int(sys.argv[3]), int(sys.argv[4])
diag = set(sys.argv[5].split(',')) - {''} if len(sys.argv) > 5 else set()
out = run / f"memory_N{n}_{'-'.join(sorted(diag)) or 'base'}.json"
rec = {'N': n, 'events': []}

def rss():
    return int(open('/proc/self/statm').read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**30

def mark(name):
    rec['events'].append({'t': time.monotonic(), 'event': name, 'rss_gib': rss()})
    out.write_text(json.dumps(rec, indent=1))

from simulators.robodojo import robodojo_cwd
from simulators.cube_to_plate_batch_scene import launch_app
app = None
try:
    with robodojo_cwd():
        import simulators.cube_to_plate_batch_scene as S
        S.DIAG = diag
        app = launch_app(run); mark('app')
        import torch
        from integrations.cube_to_plate.layouts import make_layouts
        from integrations.cube_to_plate.policy import PolicyAdapter
        from orchestrator.harness.contracts import PolicyBundle
        from simulators.cube_to_plate_batch import BatchRuntime
        from orchestrator.runtime import CODE_ROOT
        bundle = sorted((CODE_ROOT / 'runs').glob('*_m3a_bundle_prep/bundle'))[0]
        policy = PolicyAdapter().load(PolicyBundle('P0', bundle, False)); policy.model.to('cuda:0')
        runtime = BatchRuntime({'target': None, 'calibration': policy.calibration}, n, False); mark('built')
        batch = runtime.reset(make_layouts('mem_probe', 0, n)); mark('reset')
        mark('reset_done')
        for t in range(steps):
            with torch.no_grad():
                batch = runtime.step(policy.model.actor(batch.actor))
            if t % 25 == 24:
                mark(f'step{t+1}')
except BaseException:
    rec['error'] = traceback.format_exc(); out.write_text(json.dumps(rec, indent=1))
os._exit(0)
