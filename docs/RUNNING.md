# Running DynamicsAdapt

Run from the repository root after [installation](INSTALL.md). The commands below
use `.venv-rl/bin/dynamicsadapt`; `.venv-rl/bin/python -m orchestrator` is equivalent.
Every experiment and P0 command requires an explicit timestamp-prefixed
`--run-dir`. New runs use new directories; evaluation and summary address an
existing run. Keep all evidence under `runs/`, outside Git.

## Prepare the configuration

Copy `tasks/DynamicsAdapt/config.example.json` to a local configuration and set:

- `p0.policy` and `p0.demos` to existing local FT5 weights and source demonstrations;
- `simulator_python` to the provisioned `.venv-robodojo/bin/python` wrapper;
- `agent.image` and `sandbox.image` to available thin Agent images;
- `agent.model`, `reasoning_effort` and `service_tier` to the intended experiment settings.

The example's historical P0 paths are references, not bundled artifacts. The
current default is one scene, source friction 1.0 and target friction 0.25, four
development episodes, 30 final episodes, and 27 required strict successes.
[The protocol](PROTOCOL.md) defines the training, scoring, budget and isolation
rules. Fix configuration before a new run; do not edit target physics to obtain a
preferred outcome.

## Optional P0 preparation

Existing compatible P0 artifacts may be used directly. Otherwise collect source
script demonstrations and fit the imitation policy with the P0 configuration.
These commands use nominal source physics; they do not adapt to the target.

```bash
P0_CONFIG=tasks/DynamicsAdapt/p0.example.json
DEMO_RUN="runs/$(date +%Y%m%d%H%M)_p0_demos"
.venv-rl/bin/dynamicsadapt p0 demos --config "$P0_CONFIG" \
  --run-dir "$DEMO_RUN" --gpu 0
BC_RUN="runs/$(date +%Y%m%d%H%M)_p0_bc"
.venv-rl/bin/dynamicsadapt p0 bc --config "$P0_CONFIG" \
  --demos "$DEMO_RUN/demos.npz" --run-dir "$BC_RUN"
.venv-rl/bin/dynamicsadapt p0 evaluate --config "$P0_CONFIG" \
  --policy "$BC_RUN/policy.pt" --episodes 30 \
  --run-dir "runs/$(date +%Y%m%d%H%M)_p0_source_eval" --gpu 0
```

Use a separate evaluation run with `--stochastic` to inspect the policy under
training exploration, and `--render` when raw videos are needed. Collection and
evaluation need the GPU simulator; BC is CPU training. Preserve both the weights
and their demonstration file and record these paths as the formal run's P0 inputs.
There is no P0 search or PPO-training CLI in the minimal preparation tool.

## Run, evaluate and summarize

```bash
CONFIG=/path/to/experiment.json
RUN_DIR="runs/$(date +%Y%m%d%H%M)_dynamicsadapt"
.venv-rl/bin/dynamicsadapt run --config "$CONFIG" --run-dir "$RUN_DIR" --gpu 0
.venv-rl/bin/dynamicsadapt evaluate --run-dir "$RUN_DIR" --gpu 0
.venv-rl/bin/dynamicsadapt summarize --run-dir "$RUN_DIR"
```

`run` freezes P0 and source demonstrations, records paired baseline rollouts, then
starts the isolated Agent and experiment service. A successful train request
includes automatic target deployment. The host applies the fixed selection rule.
`evaluate` runs P0 and the selected network, without further training, using the
configured deterministic/stochastic evaluation episodes. `summarize` derives the
report from retained records and launches no simulator or model.

The Agent uses `scripts/adapt.py` inside its workspace. Its instructions explain
`status`, `probe`, `train`, `wait` and `submit`, reward programs and diagnosis fields.
The host instructions in this document, including true target parameters, must
not be copied into that workspace.

## Separate checks

CPU tests, when the local suite is available (it is excluded from Git):

```bash
.venv-rl/bin/python -m unittest discover -s tests -v
```

The scripted smoke uses real simulation and Docker reward evaluation but no model
Agent. Update its P0/image paths as for the main configuration before running:

```bash
SMOKE_RUN="runs/$(date +%Y%m%d%H%M)_dynamicsadapt_smoke"
.venv-rl/bin/dynamicsadapt run --config tasks/DynamicsAdapt/smoke.json \
  --scripted-agent tasks/DynamicsAdapt/smoke_agent.py --run-dir "$SMOKE_RUN" --gpu 0
.venv-rl/bin/dynamicsadapt evaluate --run-dir "$SMOKE_RUN" --gpu 0
```

The smoke deliberately uses the true target parameters to test the reproduction
gate and job lifecycle. It supplies no scientific diagnosis evidence.

The researcher can separately test whether the training procedure can recover
performance when given the true dynamics and an oracle reward:

```bash
.venv-rl/bin/dynamicsadapt trainability --config "$CONFIG" \
  --parameters /path/to/source-parameters.json \
  --reward tasks/DynamicsAdapt/oracle_reward.py --iterations 100 \
  --run-dir "runs/$(date +%Y%m%d%H%M)_dynamicsadapt_trainability" --gpu 0
```

The parameter file is a JSON object of permitted source physics. This host-only
check has its own run, costs and outcomes; it has no Agent and does not consume
Agent budgets. Its weights, trajectories and logs must not initialize or inform
the formal Agent. A failure describes the training procedure, not Agent diagnosis.

## Records

| Run path | Content |
|---|---|
| `config.json`, `networks/N000/policy.pt`, `prior/demos.npz` | Frozen configuration and initial policy/prior |
| `baseline/{source,target}/` | Paired development rollout evidence |
| `calibration/workspace/` | Public instructions, observations, Agent notes, rewards and submission |
| `calibration/requests.jsonl`, `calibration/jobs/J*/` | Requests/refusals and retained job specifications/results |
| `calibration/networks/N*/` | `policy.pt`, full `training.pt`, reward snapshot and iteration records |
| `calibration/selection.json`, `diagnosis.json`, `agent_sessions/` | Host selection, Agent diagnosis and model sessions |
| `evaluation/results.json`, `summary.json` | Frozen evaluation counts and derived report |

Host episode folders retain traces and native records; public copies contain only
whitelisted `trajectory.npz`, `result.json` and raw camera videos when available.
Training exports include iteration metrics and final-iteration trajectories with
the rewards actually used. `training.pt` preserves the model, both optimizers and
cumulative counters; `policy.pt` is the inference state dictionary.

Keep partial and failed evidence. Count incomplete planned episodes as failures,
record actual resource use, and assess diagnosis separately from strict/lenient
success. Never overwrite or retry an attempt automatically.

## Deployable policy

DP is a separate nominal-physics pipeline with random ±3 cm layouts, one-shot
camera detection, a 21-feature actor and a privileged training-only critic. See
[the DP specification](DEPLOYABLE_POLICY_SPEC.md) for the fixed settings and gates.
Each command needs a new timestamp-prefixed directory. The simulator commands
below require GPU resources; `bc` runs on CPU.

```bash
DP=tasks/Deployable/config.json
C="runs/$(date +%Y%m%d%H%M)_dp_calibrate"
.venv-rl/bin/dynamicsadapt deploy calibrate --config "$DP" --run-dir "$C"
CAL="$C/calibration.json"
.venv-rl/bin/dynamicsadapt deploy evaluate --config "$DP" --calibration "$CAL" \
  --script --stream dev --detector simulated --run-dir "runs/$(date +%Y%m%d%H%M)_dp_script"
D="runs/$(date +%Y%m%d%H%M)_dp_demos"
.venv-rl/bin/dynamicsadapt deploy demos --config "$DP" --calibration "$CAL" --run-dir "$D"
B="runs/$(date +%Y%m%d%H%M)_dp_bc"
.venv-rl/bin/dynamicsadapt deploy bc --config "$DP" --demos "$D/demos.npz" --run-dir "$B"
.venv-rl/bin/dynamicsadapt deploy evaluate --config "$DP" --calibration "$CAL" \
  --policy "$B/policy.pt" --stream dev --detector simulated --run-dir "runs/$(date +%Y%m%d%H%M)_dp_bc_dev"
P="runs/$(date +%Y%m%d%H%M)_dp_train"
.venv-rl/bin/dynamicsadapt deploy train --config "$DP" --calibration "$CAL" \
  --demos "$D/demos.npz" --policy "$B/policy.pt" --run-dir "$P"
# Read selection.json (and fallback.json if present) to locate the selected policy.
SELECTED=/path/from/selection/policy_itNNN.pt
for LABEL in bc selected; do
  POLICY="$B/policy.pt"
  if [ "$LABEL" = selected ]; then POLICY="$SELECTED"; fi
  .venv-rl/bin/dynamicsadapt deploy evaluate --config "$DP" --calibration "$CAL" \
    --policy "$POLICY" --stream eval --detector simulated \
    --run-dir "runs/$(date +%Y%m%d%H%M)_dp_${LABEL}_eval"
  .venv-rl/bin/dynamicsadapt deploy evaluate --config "$DP" --calibration "$CAL" \
    --policy "$POLICY" --stream render_eval --detector image \
    --run-dir "runs/$(date +%Y%m%d%H%M)_dp_${LABEL}_render"
done
```

Stop on a calibration or script/demo gate failure and consult the user. Training
checks the BC gate (6/20 strict development successes or 30% stochastic online),
retains every iteration and selects by strict successes, lenient successes, then
earlier checkpoint. The registered three-iteration regression condition stops and
retains the run, then starts a separate `_gripper` run from BC with exploration
restricted to outputs 3 and 4. `--stochastic` enables sampled evaluation.

Compare BC and selected results, including detection errors and stop reasons.
Final acceptance requires the selected policy to reach 45/50 strict successes on
`eval` and 18/20 on `render_eval`. This is simulation-only evaluation of one fixed
robot/camera, nominal physics, fixed cube yaw and force-mode gripping. Calibration
uses simulator frames; camera noise, latency and lighting variation are absent.
Training rewards and termination use privileged simulator data.
