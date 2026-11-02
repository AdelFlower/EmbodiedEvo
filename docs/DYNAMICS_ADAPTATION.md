# Source-simulator calibration and neural policy adaptation

This experiment converts a fixed SAC actor (P0) to PPO and continues training
after an agent calibrates source dynamics. It reports target performance separately
from the evidence for the agent's diagnosis. The [engineering specification](DYNAMICS_ADAPTATION_SPEC.md)
defines the protocol; [DynamicsRL](DYNAMICS_RL.md) describes the policy and setup.
These results are separate from RoboDojo formal Test scores.

Revision 1 replaces SAC fine-tuning after the SAC smoke collapsed P0 even at nominal
physics. Conversion preserves P0's deterministic actions. PPO uses fixed Gaussian
exploration, separate actor/value learning rates, and a value-only first rollout
in every round. Later updates use SB3's clipped objective and `target_kl` stopping,
then exact Gaussian KL on the rollout observations. Actor steps exceeding
`max_update_kl` are halved up to ten times, or fully reverted; value updates and
Adam state are retained. The robust input mask stays active. Parent action drift
is recorded. These safeguards still require the overseer's GPU stability checks.

Run from the repository root using `.venv-rl`. GPU commands need the provisioned
simulator and run serially. Formal calibration also needs the configured Codex
container and model credentials. Every command takes a timestamp-prefixed
`--run-dir`; keep all runtime artifacts under `runs/`, outside Git.

## Phase 0: establish the gap and freeze the procedure

Start with a complete copy of
[`adaptation.example.json`](../tasks/DynamicsAdaptation/adaptation.example.json).
Its training values are placeholders until procedure development is complete.
Use a distinct run directory for every attempt; never overwrite a failed attempt.

```bash
CONFIG=/path/to/adaptation.json
PREFLIGHT_RUN="runs/$(date +%Y%m%d%H%M)_adaptation_preflight"
.venv-rl/bin/python -m orchestrator.dynamics_adaptation preflight \
  --config "$CONFIG" --run-dir "$PREFLIGHT_RUN" --gpu 0
.venv-rl/bin/python -m orchestrator.dynamics_rl preflight \
  --config tasks/DynamicsRL/train.example.json \
  --run-dir "runs/$(date +%Y%m%d%H%M)_adaptation_reset_regression" --gpu 0
.venv-rl/bin/python -m orchestrator.dynamics_rl rollout \
  --config tasks/DynamicsRL/train.example.json \
  --checkpoint runs/202610302107_dynamics_rl_train_v4/checkpoints/best_rl.pt \
  --friction-scale 0.25 --episodes 10 \
  --run-dir "runs/$(date +%Y%m%d%H%M)_adaptation_screening" --gpu 0
```

Require the preflight checks in spec §10.2 and the reset regression to pass.
The `preflight` command records P1–P3. The overseer separately runs revision-1 P4
(two converted PPO rounds, save/reload, cumulative counters and exact KL) and P5
(25,000 nominal actions, then 5/5 nominal deployment successes). Old SAC P4 evidence
does not qualify. Pass P5 before developing the procedure on a non-target shift.
Screening measures the transfer gap: stop if P0 succeeds on at least 9/10 layouts
(D2), and record any target revision before a formal run.

D1 option A is signed off: develop training settings on a non-target shift where
P0 fails, chosen without target-result tuning. `DEV_SHIFT.json` contains all three
parameter keys (`friction_scale`, `mass_scale`, `restitution`). Each candidate
procedure gets a separate run:

```bash
.venv-rl/bin/python -m orchestrator.dynamics_adaptation trainability \
  --config "$CONFIG" --parameters /path/to/DEV_SHIFT.json \
  --run-dir "runs/$(date +%Y%m%d%H%M)_adaptation_development" --gpu 0
```

Freeze the chosen training settings, choose `TRAINABILITY_RUN`, and set the
config's `trainability_run` to that directory before the target gate and formal run.

```bash
TRAINABILITY_RUN="runs/$(date +%Y%m%d%H%M)_adaptation_trainability"
.venv-rl/bin/python -m orchestrator.dynamics_adaptation trainability \
  --config "$CONFIG" --run-dir "$TRAINABILITY_RUN" --gpu 0
```

`trainability.json` contains the chain, selected network and independent
`trainability_eval` results. Require at least 45/50 successes. A failure stops the
formal experiment and describes the training procedure, not the agent. Any later
procedure change is an explicitly recorded protocol revision.

## Phase 1: calibration and matched control

```bash
RUN_DIR="runs/$(date +%Y%m%d%H%M)_adaptation"
.venv-rl/bin/python -m orchestrator.dynamics_adaptation run \
  --config "$CONFIG" --preflight "$PREFLIGHT_RUN" --run-dir "$RUN_DIR" --gpu 0
```

Stages are `freeze`, `baseline`, `calibration`, then `control`. The host copies P0
and converts its training state to PPO, persists the development layouts, runs the paired baseline, starts the agent, and
then trains the nominal control. Control matches every completed calibration train
sub-job's actions, round index, learner seed and online layouts, even if that
round's later target deployment failed or stopped. It deploys only if the matching
calibration `deploy/` sub-job exists. A completed train with no deployment still
continues the control chain, but is excluded from selection. Failed or stopped
training is listed in `control/result.json` and omitted from control's schedule. Control uses
its own chain, the same target layouts and selection rule, with no probes, agent
or wall-time limit. Ties select the earlier candidate, including P0; the agent's
preferred network never overrides host selection.

`--stage NAME` executes only an unattempted stage in an existing run whose earlier
stages completed. It refuses any existing stage directory, including failed ones.
Use the original config and agent kind. Revision 1 requires a fresh smoke run:

```bash
SMOKE_RUN="runs/$(date +%Y%m%d%H%M)_adaptation_r1_smoke"
.venv-rl/bin/python -m orchestrator.dynamics_adaptation run \
  --config tasks/DynamicsAdaptation/smoke.json --preflight "$PREFLIGHT_RUN" \
  --run-dir "$SMOKE_RUN" --gpu 0 \
  --scripted-agent tasks/DynamicsAdaptation/smoke_agent.py
```

The test-only host script first asserts that training without reproduction is
refused. It selects the first failing target-baseline layout and probes N000 at
friction 0.25, deliberately using the true target value for deterministic
reproduction. It also probes mass 2.0 on that layout, trains for 2,000 actions at
friction 0.25, waits for completion, and submits. The budget permits exactly two
diagnostic episodes; smoke evaluation uses two layouts. This checks the loop and
makes no scientific diagnosis claim. Its training block equals the example's.

Before every agent training round, the service requires a completed probe of the
current parent network at exactly the requested parameters (with nominal values
filled in). Every probed outcome must match that network's target-development
outcome, including at least one target failure. Incomplete target episodes count
as failures; incomplete probes do not qualify. No target failure or deployment
record means training is refused. `status` exposes `target_dev_outcomes` per
network (`null` without deployment evidence). Agents must also compare failure
modes in their analysis. Refusals consume no budget. The host control and
trainability chain have no reproduction requirement.

`actions` must be a multiple of `training.n_steps`, within the existing action
bounds. Configuration loading also requires `checkpoint_interval` and
`budgets.actions_per_round` to be multiples of `n_steps`.

## Phase 2: frozen evaluation and diagnosis review

D3 includes the trainability-selected `oracle` as a reference outside the
agent-versus-control comparison and agent budget:

```bash
.venv-rl/bin/python -m orchestrator.dynamics_adaptation evaluate \
  --run-dir "$RUN_DIR" --gpu 0 --oracle "$TRAINABILITY_RUN"
```

Evaluation requires completed calibration and control and refuses an existing
`evaluation/`. It renders 50 `adaptation_eval` layouts per distinct selected
network, with a fresh simulator per episode. Arms sharing P0 reference its single
evaluation. `accepted` means calibration achieved at least 45/50 successes.
Errors propagate, stop evaluation and retain all planned episodes, including
unstarted episodes, in `evaluation/results.json`; nothing retries. A completed
evaluation returns exit code 1 if calibration misses the threshold.

The researcher then writes root-level `diagnosis-review.json`, following spec
§10.3. Supporting a diagnosis requires naming decreased friction, pre-registered
predictions that distinguish friction from mass, and observations matching the
predicted direction. A nearby estimate without discrimination is ambiguous.

```bash
.venv-rl/bin/python -m orchestrator.dynamics_adaptation summarize --run-dir "$RUN_DIR"
```

This reads evidence without running jobs or modifying it, writes `summary.json`,
and prints performance, round tables, costs and diagnosis separately. Without a
host review, diagnosis is `not_reviewed`, regardless of success or agent confidence.

## Run directory and outputs

| Path | Meaning |
|---|---|
| `run.json`, `config.json`, `source/` | Invocation, frozen host configuration and source snapshot |
| `layouts.json`, `networks/N000/` | Paired layouts, copied P0 `policy.pt` and `sac-training.zip`, and converted PPO `training.zip` |
| `freeze/result.json`, `baseline/{source,target}/`, `baseline/result.json` | Stage completion and paired P0 development outcomes |
| `calibration/workspace/` | Agent-visible instructions, raw exported observations, notes and submission |
| `calibration/jobs/J*/`, `calibration/requests.jsonl` | Accepted jobs, pre-registered hypotheses/predictions, sub-job evidence; request log includes refusals |
| `calibration/networks/`, `control/networks/` | Independent continued-training chains and counters |
| `control/rounds/R*/{train,deploy}/`, `control/result.json` | Matched jobs, skipped calibration training attempts and elapsed time |
| `{calibration,control}/selection.json` | Ordered candidates and host-selected winner |
| `calibration/result.json`, `diagnosis.json`, `agent.log`, `agent_sessions/` | Branch end reason, budgets, model usage, accepted self-report and agent records (all under `calibration/`) |
| `evaluation/{P0,calibration,control,oracle}/` | Distinct-policy rollout jobs with episode records and raw videos; shared P0 has no duplicate job |
| `evaluation/results.json` | All arms, shared evaluation references, successes, incomplete counts, episodes, timing and acceptance |
| `diagnosis-review.json`, `summary.json` | Independent researcher review and derived report |

PPO `rl-metrics.jsonl` records every rollout update, including value-only warm-up,
epochs, losses, approximate/exact KL, accepted `step_fraction`, and
`drift_from_parent`. Public `losses.csv` contains `online_steps, value_only,
policy_loss, value_loss, exact_kl, step_fraction, drift_from_parent`.
`online_steps` continues P0's counter. `policy_updates` counts accepted PPO policy
updates, excluding warm-up and fully reverted steps; N000 starts at zero. The
inference payload retains its `rl_updates` field with this PPO count. Checkpoints
are evidence only and are saved after updates at configured rollout boundaries.

Summary costs include wall time per stage, recorded GPU job time by kind, model
tokens, requested training actions, completed training actions, `policy_updates` deltas and
planned target/diagnostic episodes. Unfinished training and jobs without records
are listed explicitly; recorded counters and timings are lower bounds when these
lists are nonempty. Five-layout results are development signals, not estimates
of success rates. Host configuration, networks, private feedback and scene plans
stay outside the agent workspace.

## Required disclosures

Report performance restoration and diagnosis support as separate conclusions.
Following spec §13, disclose P0's R1 teacher provenance: R1 was designed by the host
for friction 0.3, while P0 demonstrations and online SAC ran at friction 1.0.
P0's recorded source evaluation is `runs/202610302128_dynamics_rl_eval` (50/50);
the summary cites the configured source-evaluation path. The target 0.25 was
chosen after observing P0 succeed at 0.3 and fail on one layout at 0.25. Report
the subsequent screening, trainability outcome and every procedure revision.

Disclose exact identification in this deterministic sim-to-sim setting: matching
parameters can reproduce target trajectories, which is not evidence of real-robot
calibration. Explain the friction–mass grasp confound and the evidence separating
them. Report control's lack of probes and agent time, its unrestricted wall time,
and the oracle's separate reference status. Retain every attempt in its planned
denominator. Crashes, timeouts and missing reports remain unclassified without
specific causal evidence; only confirmed infrastructure faults can justify a
separately documented rerun preserving the original evidence.
