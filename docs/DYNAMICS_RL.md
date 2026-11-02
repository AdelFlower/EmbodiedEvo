# Cube-to-plate neural policy

This independent experiment trains a neural policy for the existing red-cube-to-plate
task. The actor observes simulator ground-truth object poses and velocities plus robot
state. It therefore has more information than the RGB code policy; its results are
not RoboDojo formal Test results or a comparison under equal observation conditions.

The source environment keeps normal friction. Cube and plate X/Y coordinates are
independently sampled within 3 cm of their original positions; their heights,
orientations and physical parameters remain unchanged. The original 25 Hz control,
500-action allowance and `dynamics-repair-score-v4` success checks remain in force.
The cube must be grasped, carried in the air, released and stably placed without robot
support. The training reward does not replace the native success signal.

## Setup

Use the existing provisioned RoboDojo/Isaac simulator image, assets and GPU described
in [INSTALL.md](INSTALL.md). The learner runs on the host CPU in a separate environment;
the simulator retains its pinned image and GPU environment. No model/API calls or new
container build are needed for this experiment.

```bash
python3 -m venv .venv-rl
.venv-rl/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv-rl/bin/python -m pip install -r requirements-rl.txt -e '.[dev]'
.venv-rl/bin/python -m unittest discover -s tests -v
```

The learner uses PyTorch 2.8.0 and Stable-Baselines3 2.9.2. Its optional imports do not
require these packages in the ordinary orchestration environment. CPU tests cannot
establish physical task success.

## Run

Run commands from the repository root. Each command requires a new explicit run
directory with a `YYYYMMDDHHMM` prefix; populated directories are never overwritten.
The default configuration is
[`tasks/DynamicsRL/train.example.json`](../tasks/DynamicsRL/train.example.json).

First check rendered/state-only teacher parity and 20 consecutive randomized resets:

```bash
PREFLIGHT_RUN="runs/$(TZ=Asia/Shanghai date +%Y%m%d%H%M)_rl_preflight"
.venv-rl/bin/python -m orchestrator.dynamics_rl preflight \
  --config tasks/DynamicsRL/train.example.json \
  --run-dir "$PREFLIGHT_RUN" --gpu 0
```

Train with that passing preflight evidence:

```bash
TRAIN_RUN="runs/$(TZ=Asia/Shanghai date +%Y%m%d%H%M)_rl_train"
.venv-rl/bin/python -m orchestrator.dynamics_rl train \
  --config tasks/DynamicsRL/train.example.json \
  --preflight-run "$PREFLIGHT_RUN" --run-dir "$TRAIN_RUN" --gpu 0
```

Omitting `--preflight-run` runs the checks at the beginning of training. The training
stage includes demonstration collection, BC, online RL and development validation,
and has an eight-hour wall-time limit by default. The online interaction limit is
100,000 actions. A persistent simulator serves training episodes; training disables
camera capture and video encoding, while keeping physics and scoring active.

The RL reset adapter omits linear/angular velocity writes to the kinematic plate
while retaining its native pose reset and the dynamic cube's velocity reset. This
avoids accumulated PhysX errors from writing velocities to a kinematic body. Run a
new preflight after reset-adapter changes; `reset_checks` can increase the default
20 checks for a longer regression check. An earlier passing check is not evidence
for changed simulator code.

After a network has been selected using development validation, freeze and evaluate it:

```bash
EVAL_RUN="runs/$(TZ=Asia/Shanghai date +%Y%m%d%H%M)_rl_evaluate"
.venv-rl/bin/python -m orchestrator.dynamics_rl evaluate \
  --config tasks/DynamicsRL/train.example.json \
  --checkpoint "$TRAIN_RUN/checkpoints/best_rl.pt" \
  --run-dir "$EVAL_RUN" --gpu 0
```

Final evaluation is separate from the eight-hour training budget; allow approximately
three to four additional hours on the current machine. It copies the selected actor
before selecting the 50 held-out layouts, starts a fresh simulator per layout, and
records native camera videos. No teacher, online updates or within-episode reset is
used. Acceptance requires at least 45 successes among all 50 planned attempts.

Replay a standalone checkpoint in one fresh randomized scene:

```bash
.venv-rl/bin/python -m orchestrator.dynamics_rl rollout \
  --config tasks/DynamicsRL/train.example.json \
  --checkpoint "$TRAIN_RUN/checkpoints/best_rl.pt" \
  --run-dir "runs/$(TZ=Asia/Shanghai date +%Y%m%d%H%M)_rl_rollout" --gpu 0
```

`best_rl.pt` exists after an RL checkpoint has completed development validation. A
stopped or failed run may instead have only `bc.pt` or `last.pt`; inspect the training
result before choosing an artifact. A BC-only checkpoint must not be described as an
RL-trained policy. Single-scene rollout is a visualization, not the 50-scene acceptance
test. Do not tune the model using final evaluation layouts or results.
The standalone `rollout` command uses its own random-layout namespace, disjoint
from the final evaluation set, so replay cannot expose a later acceptance scene.

To check transfer without learning, pass `--friction-scale 0.3 --episodes 5` to
`rollout`. The host freezes the checkpoint and runs five fresh, rendered simulators
with cube static/dynamic friction `0.3/0.3`; mass, scoring and action limits remain
unchanged. The fixed friction survives episode initialization and is verified against
live PhysX materials. Each attempt keeps its result, state/action trace and videos,
including failures. `inference-config.json` and `evaluation-result.json` record the
friction scale. These flags also work with `evaluate`; `train` and `preflight` reject
non-nominal friction. A short transfer rollout is separate from 50-scene acceptance.

## Teacher and learning

The default `teacher_policy` points to the local R1 development reference archived at
`runs/202610301439_dynamics_calibration_v4_0.3/policies/reference/policy.json`. This
archive must be present; it is runtime evidence and is not bundled with the source
distribution. The new run records its original text and provenance as supplied
development knowledge, separately from learned Skills/Lessons.

The adapted teacher translates R1's grasp and placement waypoints with the measured
initial cube/plate positions. It preserves R1 interpolation and timing and uses the
same fixed-left-arm action decoder as the learner. It must be physically validated;
the adapted controller is not an unchanged replay of the archived program.

The collector makes 128 planned attempts and records every outcome. Successful,
complete episodes supervise behavior cloning; all valid transitions, including
physical failures, can train the critic. It never retries a failed layout to obtain
a desired number of successes. Existing run logs are not treated as complete
observation-action datasets.

To continue an interrupted collection, set `demonstration_source` in a new config
to the previous run directory and set `max_train_hours` to the **remaining** agreed
budget. Start `train` with that config and a new timestamp-prefixed `--run-dir`;
use new passing preflight evidence after any simulator fix. This is an explicit
continuation of data collection, not an optimizer/checkpoint resume or an automatic
retry.

The importer requires the same demonstration count, seed, position radius, full
ordered layout plan, observation/action contract, teacher adaptation and supplied
teacher text. It inherits the existing manifest prefix, including every failed or
incomplete attempt, and collects only the unattempted suffix. For example, a prefix
of 64 attempts in a 128-attempt plan leaves 64 new attempts; an inherited failed
scene is not repeated. BC still uses only complete successes.

Inherited `data_path` entries refer to the original NPZ files by absolute path;
keep the source run accessible and unchanged. `demonstration-inheritance.json`
records the source, inherited outcome counts, next plan index and remaining time
budget. Empty failed-attempt archives stay in the manifest and contribute no
training samples. The new run initializes a fresh learner after collection.

The actor is a two-layer, 256-unit MLP. Its 57-dimensional state includes object
poses/velocities, right-arm joints/velocities, end-effector pose, actual gripper
opening, relative positions, previous action and remaining action allowance. The
7-dimensional output is absolute right-arm XYZ, a rotation vector relative to the
downward reference orientation, and gripper opening. The native IK/controller executes
each decoded target. Contacts and private scorer progress can shape rewards but are
excluded from the actor observation.

BC initializes the actor before genuine SAC updates. SAC uses 25% demonstration replay,
a decreasing BC auxiliary loss on successful demonstrations, and small initial
exploration variance; it does not begin with random-action warmup. Default batch size
is 256, learning rate is `3e-4`, and discount is `0.99`. Both success and the native
500-action cutoff stop critic bootstrap. Twenty separate development layouts select
the RL checkpoint; final held-out scenes do not participate in that selection.

The default `bc_mode: "robust"` addresses closed-loop failures observed with ordinary
one-step action MSE. It uses a generic clock-conditioned network initialization,
logit-space supervision, and actor gradient masks for endogenous shortcut inputs.
The encoded observation remains unchanged; zeroed actor weights are saved with the
network, while the critic retains the full observation. This adds no trajectory code
or teacher calls to inference. Terminal-hold augmentation supervises only the actor;
it is never inserted as simulated transitions in critic replay or demonstration counts.
The robust actor has 13 effective inputs: cube XYZ, plate XYZ, right end-effector XYZ,
plate-minus-cube XYZ, and remaining action allowance. The other 44 encoded columns
have zero first-layer weights. Its behavior therefore depends on simulator state and
episode timing; it has not been validated for arbitrary starting robot configurations,
delays, external disturbances, visual observations, or friction changes.
The default is 60,000 BC updates and `initial_log_std: -6` for small initial SAC
exploration. Set `bc_mode: "classic"`, `bc_updates: 20000`, and `initial_log_std: -4`
to reproduce the original baseline initialization. The robust mode uses an actor
learning rate of `1e-6`, a critic learning rate of `3e-4`, and an auxiliary BC weight
decaying from 10000 to 1000 over 50,000 online actions. Development stress tests showed
that a `3e-4` actor update destroyed the pretrained clock-conditioned controller;
the separate actor rate and stronger demonstration loss limit this drift. The
original SAC baseline used actor learning rate `3e-4` and BC weights 10 to 1.
Online BC rotation-logit errors receive weight 10, independently of the unchanged
pretraining objective. Gaussian exploration log standard deviation is bounded above
at `-6`; otherwise actual training increased position noise beyond 6 mm and rotated
the gripper by up to 17 degrees. The bound changes only stochastic training;
standalone actors use deterministic inference. Set `actor_log_std_max: null` to
restore SB3's original upper bound. The default checks development performance at
native episode boundaries after every 500 online actions and saves checkpoints
every 500 actions, allowing selection before task performance deteriorates.
Development results for both
initializations must be reported separately.

## Evidence and failures

Useful host artifacts include:

| Artifact | Meaning |
| --- | --- |
| `config.json`, `contract.json` | Frozen experiment and observation/action configuration |
| `preflight.json` or `preflight-reference.json` | Actual simulator compatibility checks |
| `teacher-provenance.json`, `supplied-teacher.json` | Teacher origin and supplied source |
| `demonstrations.json`, episode `transitions.npz` | All demonstration outcomes and transitions |
| `demonstration-inheritance.json` | Explicit prior-attempt prefix and remaining collection budget |
| `checkpoints/bc.pt`, `best_rl.pt`, `last.pt` | Standalone inference actors; no teacher needed |
| `checkpoints/*.zip`, `last-replay.pkl` | Full learner checkpoints and final replay state |
| `bc.jsonl`, `rl-metrics.jsonl`, `validation.jsonl` | BC losses, actual RL updates and validation results |
| `training-result.json`, `training-curves.png` | Budget/status and training plots |
| `evaluation-result.json`, `evaluation-scenes.json` | All held-out attempts and acceptance result |
| Per-worker/per-episode logs and videos | Native physics, actions, raw feedback and failure evidence |

A timeout, crash or missing report remains an incomplete, unclassified attempt unless
there is evidence for a more specific cause. It is retained in the final denominator;
there is no automatic rerun or silent retry. If training reaches its budget without
meeting the target, retain the available checkpoints and report the measured shortfall.
An implemented pipeline or passing CPU tests alone do not establish a trained policy's
success rate.

All runtime artifacts stay under `runs/`. The original four primitives, isolated
coding-agent workspaces, formal Playground/Test transfer protocol, third-party recipes
and `vendor/` sources remain unchanged.
