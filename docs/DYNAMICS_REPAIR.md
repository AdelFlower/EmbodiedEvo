# Unknown Dynamics Changes and Code Policy Repair

This experiment uses a separate red-cube-to-plate task to study whether a coding
agent can investigate transfer failure by modifying diagnostic copies and repair
its original complete code policy. For the subsequent neural-policy adaptation
experiment, see the [user guide](DYNAMICS_ADAPTATION.md) and
[engineering specification](DYNAMICS_ADAPTATION_SPEC.md).

This is a research Playground workflow, not the original RoboDojo formal Test;
its results must not be included in that benchmark. Existing commands, tasks,
the four handwritten primitives, and formal knowledge-transfer rules remain
unchanged. The new task permits 500 native actions per episode. Success requires
grasping, lifting, carrying, releasing, and stable placement inside the plate.
Pushing the cube into the plate or holding it above the plate does not count.
During the stability window no robot link may touch the cube: fingers, palm,
wrist, and either arm's other links all count as continued support.
Touching the table after a carry invalidates that carry, even if a finger remains
in contact. A new bilateral grasp, lift, and carry can establish valid evidence again.

Code and CPU tests cannot prove that physical materials are effective or that a
model can complete the loop. GPU contact behavior, task solvability, source-policy
transfer degradation, and successful repair require evidence from actual runs.
Do not report unperformed GPU runs, container builds, or model calls as passed.

## Environment and configuration

Use the repository's installed environment, provisioned local RoboDojo/Isaac Sim,
assets, an available NVIDIA GPU, and the agent container. Model runs also require
the existing credential setup. See [INSTALL.md](INSTALL.md).
Do not commit runtime artifacts, machine configuration, credentials, or Python
environments to Git.

Run commands from the repository root. The example configuration is
[`tasks/DynamicsRepair/experiment.example.json`](../tasks/DynamicsRepair/experiment.example.json).
It uses the existing `RoboDojo` agent model and container settings and specifies
the simulator Python separately. The host's `target_friction_scale` fixes the
target perturbation before the run; it must be greater than zero and less than
one. The example configuration freezes `0.3`, the scale of the development
calibration below; a configuration that omits the field falls back to `0.25`. This value and its provenance stay outside the agent
workspace. Source learning starts directly and creates the original policy.

| Setting | Default |
|---|---|
| `agent_task` | `RoboDojo` |
| `simulator_python` | `.venv-robodojo/bin/python` |
| `seed` | `0` |
| `action_limit` | `500` |
| `target_friction_scale` | `0.3`, frozen before the run |
| `budgets.executions` | `100` |
| `budgets.episodes` | `20` |
| `budgets.target_episodes` | `8` |
| `budgets.actions` | `10000` |
| `execution_timeout_sec` | `1200` |
| `agent_timeout_sec` | `0`, no overall agent timeout |
| `startup_timeout_sec` | `1200` |
| `feedback_fps` | `25` |

Timeouts must be finite numbers: execution and startup timeouts are positive,
while an agent timeout of zero disables its overall time limit. `feedback_fps`
must be a finite positive whole number; `25.0` is normalized to `25`.

`startup_timeout_sec` bounds broker readiness and, for every trial, simulator
launch, worker connection, scene construction, and the initial reset. After the
first reply, `execution_timeout_sec` bounds each submitted program inside the
worker, and the host waits at most that value plus 60 seconds for its reply.
Source learning, repair trials, frozen replays, calibration, and the startup
check all use the same two settings. An expired startup is an unclassified
failed trial; it is not retried.

Every invocation requires an explicit `--run-dir` whose name starts with
`YYYYMMDDHHMM`. The timestamps below illustrate the format; choose a fresh run
directory for actual use. `run` does not automatically resume or reuse a
directory that already contains data.

Before spending model calls on a newly provisioned simulator, run the source
startup check:

```bash
.venv/bin/python -m scripts.smoke_dynamics \
  --config tasks/DynamicsRepair/experiment.example.json \
  --run-dir runs/202610281200_dynamics_startup \
  --gpu 0
```

This uses the same simulator worker and container transport, creates one source
episode, reads the instruction and observation, and submits one action holding
the measured joints. It calls no coding agent and never retries. A passing check
establishes startup and one control step, not grasp success or target
recoverability. Inspect `smoke.json` and the trial's native logs if it fails.

For the pinned Isaac Sim 5.1 image with URDF importer `2.4.30`, the repository
adapter handles the known IsaacLab experience pin of `2.4.31` by creating a
temporary experience copy with the installed version. It retains the remaining
settings and original extension search paths, logs the applied compatibility
change, and leaves `vendor/` untouched. Other version combinations keep their
original launch configuration.

Two further model-free checks use the GPU but no agent. The material probe starts
one fresh simulator per condition and records the cube's live PhysX shape
materials, the materials named by contact reports, a fixed-speed slide of the
resting cube across the table before and after a Playground reset, and a
side-by-side construction-order comparison on two extra cubes:

```bash
.venv/bin/python -m scripts.probe_dynamics_physics \
  --run-dir runs/202610281210_dynamics_material_probe --gpu 0
```

The frozen-policy replay re-freezes an archived policy unchanged and replays it
three times in fresh source and three times in fresh target simulators at the
configured `target_friction_scale`. It reports whether the experiment's
thresholds would be met, but never qualifies an experiment, starts repair or
changes the perturbation:

```bash
.venv/bin/python -m scripts.replay_dynamics_policy \
  --config tasks/DynamicsRepair/experiment.example.json \
  --policy-archive runs/<earlier run>/policies/P0 \
  --run-dir runs/202610281220_dynamics_p0_replay --gpu 0
```

The target sweep is a development calibration stage that uses an archived probe
policy instead of a reference policy. It re-freezes the probe unchanged, requires
`3/3` in three fresh source replays, then replays the candidate scales
`0.3, 0.4, 0.5, 0.6, 0.7, 0.8` in ascending order, three fresh target replays
each. A candidate passes when all three are complete physical failures. The sweep
stops at the first candidate that does not pass and selects the highest passing
one. The rule is fixed in `scripts/sweep_dynamics_target.py` before any replay;
`--candidates` replaces the default grid and is recorded in `result.json`.
The selection shows only that the probe fails there; it does not establish
recoverability, and an experiment still learns its own `P0`:

```bash
.venv/bin/python -m scripts.sweep_dynamics_target \
  --config tasks/DynamicsRepair/experiment.example.json \
  --policy-archive runs/<earlier run>/policies/P0 \
  --run-dir runs/202610281230_dynamics_target_sweep --gpu 0
```

Sweep `runs/202610292100_dynamics_target_sweep` (`dynamics-repair-v3`, probe
`P0` of run `202610291735`, no model calls):

| Scale | Effective friction | Probe successes | Incomplete |
|---|---|---|---|
| source `1.0` | `0.5/0.5` | `3/3` | `0` |
| `0.3` | `0.15` | `0/3` | `0` |
| `0.4` | `0.20` | `0/3` | `0` |
| `0.5` | `0.25` | `0/3` | `0` |
| `0.6` | `0.30` | `0/3` | `0` |
| `0.7` | `0.35` | `3/3` | `0` |

The rule selected `0.6` and stopped at `0.7`; `0.8` was not replayed. Rescoring
these recorded replays under `dynamics-repair-score-v4` leaves every outcome
unchanged: each target failure is a slip or a drop, not a speed check. The source
replay reproduced the `202610291735` qualification trajectory exactly (positions,
velocities, orientations and contact points) apart from the cube-side contact
material name. The probe's failure boundary therefore lies between `0.6` and
`0.7` for this probe; a newly learned `P0` may have a different boundary.

Run `202610292357_dynamics_run` then froze `0.5`. Its newly learned `P0` passed
`3/3` in the source and was scored `0/3` in the target under score v3, but in all
three target replays the cube rested on the plate after a valid release; only
the angular speed check failed (see score v4 below). Under v4 those replays are
successes, so the transfer failure was not established and the run is
superseded. The example configuration then froze `0.4`, chosen by the
experimenter, not by the sweep rule. A model-free replay of that run's `P0` at
`0.4` under score v4 (`runs/202610300113_dynamics_p0_replay_v4`) succeeded `3/3`
in the source and failed `0/3`, all complete, in the target: the cube slipped
from the fingers about 2 cm into the lift and stayed on the table. This is
evidence about that policy only; a new run learns its own `P0`.

Run `202610300143_dynamics_run` at `0.4` confirmed that limitation: its newly
learned `P0` succeeded `3/3` in the source and `3/3` in the target, so the run
stopped as `transfer_failure_not_established`. The target successes are valid
under score v4: a bilateral grasp, a 12.5 cm airborne carry, release 1.6 cm above
the plate and 13 supported cycles inside it. All three learned policies close the
gripper fully and make the same four-point, face-on contact on each finger, yet
their failure thresholds differ: about `0.65`, `0.45` and below `0.4`. The
robustness of a newly learned `P0`, and therefore whether a fixed scale
establishes a transfer failure, varies between source-learning runs.

Sweep `runs/202610300235_dynamics_target_sweep_p0_3` (candidates
`0.2, 0.25, 0.3, 0.35`, probe: that run's `P0`, no model calls) found its boundary
between `0.35` and `0.4`. It succeeded `3/3` in the source and failed `0/3`, all
complete, at every candidate. At `0.2` to `0.3` the cube slipped from the
bilateral grasp within 1 cm of lift. At `0.35` no bilateral grasp formed: the
fingers closed beside the cube. Contact impulses show about 25 N of normal force
per finger in every grasp of these policies, holding or slipping. Coulomb
friction would predict a capacity of about 2.3 times the cube's weight at `0.3`
and 2.9 times at `0.4`, yet the grasps slipped near those scales.

The recorded trajectories explain both failures by where the cube is when the
fingers close. The policy's finger pads centre 1.2 mm below its commanded `y`.
In the source the cube stays at `(0.2, -0.13)`, 8.8 mm from the pad centre along
the pad. In the target the cube moves on the table during the native reset's 300
settling steps and the first 40 control steps, with no robot contact. Its solver
angular velocity reaches 2 to 5 rad/s while its tilt stays below 0.1°. At the
grasp (control step 120) it sits 6.6 mm (`0.2`), 6.9 mm (`0.25`), 11.4 mm (`0.3`)
and 24.1 mm (`0.35`) from the pad centre. The drift is not monotonic in friction,
and it is identical in every fresh replay at the same scale. An off-centre grasp
leaves a gravity torque about the grip axis; the cube pivots out of the fingers
within 1 cm of lift (tilt 12° at `0.3`). At `0.35` the cube had
moved past the pads entirely.

#### Development calibration at `0.3`

The experimenter chose `0.3` after the runs above, before any reference policy
existed. A host-written development reference was then developed against it, in
`runs/202610300825_dynamics_reference_dev`. That directory holds each candidate's
program and one fresh target replay per candidate; replays at one scale are
deterministic, so one replay was used per candidate while developing. No model
was called. Every candidate keeps `P0 #3`'s skill and carry orientation and grasps
centred on where the cube settles at `0.3`, `(0.1972, -0.1098)`:

| Candidate | Difference | Outcome at `0.3` |
|---|---|---|
| A | pads 6 mm lower on the faces; lift 50, carry 80 steps | held through the lift; fell out 110 steps after lift began |
| B | as A, at `P0 #3`'s grasp height | pivoted during the lift; fell during the carry |
| C | as A, higher release | incomplete: simulator rendering failure at startup with three concurrent simulators; not rerun |
| D | as A, carry tilted 30° about the grip axis (`P0 #1`'s orientation) | fell out 105 steps after lift began |
| E | as A, carry over 150 steps | fell out 125 steps after lift began |
| F | as A, 20-step lift, one 35-step carry-and-tilt, 10-step lower, release | success at control step 231 |
| G | as F, pads 2 mm lower still | success at control step 231 |

At `0.3` even a centred grasp does not hold the cube statically. From the lift
on, the cube creeps down between the pads: 3.4 mm over the first 50 steps, then
faster, until it falls out about 105 to 125 steps after the lift begins, however
the carry is oriented or paced. The normal force stays at about 25 N per finger
throughout. Holding the cube still would need an effective coefficient of
`mg/(2N) ≈ 0.07`, so the effective grip friction at `0.3` is about half of the
nominal `0.3 × 0.5 = 0.15`. The reference succeeds by keeping the cube airborne
for about 65 control steps and releasing it from 1 cm above the plate floor.

`F` is the frozen reference. It uses the four primitives and the same skill code
as `P0 #3`. Its grasp position was measured from recorded trajectories, so it
shows that legal control can solve the `0.3` target, not that an agent will find
this strategy. A repair agent sees the displaced cube in the camera images. The
reference stays outside every agent workspace.

Calibration `runs/202610300855_dynamics_calibration_0.3` (no model calls) used
baseline `P0 #2` (run `202610292357`), the only learned policy not yet replayed
below `0.4`:

```bash
.venv/bin/python -m orchestrator.dynamics_repair calibrate \
  --config tasks/DynamicsRepair/experiment.example.json \
  --run-dir runs/202610300855_dynamics_calibration_0.3 --gpu 0 \
  --baseline-policy runs/202610292357_dynamics_run/policies/P0/policy.json \
  --reference-policy runs/202610300825_dynamics_reference_dev/iter03-F/policy/policy.json \
  --scales 0.3
```

| Replays | Successes | Incomplete |
|---|---|---|
| baseline, source | `3/3` | `0` |
| baseline, target `0.3` | `0/3` | `0` |
| reference, target `0.3` | `3/3` | `0` |

The status is `calibrated` with `selected_alpha` `0.3`. The live cube shape
material was `0.3/0.3` in every target replay. The three reference replays repeat
the development replay of `F` exactly. The baseline closes where the cube rests in
the source; at `0.3` the first finger pushes the settled cube aside and no grasp
forms. All three learned policies therefore fail at `0.3`: `P0 #1` and `P0 #3` in
their sweeps (rescored under score v4 with unchanged outcomes) and `P0 #2` here.
A new run that passes `--calibration runs/202610300855_dynamics_calibration_0.3`
records `recoverability: "validated_by_reference"`. Whether its newly learned
`P0` fails at `0.3` remains that run's own result.

The cube's motion during settling is a simulator behaviour. A resting cube on a
level table should not move; with no robot contact, its solver angular velocity
reaches 5 rad/s. It appears only at lowered friction, and it is identical in every
fresh replay at a given scale. Up to `dynamics-repair-v3` it was part of the
target condition the agents faced. Run `202610300959` showed that it misleads
diagnosis, and `dynamics-repair-v4` removes it (below). This calibration is
recorded under v3, so `run --calibration` rejects it for v4 runs.

#### Diagnostic note: run `202610300959_dynamics_run`

This note records defects found in that run. The run directory itself is
unchanged. The run used `dynamics-repair-v3` at `0.3`: its `P0` succeeded `3/3`
in the source and failed `0/3`, all complete, in the target.

| Branch | Episodes (target) | `P1` target | `P1` source | Status |
|---|---|---|---|---|
| intervention | 12 of 12 (6 of 6) | `0/3` | `3/3` | `repair_failed` |
| control | 6 of 12 (5 of 6) | `3/3` | `0/3` | `repair_succeeded` |

The intervention agent published a `P1` it had validated only in a diagnostic
copy at `friction_scale` `0.5`. It had 81 of 100 executions and 1,103 of 6,000
actions left; episodes were the binding limit. The control agent turned the
gripper so that a finger supported the cube from below, the strategy the
intervention agent had found in its trial `000006` (item 2).

1. **Stale nominal values.** Every v3 workspace's `nominal.json` reported cube
   friction `0.8/0.6`. v3 had changed the nominal material to `1.0/1.0`, but the
   literal written by `prepare_workspace` was not updated. Both branches of a run
   saw the same file, and this agent never reasoned from the absolute values, so
   the branch comparison is unaffected; the published value was still wrong.
2. **The release check was not stated.** In target trial `000006` the agent's
   sideways-support repair held the cube with both fingers from control step 109
   to 400, carried it 0.22 m, and left it resting 2 mm from the plate centre with
   no robot contact. The scorer rejected the carry at step 401: the tilted cube's
   centre was 25 mm from the plate centre but one corner was 44.6 mm out, past the
   40 mm inner radius, when the fingers lost contact. The agent saw only
   `success: false`. It guessed an unmet lift requirement and spent the next two
   target episodes on that guess. The check behaved as designed; the agent had no
   way to know it.
3. **Pre-contact drift as a false fingerprint.** Before any contact the target
   cube drifted 20 mm (`y` from `-0.130` to `-0.110`), while in the agent's
   diagnostic at `friction_scale` `0.25` it drifted 2 mm and at `0.5` 1.2 mm. The
   agent used the target's pre-grasp offset to compare conditions and found no
   diagnostic that matched it. It recorded friction as unidentified, although a
   diagnostic at `0.3` has exactly the target's settings. The drift is also visible
   in the target's first camera frame, before the agent acts.

   The drift also decided the control branch. Its `P1` grasps at a fixed hand
   position, as `P0` does, and the drift moves the cube 20 mm in `y` relative to
   that position. In the target the release then came out centred (farthest corner
   27.7 mm from the plate centre); in the source the same program released the cube
   with a corner 42.3 mm out, failing the release check in all three replays. Its
   target success depends on a displacement that a resting cube should not have.
4. **Frozen intervention `P1` at the real target.** In all three target replays
   `P1` carried the cube for 290 control steps and lost it at step 399, on the
   descent towards the plate. In the agent's `0.5` diagnostic it reached its release at step 417.

The repair agent did not continue after the frozen evaluation because it had no
budget left and never receives evaluation results; the host then starts the
control branch, as the protocol requires. Defects 1 to 3 are addressed by
`dynamics-repair-v4`.

#### Version 4: a resting cube that stays put, stated criteria, true nominal values

`dynamics-repair-v4` has four changes; scoring is unchanged
(`dynamics-repair-score-v4`).

1. **No pre-contact drift.** The settling probe records the cube at every physics
   step of the native reset, tagged by reset phase, then holds both arms still for
   150 control steps. It calls no model:

   ```bash
   .venv/bin/python -m scripts.probe_dynamics_settling \
     --run-dir runs/202610301422_dynamics_settling_v4 --gpu 0 \
     --friction-scales 2.0 1.0 0.5 0.4 0.35 0.3 0.25 0.2 0.1 0.05 \
     --mass-scales 0.25 4.0 --restitutions 1.0
   ```

   Under the v3 adapter at `0.3` (`runs/202610301406_dynamics_settling_check`)
   the cube, from its first physics step on the table, alternates between two
   states every step: 10 µm apart in height, with its angular velocity about `x`
   flipping between about −0.3 and +1.9 rad/s. Each pair of steps moves it 0.04 mm
   in `+y`. Its contacts stay four corner points on one flat table collider. The
   reset leaves it 12.0 mm from its authored position and it moves 8.3 mm more
   during the hold, with no robot contact. The bounce is a solver artefact, not
   friction physics: applying external forces (gravity) in every TGS position
   iteration instead of once per step removes it. Three frictions screened two
   candidate fixes:

   | Candidate | Offset after reset at `0.35` / `0.3` / `0.05` | Hold drift at `0.05` | Solver angular speed at rest |
   |---|---|---|---|
   | none (v3) | – / 12.0 mm / – | – | 2.4 rad/s at `0.3` |
   | cube solver iterations 64/4 | 0.00 / 0.00 / 0.10 mm | 0.30 mm | 0.04–0.63 rad/s |
   | external forces every TGS iteration | 0.02 / 0.02 / 0.00 mm | 0.00 mm | 0.02 rad/s |

   `DynamicsTask` now sets `enable_external_forces_every_iteration` on the PhysX
   scene, IsaacLab's documented setting for noisy TGS velocities. The host snapshot's
   `physics_scene` records it and reset refuses a scene without it. The v4 probe
   covers the whole diagnostic range. At friction `2.0` to `0.05`, mass `0.25` and
   restitution `1.0` the cube rests within 0.02 mm of its authored position, at
   mass `4.0` within 0.09 mm. It does not move during the hold in any condition
   (largest single-step displacement below 1 µm). The target's first camera frames
   now differ from the source's only by render noise: 24 head-camera pixels change
   by more than 8 levels between `0.3` and `1.0`, against 99 between `2.0` and
   `1.0`. Under v3, the drifted target cube changed 9,801 right-wrist pixels.
   Diagnostic copies are built from the source settings plus the agent's
   overrides, so a diagnostic at the target's values has exactly the target's
   settings (`tests/test_dynamics_session.py`).
2. **Stated success criteria.** Both workspace instructions carry the scorer's
   criteria (see "Using the workspace").
3. **True nominal values** in `nominal.json`.
4. **Budgets** of 20 episodes, 8 of them target episodes, and 10,000 actions per
   branch. Executions stay at 100.

#### Development calibration at `0.3` under v4

The scale stays `0.3`. The v3 calibration above is rejected for v4 runs, so a
reference was developed again in `runs/202610301421_dynamics_v4_reference_dev`,
with no model calls. It refines the original policy of run `202610300959`: the
same skill file and the same approach, closing and release structure. Two changes
were made:

- the grasp is centred on the cube's rest position, commanded `y` `-0.1288`
  instead of `-0.115`, because the pads centre 1.2 mm below the commanded `y`;
- the hold is short, as in v3 reference `F`: a 20-step lift, one 35-step
  carry-and-tilt move over the plate, a 10-step lower and the release.

The first candidate, `R1`, succeeded in its development replays at `0.3` and in
the source, so no other candidate was written. Under v4 the unchanged `P0`
behaves in the source as under v3: carry complete at control step 256, release
at 376, success. At `0.3` its two-finger grasp, 13.8 mm off the cube's centre
along the pads, lasts from step 109 to 155. The cube rises at most 5 mm, slips
out and stays on the table.

One development replay per scale shows how the unchanged `P0` responds to
friction under v4. The cube's centre rests at `z = 0.7775` m:

| Friction scale | Two-finger grasp (control steps) | Highest cube centre | Outcome |
|---|---|---|---|
| `1.0` (source) | 109–375 | 0.932 m | success |
| `0.6` | 109–216 | 1.014 m | lost during the carry |
| `0.5` | 109–202 | 0.929 m | lost during the carry |
| `0.4` | 109–159 | 0.800 m | slipped out about 2 cm into the lift |
| `0.35` | 109–157 | 0.788 m | slipped out about 1 cm into the lift |
| `0.3` | 109–155 | 0.783 m | slipped out at the start of the lift |
| `0.25` | 109–157 | 0.783 m | slipped out at the start of the lift |

Lower friction never lets this policy carry the cube further or lift it higher.
Under v3 the drift reversed that order. In run `202610300959` the unchanged `P0`
held the cube until control step 157 in the agent's diagnostic at `0.25`, never
lifting it more than 5 mm. In the target at `0.3`, where the drift had moved the
cube towards the pad centre, it held it until step 209 and lifted it 15 cm.
With `P0` alone, diagnostics now bracket the target to at most about `0.35`.
Separating `0.3` from `0.25` needs a grasp that holds longer, such as `R1`'s.

Calibration `runs/202610301439_dynamics_calibration_v4_0.3` (no model calls):

```bash
.venv/bin/python -m orchestrator.dynamics_repair calibrate \
  --config tasks/DynamicsRepair/experiment.example.json \
  --run-dir runs/202610301439_dynamics_calibration_v4_0.3 --gpu 0 \
  --baseline-policy runs/202610300959_dynamics_run/policies/P0/policy.json \
  --reference-policy runs/202610301421_dynamics_v4_reference_dev/R1-target-0.3/policy/policy.json \
  --scales 0.3
```

| Replays | Successes | Incomplete | Outcome |
|---|---|---|---|
| baseline `P0`, source | `3/3` | `0` | released at step 376 |
| baseline `P0`, target `0.3` | `0/3` | `0` | slipped out during the lift (step 155) |
| reference `R1`, target `0.3` | `3/3` | `0` | carry at step 171, released at 211, rests 2 mm from the plate centre |

The status is `calibrated` with `selected_alpha` `0.3`. Every replay started
with the cube exactly at `(0.2, -0.13)`, used the live cube material `0.3/0.3`
(target) or `1.0/1.0` (source), and recorded
`enable_external_forces_every_iteration: true`. The three replays of each
condition are identical and repeat the development replays. Pass
`--calibration runs/202610301439_dynamics_calibration_v4_0.3` to record
`recoverability: "validated_by_reference"`. Whether a newly learned `P0` fails at
`0.3` remains that run's own result. `R1` uses the four primitives only, but its
grasp position comes from the scene layout, which agents never see; an agent has
to estimate the cube's position from camera images. `R1` therefore shows that a
refinement of the original policy can solve the target, not that an agent will
find it. It stays outside every agent workspace.

The custom task explicitly enables PhysX contact reporting after IsaacLab creates
its simulation context, whose default disables contact processing. This supplies
the private scorer with real contact events during normal initialization and
control steps. Missing reports and callback errors still fail the check; the
adapter does not weaken placement criteria or add observation-time physics steps.
The simulator container mounts the workspace and its sibling `raw_observations`
directory for exports. Raw archives remain outside the coding-agent mount.

The evaluator uses `dynamics-repair-score-v4`. Version 4 removes the linear and
angular speed checks (`linear_speed`, `angular_speed`) from the dwell condition
and from the pinned criteria. Success requires the same grasp, airborne carry,
release within `release_height` above the plate, withdrawn gripper, no robot
contact, the cube's full projection inside the plate, plate support and no table
contact, for `stable_steps` consecutive control cycles. A cube dropped onto the
plate from higher up still fails. In run `202610292357` the target replays left
the cube on the plate, unmoved over the last 100 control cycles (position change
below 1 µm, rotation below 0.0001°), while its measured peak angular speed stayed
at 0.105–0.116 rad/s, above the 0.1 rad/s limit. Measured speeds remain in the host
trajectory as diagnostics. Earlier runs keep their recorded scores.

Version 3 adds one
release condition to version 2: the host sample's `robot_contact`, true when any
reported cube contact pair involves a prim beneath any robot root
(`/World/envs/env_0/robot<N>`), must be false during the stability window.
Version 2 checked only finger contacts, finger distance, and gripper opening, so
a non-finger link still supporting the cube could pass. Thresholds, dwell
duration, and action limits are unchanged. Preserve earlier results and
re-evaluate frozen policies in a fresh run directory. Whether every robot link
collider appears in PhysX contact reports under its robot root requires GPU
validation.

The evaluator measures cube translation and shortest quaternion rotation after
every native physics step (`dt=0.004`, `decimation=1`) and retains the peak linear and angular speed over
each 25 Hz control interval. This prevents within-interval motion from cancelling
out. Only completed actions finalize a measurement; reading observations or host
snapshots does not create samples or settling time. Reset reseeds the tracker
after scene initialization, excluding teleports and old bodies.

Version 1 used raw solver velocities directly. GPU diagnosis found
nonzero solver velocity despite almost constant transforms even at individual
physics steps. The updated host records preserve those values separately as
`cube_solver_linear_velocity` and `cube_solver_angular_velocity`, alongside the
measured velocities and measurement version. Physics settings, task criteria,
thresholds, dwell duration, action limits, and policy code remain unchanged.
This is an explicit evaluator measurement revision: preserve earlier results
and validate frozen policies in a fresh run directory before new model runs.

## 1. Start source learning and paired repair

The coding agent creates the complete grasp, carry, and placement policy `P0`
during the source phase. Start the experiment directly:

```bash
.venv/bin/python -m orchestrator.dynamics_repair run \
  --config tasks/DynamicsRepair/experiment.example.json \
  --run-dir runs/202610281000_dynamics_run \
  --gpu 0
```

This command needs no supplied baseline policy, reference policy, or calibration
directory. The workspace's initial `submission/solution.py` is an exploratory
instruction-reading snippet; the source agent replaces it with its learned
complete policy before submission.

The run contains these stages:

1. `source`: start a coding agent in the fixed source environment to observe,
   write control code, execute it, and revise it until it submits a complete `P0`
   or exhausts its budget.
2. Freeze `P0` and replay it in three fresh source environments. Require `3/3`
   successes before transferring the policy. A failed source qualification ends
   the run with the original policy and evidence preserved.
3. Rebuild the same initial scene with the fixed target friction scale and replay
   the unchanged `P0` three times. Transfer degradation requires `0/3` target
   successes with all three executions complete, so that each is a genuine
   physical failure. A policy exception, simulator crash or timeout makes an
   execution incomplete and never counts as a physical failure. Any incomplete
   execution stops the run as `transfer_incomplete`; any target success (`1/3`,
   `2/3` or `3/3`) stops it as `transfer_failure_not_established`. Neither starts
   the repair branches. The runner does not scan or reduce friction until a
   failure appears.
4. `intervention` and `control`: start fresh repair agents with identical
   read-only source experience and raw feedback from all source/target
   qualification replays, including successful trials. The intervention branch
   may modify diagnostic parameters; the control branch uses nominal copies.
   Both branches use the same budgets and model settings.
5. Freeze each branch's final `P1` and replay it in three fresh target
   environments. Require `3/3` target successes to count repair as successful.
   Also replay three times in the source to check for regressions. Evaluation
   results are not returned to the repair agents.

Runs record `experiment: "dynamics-repair-v4"`. Experiment versions are
independent of the scoring schema versions below:

- `dynamics-repair-v2` differs from `dynamics-repair-v1` in two ways. Transfer
  qualification requires `0/3` rather than at most `1/3` target successes, and the
  cube material is applied before PhysX creates the cube's collision shapes (see the
  material section below). In `dynamics-repair-v1` runs the authored cube friction
  was not in effect in either condition, so their source and target physics were
  identical.
- `dynamics-repair-v3` changes the nominal cube friction from `0.8/0.6` to
  `1.0/1.0`. Against the scene-default fingers and table this gives the source
  contacts `0.5/0.5`, the friction actually in effect in `dynamics-repair-v1`
  runs. Under `0.8/0.6` (`0.4/0.3` effective) the `P0` of run `202610291735`
  failed its source replays.
- `dynamics-repair-v4` applies external forces in every TGS iteration, so the
  resting cube no longer drifts before contact. It also states the success
  criteria in every workspace, writes the true nominal values to `nominal.json`,
  and raises the default budgets to 20 episodes (8 target) and 10,000 actions.
  See "Version 4" above.

Preserve earlier runs unchanged; `run --calibration` rejects a calibration
recorded under another experiment version.

The configured target scale is not evidence that the target is solvable or that a
newly learned `P0` will fail there; the target sweep only shows that an earlier
probe policy failed. The host records
`target_configuration` with `source: "config"`, the effective `friction_scale`,
and `recoverability: "not_prevalidated"`. Report that limitation when repair
fails; failure alone cannot establish that the target was recoverable. Optional
development validation is available below.

Scene copies are rebuilt from identical construction settings and initial states;
this does not imply arbitrary-time physical snapshots or bitwise determinism.
The original layout and scoring-threshold JSON is stored with the configuration.
Preserve failed source learning, absent transfer degradation, and failed repair
as outcomes rather than repeating runs until the expected result appears.
Host and simulator work is serial; the first version does not require multiple
GPUs to run concurrently.

Simulator workers receive an explicit trial `--run-dir` and workspace path.
They communicate through a private Unix socket, supporting both a native Python
interpreter and the provisioned Docker/Apptainer wrapper without inherited file
descriptors. The simulator mounts the socket directory and workspace; the coding
agent never receives the worker socket. Container and native startup logs remain
under each `trials/<identifier>/` directory alongside that attempt's records.

### Optional development calibration

Researchers who already have development policies may separately validate a
recoverable perturbation with `calibrate`. It executes a supplied baseline to
measure degradation and a supplied reference to confirm that legal control can
still solve the target task. This optional command neither generates policies
nor calls a model. Keep development policies and calibration ground truth away
from the experimental source and repair agents.

The **parent directory of a development policy manifest is its workspace root**.
Both `entry` and `includes` are relative to that root:

```text
/path/development/
├── baseline/
│   ├── policy.json
│   ├── submission/solution.py
│   └── skills/helper.py
└── reference/
    ├── policy.json
    └── submission/solution.py
```

Example baseline manifest:

```json
{
  "entry": "submission/solution.py",
  "includes": ["skills/helper.py"]
}
```

Policies control the robot only through the four primitives. Complete replay
starts with an empty Python namespace and cannot depend on earlier snippets.
Frozen replay forbids reset. Files in `includes` execute in order before `entry`;
this is source execution, not Python import.

```bash
.venv/bin/python -m orchestrator.dynamics_repair calibrate \
  --config tasks/DynamicsRepair/experiment.example.json \
  --run-dir runs/202610280900_dynamics_calibration \
  --gpu 0 \
  --baseline-policy /path/development/baseline/policy.json \
  --reference-policy /path/development/reference/policy.json
```

Calibration checks friction scales `0.75, 0.5, 0.25, 0.125` in that fixed order
and retains every candidate's execution records. `--scales` replaces that list
with distinct descending values fixed before calibration starts; the list is
recorded as `scales` in `result.json`, and `run --calibration` accepts only a
selected scale from it. Each policy/environment
condition is replayed three times. The baseline must achieve `3/3` in the source.
Select the first candidate where the baseline fails all three target executions
(`0/3`) and the reference succeeds in all three (`3/3`), with every candidate
execution complete. These are the experiment's own transfer and repair thresholds.
If no candidate qualifies, calibration fails. Do not ignore failures or continue
tuning based on experimental agent performance.

To use a completed qualifying calibration, explicitly pass its directory:

```bash
.venv/bin/python -m orchestrator.dynamics_repair run \
  --config tasks/DynamicsRepair/experiment.example.json \
  --run-dir runs/202610281100_dynamics_validated_run \
  --gpu 0 \
  --calibration runs/202610280900_dynamics_calibration
```

The calibration's selected scale replaces `target_friction_scale` for this run.
The source environment, layout, action limit, and scoring criteria must match the
calibration. The coding agent still learns a new `P0` from scratch. The host
records `target_configuration.source: "calibration"` and
`recoverability: "validated_by_reference"`, together with the effective scale.
The optional reference establishes prior evidence of recoverability; it does
not guarantee that either repair branch will find a successful strategy.

Nominal cube static/dynamic friction is `1.0/1.0`; the target scale affects both.
The cube uses `multiply` combination and the fingers and table keep the scene
default `0.5`, so the effective cube contact friction is `0.5 x scale` in each
condition (`0.5/0.5` in the source).
Override the actual collision material, not merely `physics.friction`, which the
rigid-body constructor does not read.

The material must also be in effect in PhysX, not only in USD. On the native CPU
physics path the rigid objects are created after the simulation is playing.
Native `RigidObject.__init__` constructs `SingleRigidPrim`, which flushes pending
USD changes into PhysX and creates a tensor view, before `_setup_physics` binds
the object's material on the body prim. PhysX therefore creates the collision
shapes with the scene default material (`/physicsScene/defaultMaterial`,
`0.5/0.5`, `average`) and does not re-resolve the later binding. The USD binding
and attributes then describe a material PhysX never uses. A GPU construction-order
comparison in the same scene confirmed this: a cube built in native order used
`0.5/0.5` while its bound material said `0.11`. A cube whose material was bound
before construction used that material (`0.12`) and ignored the vendor's later
binding (`0.13`).

The adapter therefore authors the cube's material at `<cube prim>/dynamics_material`
(friction, restitution, `multiply` combination) and binds it to the cube prim
*before* native construction, every time a cube is created, including after
Playground reset. It refuses to proceed if a prim already exists at the new cube's
path. After construction it restores that binding in USD, because native
`_setup_physics` rebinds the vendor's own unused material. Only the cube is
changed: the table is static and keeps the scene default by native design. The
plate and other rigid objects keep the native behaviour, which, by the same
ordering, is also the scene default. Native tasks are unchanged.

After every reset, `prepare_scoring` compares PhysX's live state with the authored
values and fails the trial (unclassified) when they disagree. It reads every cube
shape's static friction, dynamic friction and restitution through the PhysX
tensor API, and requires every cube-side contact report material to be the
authored material. The host snapshot records `cube_physx_shape_materials`,
`cube_contact_materials`, `cube_collision_materials` (USD resolution) and
`contact_materials` (both sides). These records stay in the trial directory,
outside the agent workspace. The tensor API does not expose combination modes;
the slide response measured by the material probe is the evidence that the
combination rule is in effect. At source friction the short slide also makes the
cube yaw, and after some Playground resets it has reproduced a second, larger
deceleration although the live material was unchanged. Treat the target slide,
and the agreement of live shape and contact materials, as the material check.
Do not treat the source slide as a precise friction estimate. The friction change affects cube contact with the
gripper, table, and plate, so do not assume all failures are slips during
transport.

## Using the workspace

The workspace's `instruction.md` comes from the source or repair template.
`source_experience/` contains the same read-only source snapshot for both repair
branches. Each branch may edit `skills/`, `lessons/`, `submission/`, and `scratch/`.
The agent cannot access target truth, private scoring state, scene plans,
simulator source, or GPUs. Public nominal parameters and tool documentation
expose friction, mass, and restitution without revealing which property changed.
`nominal.json` holds the source cube's values: static and dynamic friction `1.0`,
restitution `0`, mass `0.35` kg, `multiply` combination against the `0.5`
fingers, table and plate, and the resulting effective contact friction `0.5`.
It never holds a target value. Before `dynamics-repair-v4` it reported `0.8/0.6`
(see the note on run `202610300959`).

`instruction.md` states the scorer's success criteria in its "Success criteria"
section, for the source agent and both repair branches. The text is generated
from the run's pinned criteria by `describe_criteria` in
`simulators/dynamics_scoring.py`, next to the scorer it describes. It states the
thresholds, including the release check: every corner of the cube within the
plate's inner radius when the carrying fingers lose contact. The scorer's state
during an episode stays private; agents still see only the success signal.
Submitted snippets use f-strings or `%` formatting. Access to `format` and
`format_map` methods is rejected because their replacement fields can traverse
private attributes without exposing those accesses to the Python AST checks.
Class-pattern keyword attributes follow the same restrictions. Positional class
patterns are unavailable because they resolve attribute names through runtime
`__match_args__`; public keyword, mapping, and sequence patterns remain supported.

```bash
python scripts/dynamics.py status
python scripts/dynamics.py start --kind target
python scripts/env.py exec submission/solution.py --include skills/helper.py
python scripts/dynamics.py end --conclusion "Observations from this target trial"
```

Source learning permits only `--kind source`; repair uses `target` or `diagnostic`.
`start` ends the current trial and creates a new simulator with an empty Python
session. `end` can record a conclusion. Within a trial, `env.py exec` preserves
Python variables and robot state but executes the entire submitted file each
time. Appending source code does not mean only the added lines will execute.

The following diagnostic example uses an agent-selected parameter and does not
prescribe a diagnosis:

```json
{"mass_scale": 1.1}
```

After writing that JSON to `scratch/parameters.json`:

```bash
python scripts/dynamics.py start --kind diagnostic \
  --parameters scratch/parameters.json \
  --hypothesis "The dynamics explanation to test" \
  --prediction "The observable behavior predicted before execution"
python scripts/env.py exec submission/solution.py
python scripts/dynamics.py end --conclusion "Observed results, competing explanations, and limitations"
```

Allowed keys are `friction_scale`, `mass_scale`, and `restitution`. Partial
overrides are supported; omitted fields remain nominal. Allowed ranges are
`[0.05, 2.0]`, `[0.25, 4.0]`, and `[0, 1]`, respectively, and values must be finite.
Target trials reject parameter overrides; the control branch rejects nonempty
overrides. Submit diagnostic hypotheses and predictions before starting, then
record conclusions after feedback. Retrospective text is not a substitute for
the experiment itself.

`reset()` and `env.py reset` restore the current trial's initial state while
preserving Python variables and trial parameters. They consume another global
episode; a target reset also consumes a target episode. Executions, diagnostics,
resets, and trial reconstruction share the outer budget. Cumulative actions and
execution requests are not replenished. Both `status` commands report session
and outer budgets without advancing physics.

### Complete policies and diagnosis records

The agent's final manifest is `workspace/submission/policy.json`, but its `entry`
and `includes` are relative to the **workspace root**, unlike the development
manifest location convention above:

```json
{"entry": "submission/solution.py", "includes": []}
```

The last successful incremental snippet is not sufficient. The published program
must run completely from the task's initial state and an empty namespace, without
reset, missing helper functions, or manually executed preparatory actions.
The manifest must be a JSON object with a nonempty string `entry`; `includes`,
when present, must be a list of nonempty strings. The host preserves the original
manifest bytes as `submitted-policy.json` alongside the archived policy. An
invalid repair submission is a `model_code_error` for that branch, and the
independent control branch still runs. Host archive failures remain distinct.

The repair agent writes `conclusion`, `evidence_trials`, `competing_hypothesis`,
and `limitations` to `lessons/diagnosis.json`. This is a self-report, not automatic
proof of a correct diagnosis. Review prior predictions, parameter interventions,
raw observations, and competing hypotheses, retaining ambiguity when explanations
cannot be distinguished. Report target repair separately from dynamics discovery.

### Human review of the dynamics diagnosis

The first version does not automatically judge a diagnosis or call another model
to summarize observations. The default assessment is `not_reviewed`, even if the
agent has submitted a diagnosis JSON and the policy has succeeded.

After examining host-side evidence, a researcher may create
`diagnosis-review.json` at the **root of the run directory**. This file is outside
the agent workspace. Do not turn the agent's self-report directly into a review
conclusion. Replace the example trial identifiers and notes with actual evidence:

```json
{
  "intervention": {
    "assessment": "ambiguous",
    "evidence_trials": ["000001"],
    "notes": "Explain the prior hypothesis, intervention results, competing explanations, and remaining limitations"
  },
  "control": {
    "assessment": "unsupported",
    "evidence_trials": ["000002"],
    "notes": "Explain why the available evidence does not support this branch's diagnosis"
  }
}
```

| `assessment` | Review convention |
|---|---|
| `supported` | The change category and direction were proposed in advance, intervention trends match raw observations, and there is evidence comparing a competing hypothesis |
| `ambiguous` | Evidence is insufficient, or multiple dynamics explanations remain indistinguishable |
| `unsupported` | Available evidence contradicts the proposed category, direction, or causal explanation |

`evidence_trials` refers to trial identifiers in each branch's `ledger.jsonl` and
`trials/<identifier>/`, not execution identifiers under `observations/`. Check the
hypothesis/prediction saved before the trial, parameter interventions, changes in
feedback under the same code policy, competing hypotheses, and the agent's
recorded limitations. Original code, videos, and images remain the evidence.
A correct guess, successful repair alone, or success after changing a diagnostic
copy is insufficient.

`summarize` reads this optional review file and reports it separately from agent
self-reports. There is no dedicated review-writing command; researchers maintain
this ordinary JSON directly. Omitted branches remain `not_reviewed` and are not
automatically promoted to correct diagnoses.

## 2. Summaries and interpretation

```bash
.venv/bin/python -m orchestrator.dynamics_repair summarize \
  --run-dir runs/202610281000_dynamics_run
```

Summaries read existing results without starting simulators, models, or new
attempts. Report raw source/target success counts for `P0` and each branch's `P1`,
branch budget consumption, agent-reported diagnoses, and failure status.
The host summary also reports `target_configuration`, distinguishing a scale
chosen in the run configuration from one selected by optional calibration and
retaining whether recoverability was validated beforehand. These host fields
are never included in the agents' matched source experience.
Stage results are saved before starting subsequent work. If evaluation is
interrupted, summaries retain completed qualification counts, started branches,
and any saved partial evaluation, including source, baseline, and reference
calibration batches. `attempts` remains the scheduled denominator of
three; `executed_attempts` counts started attempts and `pending` counts those not
started. `incomplete` includes interrupted or failed executions and pending
attempts. The `stage` field identifies the last entered stage.
Each batch's `failed_attempts` preserves recorded failure classifications and
error feedback. Root, branch, and phase errors and policy submission errors are
reported separately. Phase summaries also list unsuccessful `execution_completed`
events as `execution_failures` and worker failures as `worker_failures`; a worker
event is not counted as another execution. No causal label is inferred from an
ordinary unsuccessful task replay.

A failed trial is not a failed phase. When a trial's simulator fails, crashes, or
times out, the broker records a `worker_failure`, adds the trial to
`failed_trials`, sets the session's `valid` field to `false` permanently, and keeps
serving. Nothing is restarted automatically. The agent may start a new trial, which
consumes a new episode from the remaining budget. The broker's exit status reports
only the broker process, so a learning or repair phase with failed trials still
freezes and independently replays its published policy. The phase record keeps
`failed_trials` and `session_valid`. A broker that crashes or is killed, or an
agent that exits nonzero, still fails the phase. Frozen replays are different:
each replay has one trial, so a worker failure makes that attempt `incomplete`
with `success: false`. It cannot qualify source learning, transfer degradation,
or repair.

Each branch's `agent_report` retains its submitted evidence identifiers,
competing hypothesis, and limitations; invalid diagnosis JSON retains its
`report_error`. The `diagnosis` assessment and optional `host_review` remain
separate from that self-report. A confident agent claim never changes the review
assessment automatically.
Small samples test the loop, not statistical significance or generalization
across scenes. If the control also succeeds, compare costs and evidence honestly.

Preserve every failed attempt and its raw feedback. Python errors are classified
as `model_code_error` in feedback and host records. They stay in the failure
denominator but are incomplete replays, so they cannot establish physical
degradation during calibration or transfer qualification. Timeouts, crashes,
disconnections, and missing reports alone do not
establish an infrastructure fault. Requests are never retried automatically and
actions are not rolled back. Only confirmed infrastructure evidence justifies a
separately recorded rerun with its reason and original attempt preserved. New
runs cannot overwrite existing evidence.

## Validation and collaboration requirements

CPU validation:

```bash
python -m unittest discover -s tests -v
```

Check the four primitives and persistent sessions, parameter permissions,
cumulative budgets, complete policy replay, read-only source experience, and
failure preservation. Separate GPU validation must check task loading, contact
and stability criteria, effective friction overrides, and complete loop outcomes.
CPU mocks cannot replace these experiments.

New code must not introduce verification hashes, signatures, repeated manifest
verification, or excessive defensive checks. Use original text, ordinary
configuration fields, and sequential identifiers for evidence. Keep necessary
permission, finite-value, budget, and input checks in their owning layers.
Do not hide errors behind broad catches and silent fallbacks or automatically
retry model failures. Preserve the existing benchmark and third-party recipe
text, licenses, and existing verification metadata.
