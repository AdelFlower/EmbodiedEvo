# Friction transfer: does RL continuation repair a friction-induced failure?

Status: specification for Codex, written 2026-11-03 by the overseer (Claude) and approved by
the user. The overseer runs every GPU stage; the implementer writes code and CPU tests only.
This is a standalone experiment. It supersedes the fixability study
(`docs/DYNAMICS_FIXABILITY_SPEC.md`), which stays as it is.

## 1. Task and question

A right X5 arm picks up the red cube, lifts it until its bottom is 5 cm above the table top,
carries it over the plate and opens the gripper, so the cube drops into the plate.

1. Learn a policy by RL at cube friction scale 1.0.
2. Deploy it unchanged at friction 0.3 and 0.25.
3. In each environment where it fails, continue RL from it.
4. If that still fails, change the reward, then raise the gripper force.
5. Report the final outcome.

**Policy class (user decision, 2026-11-03).** The policy is a parametric pick–lift–carry–
release motion with 14 continuous parameters (§5).
- It is object-relative, and each phase's targets come from the state observed when that
  phase starts.
- A Gaussian search distribution over the parameters is trained by episodic policy search,
  using CMA-ES from `cma==4.0.0` (§8).
- "Continue training" means a new CMA-ES run centred on the earlier policy, in the failing
  environment.

Step-level deep RL from scratch is not feasible here. The simulator manages about 9.5
actions/s, with at most two simulators at once.

**Reuse boundary (user decision).** The experiment may reuse the RoboDojo scene and the env
adapter. This covers:
- `DynamicsRLTask`, the cube friction plumbing and the ±3 cm layout reset;
- contact reports and the native `PlacementScorer`;
- the simulator socket worker and its transport.

It must not use:
- the agent-derived R1 teacher (`integrations/dynamics_rl/teacher.py`) or any of its
  waypoints and numbers;
- P0, demonstrations or BC;
- the orchestrator, adaptation, repair or fixability pipelines.

## 2. Engineering rules (normative)

- No hashes, checksums, verification files, or re-reading of files you just wrote.
- Validate only at trust boundaries: the config file and CLI arguments.
- Let errors propagate. Do not catch-and-continue, and do not add broad `try/except`
  ladders. The simulator transport already records its own failures.
- No retries, automatic reruns, fallbacks, mode switches, compatibility shims or speculative
  options. Implement exactly the options listed here.
- One-line docstrings. No logging framework: write records as JSON lines.
- Reuse existing functions (§3). Tests mock only the simulator client.
- Every command takes an explicit `--run-dir` with a `YYYYMMDDHHMM` prefix and refuses a
  directory that already exists.
- Do not modify `vendor/`, `simulators/`, the native scorer, the existing worker,
  `features.py`, or any adaptation, repair or fixability file.
- The one permitted edit to existing code is §4.2.

## 3. Files

| File | Change | Size guide |
|---|---|---|
| `integrations/friction_rl/__init__.py` | New, empty | — |
| `integrations/friction_rl/worker.py` | New: in-container task subclass with gripper-force scale and grip-force readout, plus `main()` (§4) | ≤ 120 lines |
| `integrations/friction_rl/primitive.py` | New: parameter space, orientation helpers, phase-machine policy (§5) | ≤ 220 lines |
| `integrations/friction_rl/episode.py` | New: one-episode runner with early stops and trace; outcome features, success, rewards (§6, §7) | ≤ 230 lines |
| `orchestrator/friction_rl.py` | New: CLI `train` and `evaluate`, simulator pool, CMA-ES loop, dev checks, records (§8, §9) | ≤ 360 lines |
| `integrations/dynamics_rl/client.py` | Edit: `worker_module` keyword (§4.2) | +3 lines |
| `tasks/FrictionTransfer/config.json` | New: §10, verbatim | — |
| `tests/test_friction_rl.py` | New: §11 | — |

Allowed imports from the existing code:
- `integrations.dynamics_rl.client.SimulatorClient`;
- from `integrations.dynamics_rl.features`: `encode_action`, `sample_scene`,
  `unit_quaternion`, `quaternion_product`, `REFERENCE_QUATERNION`, `ACTION_LOW`,
  `ACTION_HIGH`;
- from `simulators.dynamics_scoring`: `cube_vertices`, `Criteria`;
- `simulators.dynamics_rl.DynamicsRLTask` and `simulators.dynamics_native.beneath`
  (worker only);
- `orchestrator.dynamics_rl_worker.run_worker` (worker only);
- `orchestrator.files.write_json`.

Nothing else from `integrations/dynamics_rl/` or `orchestrator/dynamics_*`.

`requirements-rl.txt` already lists `cma==4.0.0`, and the overseer has installed it in
`.venv-rl`.

## 4. Simulator worker

### 4.1 `integrations/friction_rl/worker.py` (runs inside the simulator image)

The class is `GripperForceTask(DynamicsRLTask)`, constructed as
`GripperForceTask(mode=..., settings=..., gripper_scale=...)`. Here `settings` holds the
three physics keys and `gripper_scale` is a positive float.

- **After every `reset_episode`.** Apply the gripper scale to the gripper joints
  (`robot.gripper_joint_indices`) of every arm in `self.native.robot_manager`.
  - On the first reset, read the nominal values from the articulation's
    `root_physx_view.get_dof_stiffnesses()` and `get_dof_max_forces()`, and store them.
  - Write `nominal × gripper_scale` with `write_joint_stiffness_to_sim` and
    `write_joint_effort_limit_to_sim`, passing `joint_ids`.
  - Apply the scale after every reset, including at scale 1.0.
- **`rl_state()`.** Return `super().rl_state()` with two extra keys:
  - `"gripper_drive"`: `{"stiffness": [...], "max_force": [...]}` for the right gripper's
    two joints, read back from the PhysX view.
  - `"right_grip_force"`: one approximate normal force in newtons per right finger,
    `Σ |impulse · normal| / physics_dt`. The sum runs over the points of every current
    contact pair whose other actor or collider lies beneath that finger's rigid-body path.
    Use `self.native.finger_paths` and `self.native.contact_monitor.contacts()`, test paths
    with `beneath`, and take `physics_dt = self.native.sim_cfg.sim.dt`.
- **`main(argv)`.** Mirror `orchestrator.dynamics_rl_worker.main` with these changes:
  - `--parameters` must hold exactly the keys `friction_scale`, `mass_scale`, `restitution`
    and `gripper_scale`, all numeric, with `gripper_scale > 0`.
  - Call `run_worker(stream, run_dir, parameters=<the three physics keys>, mode=...,
    builder=...)`. The builder constructs `GripperForceTask` with the scale.

  The `ready` message then echoes the three physics keys. The host records all four in its
  own config.

### 4.2 Edit to `integrations/dynamics_rl/client.py`

Add a `worker_module="orchestrator.dynamics_rl_worker"` keyword to
`SimulatorClient.__init__`, store it, and use it in the launch command in place of the
literal string. Nothing else changes.

## 5. Policy (`primitive.py`)

### 5.1 Geometry and constants

Quaternions are scalar-first (w, x, y, z), as in the worker.
- `Q_DOWN = REFERENCE_QUATERNION` is a top-down hand. Its local x axis, the approach axis,
  points along world −z, and its fingers close along world x.
- `GRASP_OFFSET = 0.145` m. This is `gripper_bias` in
  `vendor/robodojo/Assets/Robots/x5/robot_config.yml`, the distance from the end-effector
  frame to the pad centre along the approach axis.
- Grasp point: `grasp_point(pose) = pose[:3] + R(pose[3:]) @ [GRASP_OFFSET, 0, 0]`. Its
  inverse, `ee_position(point, q) = point − R(q) @ [GRASP_OFFSET, 0, 0]`.
- Hand orientation: `hand_q(yaw, tilt) = q_y(tilt) ⊗ q_z(yaw) ⊗ Q_DOWN`. `q_axis(angle)`
  rotates about that world axis. A positive tilt moves the approach axis from −z toward −x,
  which is toward the plate and away from the right arm's base.
- `TABLE_Z = Criteria().table_height` (0.765) and `CUBE_HALF = Criteria().cube_size / 2`.

### 5.2 Parameters

Each parameter is searched in normalised form `u ∈ [−1, 1]` and mapped linearly to its
physical range. Durations are rounded to whole control steps.

| # | Name | Range | Unit | Meaning |
|---|---|---|---|---|
| 0 | `grasp_dx` | −0.008 … 0.008 | m | grasp point − cube centre, world x |
| 1 | `grasp_dy` | −0.008 … 0.008 | m | world y |
| 2 | `grasp_dz` | −0.015 … 0.010 | m | world z |
| 3 | `yaw` | −15 … 15 | deg | hand yaw about world z |
| 4 | `grip_cmd` | 0.0 … 0.4 | — | gripper command while holding (0 = tightest) |
| 5 | `t_descend` | 10 … 40 | steps | |
| 6 | `t_close` | 6 … 30 | steps | |
| 7 | `lift_clear` | 0.05 … 0.08 | m | commanded clearance of the cube bottom above the table |
| 8 | `t_lift` | 8 … 40 | steps | |
| 9 | `tilt` | 0 … 40 | deg | hand tilt during carry and release |
| 10 | `t_carry` | 12 … 60 | steps | |
| 11 | `release_dx` | −0.03 … 0.03 | m | release grasp point − plate centre, world x |
| 12 | `release_dy` | −0.03 … 0.03 | m | world y |
| 13 | `t_settle` | 0 … 20 | steps | hold before opening |

Required functions:
- `physical(u)`, which returns a dict of the 14 values;
- `normalized(dict)`, its inverse;
- `PARAMETER_NAMES`.

### 5.3 Phase machine

The policy is `PrimitivePolicy(params_dict)`, and `act(state) -> (action7, phase_name)`.
- Each step's command is an absolute end-effector pose plus a gripper value, encoded with
  `encode_action`.
- Each phase interpolates from the target commanded at the end of the previous phase to its
  own target. The first phase starts from the observed `right_ee_pose` at reset.
- The grasp point moves on a minimum-jerk profile, `s(τ) = 10τ³ − 15τ⁴ + 6τ⁵` with
  `τ = k / T` for k = 1…T. Orientation follows a slerp with the same `s`.
- `cube` means the cube position observed at the first step of the phase. `plate` means
  `state["plate_position"]`.

| Phase | Steps | Grasp-point target | Orientation | Gripper |
|---|---|---|---|---|
| `approach` | 40 | `cube + (dx, dy, dz + 0.06)` | `hand_q(yaw, 0)` | 1.0 |
| `descend` | `t_descend` | `cube + (dx, dy, dz)` | same | 1.0 |
| `close` | `t_close` | hold | same | `grip_cmd` |
| `lift` | `t_lift` | same xy, z = `TABLE_Z + CUBE_HALF + lift_clear + dz` | same | `grip_cmd` |
| `carry` | `t_carry` | `plate_xy + (release_dx, release_dy)`, z unchanged | slerp to `hand_q(yaw, tilt)` | `grip_cmd` |
| `settle` | `t_settle` | hold | hold | `grip_cmd` |
| `release` | 10 | hold | hold | 1.0 |
| `withdraw` | 15 | end-effector rises 0.08 m along world z | hold | 1.0 |
| `hold` | 30 | hold | hold | 1.0 |

After `hold` the policy is finished, and `act` raises if it is called again. A phase of
length 0 (`t_settle = 0`) is skipped.

## 6. Episode runner (`episode.py`)

`run_episode(client, params, scene, episode_dir) -> (record, trace)`:

1. Call `client.reset(scene, episode_dir)`.
2. Loop: `act` on the latest `state`, then `client.step`.
3. Stop at the first of these conditions, and record it as `stop_reason`:
   - `success`: the host success of §7 is reached (the rest counter reaches 13), or native
     `terminated` with `info.success`;
   - `native_end`: native `terminated` or `truncated` otherwise;
   - `no_lift`: at the last step of `lift`, both right fingers are not touching the cube, or
     the cube's lowest-corner clearance is below 0.02 m;
   - `dropped`: during `carry` or `settle`, the cube touches the table, or neither right
     finger has touched it for 3 consecutive steps;
   - `not_placed`: the `hold` phase ends.

   The host then simply resets for the next episode; the worker records `host_reset`.
4. Write `trace.npz` and `episode.json` into `episode_dir`.

The trace has one row per step for these fields:

| Field | Width |
|---|---|
| phase index | 1 |
| commanded end-effector pose | 7 |
| observed end-effector pose | 7 |
| gripper command | 1 |
| `gripper_opening.right` | 1 |
| right finger contacts | 2 |
| table, plate and robot contacts | 3 |
| cube position | 3 |
| cube quaternion | 4 |
| `right_grip_force` | 2 |
| scorer `transport_completed`, `released_over_plate`, `stable_steps` | 3 |

The trace also holds `gripper_drive` from the reset state.

## 7. Outcome, success and rewards (`episode.py`, pure functions of the trace)

Definitions:
- **Held airborne:** both right fingers touch the cube, and the cube touches neither the
  table nor the plate.
- **Clearance:** the minimum z of `cube_vertices(...)` minus `TABLE_Z`.
- **Release step:** the first step of the `release` phase. Trace row `t` holds the state
  after action `t`, so the state at the start of the release phase is row
  `release_step − 1`.
- **Grasp-point slip:** the cube offset in the hand frame, `R(q_obs)ᵀ (cube −
  grasp_point(observed pose))`, measured relative to its value at the last `lift` row. A
  hand-frame offset makes a rigid hold read zero slip even while the hand tilts.

Features:
- `grasped`: at some step, both right fingers touch the cube.
- `max_clear`: the maximum clearance over held-airborne steps before the release step
  (0 if there are none).
- `lifted5`: `max_clear ≥ 0.05`.
- `held_to_release`: the release step is reached, and both right fingers touch the cube in
  row `release_step − 1`.
- `release_dist`: the horizontal distance from the cube to the plate centre.
  - It is measured at the first step at or after the release step on which neither right
    finger touches the cube.
  - For a `dropped` episode it is measured at the drop.
  - Otherwise it is measured at the last step.
- `placed`: the scorer's `released_over_plate` was ever true.
- `slip`: the maximum grasp-point slip over `carry` and `settle` (m).
- `grip_force_min_carry`: the minimum over `carry` of the mean of the two finger forces.
- `track_err_carry`: the maximum `|commanded − observed|` end-effector position over
  `carry` (m).
- `steps`: the number of actions taken.

**Revision 2 (user decision, 2026-11-03): the cube must rest in the plate.** In the first S
attempt (`runs/202611031213_ft_S`), about a third of releases ended with the cube flat
inside the plate, a bowl about 13 cm across, after tumbling once on impact. Its centre came
to rest 2–3.5 cm from the plate centre. The native rest rule requires every corner within
4 cm of the centre, so it rejected these placements. No policy parameter controlled the
tumble.

The user's task says the block "falls into the plate". The native scorer still runs and is
recorded, but its 4 cm all-corners **rest** rule no longer decides success.

- **Rest row:** a row at or after `release_step` in which all of these hold:
  - the cube touches the plate, and touches neither the table nor any robot link;
  - neither right finger touches the cube;
  - `gripper_opening.right ≥ 0.8`;
  - the cube's lowest corner lies in the native supported-height band,
    `plate_bottom − 0.002 ≤ z_min ≤ plate_top + 0.003`, using `Criteria()`;
  - the cube centre lies horizontally within `PLATE_REST_RADIUS = 0.045` m of the plate
    centre.
- `rest_steps`: the longest run of consecutive rest rows.
- `in_plate`: `rest_steps ≥ 13`, where 13 is `Criteria().stable_steps`.
- `native_success`: the native scorer reported success, which is now recorded only.

**`success`** requires all four of:
1. `lifted5`;
2. `held_to_release`, so the fingers did not lose the cube before the gripper opened;
3. `placed`: the native release check passed when the fingers let go. The cube was over the
   plate, and this part of the native scorer is kept.
4. `in_plate`.

Native stable rows are a subset of rest rows, so native success always implies `in_plate`.
The runner keeps the rest counter as it goes, using the same per-row predicate as `outcome`,
and stops with `success` as soon as all four hold.

Rewards are always computed and recorded; the stage's `--reward` chooses which one drives
the search:

- `reward_v1 = grasped + 2·min(max_clear / 0.05, 1) + 2·held_to_release·(1 − tanh(release_dist / 0.05)) + placed + success·(6 − 0.005·steps)`
- `reward_v2 = grasped + 2·min(max_clear / 0.05, 1) + 2·held_to_release·(1 − tanh(release_dist / 0.05)) + placed + 6·success − 2·min(slip / 0.02, 1)`

`reward_v2` is for the reward-change stage. It drops the time term and penalises slip in
the grasp.

## 8. Search (`orchestrator/friction_rl.py train`)

Arguments:
- `--run-dir`, `--config`, `--friction-scale`, `--gpu` (default 0);
- `--gripper-scale` (default 1.0);
- `--reward {v1,v2}` (required);
- `--init-policy PATH`: optional. Absent means start at `u = 0` with `sigma0_source` and
  `max_iterations_source`. Present means start at the policy's `normalized` with
  `sigma0_continue` and `max_iterations_continue`.

Procedure:
1. Start `num_envs` simulator clients concurrently.
   - Each client uses `worker_module="integrations.friction_rl.worker"`, mode `state`, and
     parameters `{friction_scale, mass_scale: 1.0, restitution: 0.0, gripper_scale}`.
   - Each client's run dir is `<run>/<prefix>_sim_w<k>`, where `<prefix>` is the run's
     12-digit stamp.
   - Close every client on exit.
2. Create `cma.CMAEvolutionStrategy(x0, sigma0, {"bounds": [-1, 1], "popsize": popsize,
   "seed": seed + 1, "verbose": -9})`.
3. For each iteration `i`:
   - Call `ask()`.
   - Run candidate `j` on the training layout `sample_scene(100000 + 1000·i + j,
     layout_radius)`, with episode folder `it{i:03d}_c{j:02d}`.
   - Spread episodes over the clients through a shared queue, one thread per client, and
     return records in candidate order.
   - Call `tell(X, [−reward])`.
   - Append one line to `iterations.jsonl` with: `i`, `xfavorite`, `sigma`, the
     per-coordinate std, rewards, successes, stop reasons and wall time.
   - Append every episode record to `episodes.jsonl`.
   - Pickle the strategy to `es.pkl`, overwriting it.
4. **Dev check.** After iterations where `(i + 1) % dev_every == 0`, and after the last
   iteration:
   - Run the deterministic policy `xfavorite` on the `dev_episodes` dev layouts.
   - Write `policy_it{i:03d}.json` and append to `dev.jsonl`.
   - Stop when the last two dev checks both reach `dev_pass` or more.
5. **Selection.** `policy.json` is the dev-checked policy with the highest dev success,
   breaking ties by the latest iteration.

A policy file holds:
- `version: "ft-primitive-v1"`;
- `normalized` (14 values) and `physical` (dict);
- the iteration;
- the dev result;
- the stage's friction, gripper scale and reward;
- `init_policy`.

`--resume-es PATH --start-iteration N` continues a pickled strategy in a new run dir after an
infrastructure crash. It replaces `--init-policy`. These are the only resume options.

## 9. Evaluation (`orchestrator/friction_rl.py evaluate`)

Arguments:
- `--run-dir`, `--config`, `--friction-scale`, `--gpu`;
- `--gripper-scale` (default 1.0);
- `--policy PATH`: optional. Absent means `u = 0`.
- `--override JSON`: optional dict of physical values that replaces entries after loading.
  It is for overseer probes.
- `--layouts {dev,eval}`;
- `--episodes N`: optional. It takes the first N layouts of the set, defaulting to the whole
  set.
- `--render`: use worker mode `render`, which records raw videos.

Write:
- `config.json`, with the resolved arguments and parameters;
- `episodes.jsonl`;
- `summary.json`, holding:
  - successes, N, the rate and `pass = rate ≥ 0.9`;
  - per-layout success;
  - counts of each stop reason;
  - the means of `max_clear`, `slip`, `grip_force_min_carry` and `track_err_carry`;
  - `release_dist` for failures.

Episode folders are named `eval_l{k:02d}` or `dev_l{k:02d}`.

**Layout sets.** Dev layout `k` is `sample_scene(2000 + k, layout_radius)` for
k = 0…`dev_episodes` − 1. Eval layout `k` is `sample_scene(3000 + k, layout_radius)` for
k = 0…`eval_episodes` − 1. Every stage and friction uses these same layouts, so results can
be compared scene by scene.

## 10. Configuration (`tasks/FrictionTransfer/config.json`)

```json
{
  "simulator_python": ".venv-robodojo/bin/python",
  "startup_timeout": 1200,
  "step_timeout": 120,
  "seed": 0,
  "num_envs": 2,
  "layout_radius": 0.02,
  "popsize": 16,
  "sigma0_source": 0.3,
  "sigma0_continue": 0.2,
  "max_iterations_source": 50,
  "max_iterations_continue": 30,
  "dev_every": 3,
  "dev_episodes": 10,
  "dev_pass": 9,
  "eval_episodes": 30
}
```

`load_config` checks the key set, and that every count and sigma is positive.

## 11. Tests (CPU, `tests/test_friction_rl.py`)

1. `physical` and `normalized` round-trip, and `u = ±1` maps to the table bounds.
2. Geometry:
   - at `hand_q(0, 0)`, the grasp point is the end-effector position minus (0, 0, 0.145);
   - a positive tilt gives the approach axis a negative x component;
   - `ee_position` inverts `grasp_point`.
3. Action bounds. Use the 2¹⁴ corners subsampled to 256, plus 256 uniform draws, on
   default-scene states with cube and plate shifted by ±0.02. Run `PrimitivePolicy` through
   every phase on a fake state whose end-effector follows the command exactly. No
   `encode_action` call may raise.
4. Phase lengths add up to the durations, `t_settle = 0` skips `settle`, and `act` after
   `hold` raises.
5. Outcome on synthetic traces:
   - a clean success;
   - native success with `max_clear = 0.04` (not a success);
   - the cube lost before the release step and then landing in the plate (not a success);
   - exact `reward_v1` and `reward_v2` values for one hand-computed trace.
6. Early stops on a fake client: `no_lift`, `dropped`, `not_placed` and native `success`
   each end the episode at the right step.
7. `train` with a fake client pool, using reward −|u − u*|²:
   - the mean approaches u*;
   - the files listed in §8 exist;
   - dev stopping and selection follow §8;
   - `--resume-es` continues from the pickled iteration.
8. The worker `main` rejects a missing `gripper_scale` or a non-positive one. Patch the
   socket and task, so no Isaac is needed.

Run `python -m unittest tests.test_friction_rl -v`. Then run the full suite. Socket tests
that fail only because the sandbox blocks Unix sockets are run by the overseer.

## 12. Overseer procedure (GPU; not for the implementer)

Runs are named `runs/<YYYYMMDDHHMM>_ft_<id>`. GPU jobs run one at a time.

| ID | Command | Purpose and gate |
|---|---|---|
| P1 | `evaluate`, `u = 0`, f 1.0, dev, 2 episodes, `--render` | Check geometry, IK and grasp. The grasp constant and §5.2 bounds may change only here. They are frozen before S. |
| S | `train --reward v1`, f 1.0 | Source policy |
| E0 | `evaluate` S, f 1.0, eval 30 | Source gate: ≥ 27/30 |
| D1, D2 | `evaluate` S, f 0.3 and f 0.25, eval 30 | An environment **fails** if it scores < 27/30 |
| C1(f) | `train --reward v1 --init-policy S`, at each failing f | Continued training: same reward and search, new physics only |
| E1(f) | `evaluate` C1, f, eval 30 | ≥ 27/30 means **repaired by continued RL**; stop and report |
| C2(f) | `train --reward v2 --init-policy C1`, f | Reward change, only if E1 fails |
| E2(f) | `evaluate` C2, f, eval 30 | ≥ 27/30 means repaired with the reward change |
| C3(f) | `train --reward v2 --init-policy C2 --gripper-scale 2.0`, f | Gripper force ×2, only if E2 fails |
| E3(f) | `evaluate` C3, f, `--gripper-scale 2.0`, eval 30 | Final verdict |

Further rules:
- Videos come from `evaluate --render --episodes 2–3` for the source at each friction, plus
  every final policy.
- If neither 0.3 nor 0.25 fails at D, stop and report. Do not search for a friction that
  does fail.
- The report must disclose:
  - the parametric policy class;
  - the time term in `reward_v1`;
  - that the 5 cm lift is checked on the host on top of the native scorer;
  - that C3 changes the robot's hardware parameters.

## 13. Scripted feasibility and direct gripper-force control (user request, 2026-11-03)

The user clarified that they intend an RL **network** to control the arm. Before building it,
they asked the overseer for a direct feasibility check, on one scenario distribution (the
§9 layout sets):

1. Can a scripted policy written for friction 0.25 complete the task?
2. If not, does adding direct control of the gripper force make the task completable?

The parametric CMA-ES stages of §12 are superseded. Stages S and S2 were stopped; their notes
are in their run directories.

### 13.1 Finding that motivates force control

Probe runs `runs/*_ft_F025_*` gave these results:
- The position-controlled gripper squeezes with at most about 25 N per finger, whatever the
  command. Any command at or below 0.2 targets the joint's lower limit, 0. Contact therefore
  ends near `2300 N/m × 0.012 m`.
- At friction 0.25 that grip lets the cube creep down between the fingers, at about
  5 mm/s during the lift and faster during the carry.
- The best script found, `cradle3`, scored 6/10 on dev. It uses a low grasp, a fast lift and
  carry, and a 40° carry tilt.

### 13.2 Implementation (milestone FT2)

1. **`orchestrator/dynamics_rl_worker.py`.** Add a keyword
   `worker_class=RLWorker` to `run_worker`, and construct `worker_class(env, run_dir)`.
   Nothing else changes.
2. **`integrations/friction_rl/worker.py`.**
   - `GripperForceTask.set_grip_force(force)` takes the force per right finger, in newtons.
     - **Position mode,** when `force == 0`. The right gripper joints get stiffness
       `nominal × gripper_scale`, and their effort target is 0.
     - **Force mode,** when `force > 0`. The right gripper joints get stiffness 0, with
       damping unchanged, and the articulation's `set_joint_effort_target` is `−force` on
       both joints.
     - Both joints open as q increases (`gripper_scale` is [−0.01, 0.044] and the URDF
       limits are [0, 0.044]), so a closing force is negative.
     - Write only when the requested value changes.
   - `reset_episode` ends with `set_grip_force(0.0)`.
   - `rl_state()` adds `"grip_force_command"`.
   - `ForceRLWorker(RLWorker)`. Its `step(command)` reads `grip_force = float(command.get(
     "grip_force", 0.0))` and requires `0 ≤ grip_force ≤ 100`; 100 N is the drive's effort
     limit. It then calls `env.set_grip_force`, then `super().step(command)`.
   - `main` passes `worker_class=ForceRLWorker`.
3. **`integrations/friction_rl/episode.py`.**
   - `run_episode(..., grip_force=0.0)` sends every step as `client.request({"op": "step",
     "action": [...], "grip_force": f})`.
   - `f` is `grip_force` during `close`, `lift`, `carry` and `settle`, and 0.0 otherwise, so
     the release opens under position control.
   - The trace gains a `grip_force_command` column.
4. **`orchestrator/friction_rl.py`.** `evaluate` gains `--grip-force N`, with
   0 ≤ N ≤ 100 and default 0. It is passed to `run_episode` and recorded in `config.json`.
   `train` is unchanged and uses 0.
5. **Tests.** Cover these with fake articulations and fake clients:
   - mode switching writes stiffness 0 and effort −F, and switching back restores
     `nominal × scale` and effort 0;
   - writes happen only on change;
   - the worker validates the force and applies it before the step;
   - `run_episode` sends the force only in the four holding phases;
   - the CLI validates `--grip-force`.

### 13.3 Overseer procedure

- **FS.** Run the best position-controlled script on eval 30 at friction 0.25. Feasible means
  27/30 or more.
- **FF.** If FS fails, run the same scripts with `--grip-force` F. Use the dev layouts to
  find the smallest F among {35, 50, 75, 100} N that reaches 9/10 or more, then evaluate
  that F on eval 30. The contact-force readout must confirm that the commanded force is
  achieved.
- **Report.** For each script and force, give the success rates and failure modes, with the
  measured grip and slip.

## 14. Revision 3: the cube need only end in the plate, and motion must be smooth (user decision, 2026-11-03)

The `cradle3` evaluation showed something the earlier rule hid. Every one of its 12 strict failures with the
position-controlled gripper still ended with the cube in the plate. The grip gave way 0–2 steps before the
scheduled opening, while the hand braked over the plate.

The user made two decisions:
1. "The cube ends in the plate" is enough to count as success.
2. Scripts must move smoothly. `cradle3`-style fast motion that flings the cube into the plate is not
   acceptable.

### 14.1 Success, revision 3

`success = lifted5 and in_plate`. `held_to_release` and `placed` are still computed and recorded, but they
no longer gate success.

Record `success_strict`, the revision 2 definition, alongside `success` for comparison.

`rest_row`, `rest_steps` and `in_plate` keep their revision 2 definitions. Rest rows count only from the
release phase on, when the gripper is open.

### 14.2 Early stops

- `dropped` now means only that the cube touched the table during `carry` or `settle`.
- The "no finger contact for 3 steps" stop is removed. A cube that leaves the fingers over the plate may
  still land in it. The script then continues through release, withdraw and hold.
- The runner stops with `success` once `lifted5` holds and the rest counter reaches 13.
- `no_lift`, `not_placed` and the native-end stops are unchanged.

### 14.3 Smoothness measurements (recorded on every episode)

These are computed from the trace, over the rows of `lift`, `carry` and `settle`. Take finite differences
at 25 Hz (dt = 0.04 s).
- `cmd_peak_speed` and `cmd_peak_accel`: the peak speed (m/s) and acceleration (m/s²) of the
  **commanded** grasp point.
- `obs_peak_speed`: the peak speed of the **observed** grasp point.
- `settle_before_release`: the number of `settle` steps before the first `release` step.

`evaluate`'s `summary.json` reports the maximum of each over the run.

The overseer's smoothness limits are checked on these values; they are not enforced in code:
- commanded speed ≤ 0.25 m/s;
- commanded acceleration ≤ 0.5 m/s²;
- `settle_before_release ≥ 10`.

### 14.4 Overseer procedure

Run at friction 0.25 on dev 10 with the stock position-controlled gripper.
- The base smooth script is:

  ```json
  {"grasp_dx": 0, "grasp_dy": 0, "grasp_dz": -0.010, "yaw": 0, "grip_cmd": 0, "t_descend": 30,
   "t_close": 15, "lift_clear": 0.07, "t_lift": 40, "tilt": 0, "t_carry": 60,
   "release_dx": 0, "release_dy": 0, "t_settle": 15}
  ```
- Also run a variant with `tilt` 40.
- As a control, run the better of the two at friction 1.0.
- Run the better script at 0.25 with `--grip-force` 25 as a control at the stock force level. It separates the
  effect of force mode from the effect of more force. Then run 35, and 50 if needed.
- Confirm on eval 30 at 0.25: the smooth script with the stock gripper, and with the smallest sufficient
  force.

## 15. Revision 4: lift, carry level, lower, release; constant grip force 15 N vs 25 N (user request, 2026-11-03)

The user specified the experiment directly:

1. The gripper applies a **continuous force**: the FT2 force mode (§13.2) holds a constant squeeze from
   `close` until the gripper opens.
2. Compare 15 N with 25 N at friction scale 0.25.
3. The sequence is fixed:
   - lift the cube so its bottom is 5 cm above the table;
   - move it horizontally until it is above the plate centre;
   - lower it by 4 cm;
   - release it.
   All motion must be stable and slow.
4. The cube may start slightly further from the plate.
5. Success is simple: the cube ends inside the plate.
6. Use one scenario; the user wants results quickly.

### 15.1 Policy (`primitive.py`)

- Insert the phase `lower` between `settle` and `release`:
  `PHASES = ("approach", "descend", "close", "lift", "carry", "settle", "lower", "release", "withdraw", "hold")`.
- `lower` is fixed, not a searched parameter:
  - module constants `LOWER_DZ = 0.04` and `LOWER_STEPS = 30`;
  - on entry, the target pad point is the entry point with `z -= LOWER_DZ`;
  - orientation is unchanged;
  - `grip` is `grip_cmd`, as in carry and settle;
  - `lengths` gains `LOWER_STEPS` at the new position.
- Raise `HIGH` for `t_carry` from 40 to 100 so a slow carry can be requested. No other parameter or
  bound changes, and the parameter count stays 14.

### 15.2 Scene: cube shift (`orchestrator/friction_rl.py`, configs)

- Add a required config key `cube_shift`: `[dx, dy]` in metres, each finite and within [−0.05, 0.05].
- One helper builds every scene: `sample_scene(seed, cfg["layout_radius"])`, then add `cube_shift`
  to `scene["Rigid"]["cube"][0]["default_pos"][:2]`. Both `layout_jobs` and `train` use it.
- `config.json` and `single_layout.json` get `[0.0, 0.0]`, so their scenes are unchanged.
- New file `tasks/FrictionTransfer/lower_release.json`: a copy of `single_layout.json` with
  `"cube_shift": [0.04, 0.0]`. The cube moves from x = 0.20 to 0.24. Its distance to the plate grows
  from 0.226 m to 0.265 m.

### 15.3 Episode (`episode.py`)

- Constant grip force (`grip_force` argument) is applied in `close`, `lift`, `carry`, `settle` and
  `lower`. It is 0 in every other phase, which restores position mode for release.
- The `dropped` stop (table contact) applies in `carry`, `settle` and `lower`.
- Slip and the smoothness peaks include `lower` rows, alongside lift, carry and settle as before.
- **Success, revision 4:** `success = in_plate`. The incremental success check in `run_episode` is
  `rest_count >= REST_STEPS`, or native termination with native success. `lifted5`, `max_clear` and
  `success_strict` are still recorded unchanged.

### 15.4 Overseer procedure

Config `lower_release.json`, friction scale 0.25, one episode, `--render`, and this override:
`{"grasp_dx":0,"grasp_dy":0,"grasp_dz":-0.010,"yaw":0,"grip_cmd":0,"t_descend":30,"t_close":15,
"lift_clear":0.05,"t_lift":40,"tilt":0,"t_carry":90,"release_dx":0,"release_dy":0,"t_settle":15}`

Run it with `--grip-force 15` and `--grip-force 25`. The two simulators run in parallel.

Expected peaks:

| Motion | Distance | Duration | Peak speed | Peak acceleration |
|---|---|---|---|---|
| Carry | 0.265 m | 90 steps (3.6 s) | 0.138 m/s | 0.118 m/s² |
| Lift | 0.05 m | 40 steps (1.6 s) | 0.059 m/s | — |
| Lower | 0.04 m | 30 steps (1.2 s) | 0.0625 m/s | — |

**§15.2 amendment (overseer).** The first run (runs/202611032011_ft4_*) stopped at reset because the simulator's RL layer
allows at most ±3 cm of XY shift per object. `lower_release.json` now uses `cube_shift: [0.03, 0.0]`: the cube sits
at x = 0.23, 0.255 m from the plate (default 0.226 m). The carry over 90 steps then peaks at 0.133 m/s and 0.113 m/s².
Pending cleanup: `load_config`'s ±0.05 bound is looser than the simulator's limit.

**§15.4 amendment (overseer).** In run 202611032014, `grasp_dz = −0.010` put the commanded pad 2.5 mm above the table.
The fingertips hit the table and the hand stopped 1 cm high, so the lift reached only 3.97 cm and the plate floor
blocked the lower. The rerun overrides `grasp_dz = +0.005` and `lift_clear = 0.0505`; the extra 0.5 mm covers the
0.2 mm arm sag under load. After the lower, the cube bottom sits about 1 cm above the table, 2 mm over the plate
floor, and the fingertips stay about 7 mm clear of the floor.

## 16. Network policy: imitate the 15 N script at friction 1.0, deploy at 0.25, continue training (user request, 2026-11-03)

Revision 4 showed that the 15 N constant-force script succeeds at friction 1.0 and fails at 0.25, while 25 N
succeeds at 0.25. The user asked for the same question with a learned network:

1. Generate trajectories at friction 1.0 with the 15 N script. They are the motion prior.
2. Train a network policy that reproduces the script's control.
3. Deploy the network at friction 0.25, where it is expected to fail.
4. Continue training it at 0.25 with RL and measure whether the success rate recovers.

The user left the algorithm, reward and representations to the overseer, and suspects the network needs explicit
control of the grip-force magnitude. The scene, success rule (revision 4: the cube ends resting in the plate) and
physics are those of §15 (`lower_release.json` geometry). §16 adds new files. It changes existing code only in
§16.8. The CMA-ES path stays as it is.

### 16.1 Design summary

| Part | Choice |
|---|---|
| Observation | 26 numbers built from simulator state plus a time clock (§16.2) |
| Action | 5 numbers in [−1, 1]: a pad-point displacement (3), gripper close/open (1), grip force (1) (§16.3) |
| Orientation | Not learned. A fixed host schedule: the script's 40-step approach rotation to top-down, then constant (§16.3) |
| Motion prior | 24 script episodes at friction 1.0 and 15 N. Episode 0 is clean; the others add DART-style command noise and record the script's corrective label (§16.5) |
| Pretraining | Behaviour cloning of the Gaussian mean (§16.6) |
| RL | PPO with a separate critic, critic-only warm-up, KL early stop, and a behaviour-cloning term on the motion and gripper outputs only; force is free (§16.6) |
| Reward | Dense lift and carry progress, a slip penalty, a success bonus and a failure penalty (§16.4) |

### 16.2 Observation (`integrations/friction_rl/network.py`)

`observe(state, t, pad_cmd)` returns a float32 vector of length `OBS_DIM = 26`, in this order:

| Values | Definition |
|---|---|
| 1 | `t / HORIZON`, where `t` is the number of actions already taken and `HORIZON = 300` |
| 3 | `(pad_cmd − [0.15, −0.15, 0.85]) / [0.2, 0.2, 0.1]`. `pad_cmd` is the host's commanded pad point (§16.3) |
| 3 | `(pad_cmd − pad_obs) / 0.01`, where `pad_obs = grasp_point(public_state.right_ee_pose)` |
| 3 | `(cube_position − pad_obs) / 0.05` |
| 3 | `(plate_position − pad_obs) / 0.3` |
| 1 | `clearance(cube_position, cube_orientation) / 0.05` (the existing helper in `episode.py`) |
| 4 | `unit_quaternion(cube_orientation)` |
| 3 | `cube_linear_velocity / 0.2` |
| 1 | `gripper_opening.right` |
| 2 | `finger_contacts.right` as 0/1 |
| 2 | `table_contact`, `plate_contact` as 0/1 |

The observation contains no action history and no measured grip force.

### 16.3 Action decoding (`network.py`)

The network outputs `a ∈ R^5`. The host executes `clip(a, −1, 1)`.

Constants: `STEP = 0.01` (m per step), `PAD_FLOOR = 0.778` (the fingertips are 12.7 mm below the pad centre and
the table is at 0.765), `F_MID = 25`, `F_HALF = 20`, `APPROACH_STEPS = 40`.

`Decoder` holds `pad_cmd` and the initial orientation `q0`, both taken at reset from
`public_state.right_ee_pose` (`pad_cmd = grasp_point(pose)`, `q0 = unit_quaternion(pose[3:])`). This matches the
script's starting point.

`Decoder.step(t, a)` returns `(native_action, grip_force, closed)`:

1. Orientation:
   - `q_t = slerp(q0, hand_q(0, 0), s)`, where `s = 10τ³ − 15τ⁴ + 6τ⁵` and `τ = min((t + 1) / APPROACH_STEPS, 1)`.
   - This is exactly the script's approach rotation for yaw 0 and tilt 0. Afterwards the hand stays top-down.
2. Position:
   - `p = pad_cmd + STEP · a[:3]`, then `p_z = max(p_z, PAD_FLOOR)`.
   - `ee = clip(ee_position(p, q_t), ACTION_LOW, ACTION_HIGH)`.
   - Set `pad_cmd = grasp_point(r_[ee, q_t])`.
3. Gripper:
   - `closed = a[3] > 0`.
   - `native_action = encode_action(r_[ee, q_t], 0.0 if closed else 1.0)`.
   - `grip_force = F_MID + F_HALF · a[4]` when closed, otherwise 0. This gives [5, 45] N; 15 N is `a[4] = −0.5`.
   - Force 0 restores position mode, so an open command really opens the gripper. A closed gripper is always in force
     mode, never the stock position drive.

### 16.4 Episode, termination and reward (`network.py`)

`run_net_episode(client, scene, episode_dir, controller)` runs one episode and returns
`(record, trace, batch)`.

- **Controller.** `controller(t, obs, state, decoder)` returns `(stored, executed)`, two length-5 arrays:
  - `executed` is decoded and sent to the simulator;
  - `stored` goes into `batch` (the raw Gaussian sample for PPO, the label for demos).
- **Each step:**
  1. Observe.
  2. Call the controller.
  3. Decode.
  4. Send `{"op": "step", "action": native_action, "grip_force": grip_force}`.
  5. Compute the reward from the new state.
  6. Apply the stop rules.
- **Episode state:**
  - `released` turns true at the first open command after a closed one.
  - `m` is the running maximum of `clearance` over rows where the cube is *held*: both right fingers in contact and
    neither table nor plate contact.
  - `D0` is the cube–plate XY distance at reset.

**Stop rules,** checked in this order after each step (`t` counts the actions taken, including this one):

1. `success`: the rest run reaches `REST_STEPS`, or the native episode terminates with native success. The rest run
   uses the existing `rest_row`, with phase argument `PHASES.index("release")` once `released` is true and 0 before.
2. `native_end`: native terminated or truncated.
3. `dropped`: `table_contact` and `m ≥ 0.005`.
4. `no_lift`: `t ≥ NO_LIFT_STEP = 160` and `m < 0.005`.
5. `timeout`: `t == HORIZON`.

**Reward** of the step that produced the new state:

```
r = 4 · (min(m, .05) − min(m_prev, .05)) / .05          # lift progress (running max)
  + 4 · (c − c_prev)                                        # carry progress
  − 0.25 · slip_mm                                          # slip in the hand
  + 10 · [stop == success] − 3 · [stop in (dropped, no_lift)]
c = [m ≥ .02] · [not table_contact] · (1 − min(d_xy(cube, plate) / D0, 1))
```

- `c_prev` and `m_prev` are the values before the step, and both start at 0.
- `slip_mm = 1000 · ‖o_t − o_{t−1}‖` when both right fingers touch the cube in both rows, and 0 otherwise.
- `o` is the cube offset from the pad point in the hand frame. It is the same transform `outcome()` uses. Move it
  into one helper, `hand_offsets(poses, cubes)`, in `episode.py`, and use it from both places.

**Trace** (saved to `episode_dir/trace.npz`):
- `commanded_ee_pose`, `observed_ee_pose`, `action` (executed);
- `closed`, `grip_force_command` (from state), `right_grip_force`;
- `gripper_opening`, `right_finger_contacts`, `contacts` (table, plate, robot);
- `cube_position`, `cube_quaternion`, `reward`;
- `plate_position`, `stop_reason`.

**Record** (saved to `episode.json`):
- `success`, `stop_reason`, `steps`, `return`;
- `max_clear` (`m` without the cap), `lifted5`;
- `slip`: the largest hand-frame offset change from the first held row to the last closed row;
- `force_mean_held` and `force_min_held`: commanded force over rows that are closed and have both fingers in contact;
- `grip_measured_held`: mean measured force over the same rows;
- `release_dist`: cube XY distance to the plate in the last row;
- `cmd_peak_speed` and `cmd_peak_accel`: `motion_peaks` of the commanded poses over closed rows;
- `native_success`.

`batch` holds `obs` (T×26), `stored` (T×5), `reward` (T), and `done` (T). `done` is true only in the last row;
every stop is terminal, because the observation includes the clock.

### 16.5 Motion prior: script demonstrations (`network.py`)

- `ScriptController(params, force, sigma, rho, rng)` wraps `PrimitivePolicy(params)`. Each step, take the script's
  next command and phase:
  - `pad*_t = grasp_point(script commanded pose)`;
  - `closed* = phase in (close, lift, carry, settle, lower)`.
- Noise `η` follows a per-axis AR(1) process:
  - `η_t = ρ η_{t−1} + sqrt(1 − ρ²) σ ε_t`, clipped to ±2σ, with `η_{−1} = 0`.
  - `σ = demo_noise` (per axis), `ρ = demo_noise_rho`.
- **Label** (stored):
  `[clip((pad*_t − pad_cmd) / STEP, −1, 1), +1 if closed* else −1, (force − F_MID) / F_HALF]`.
- **Executed:** the label with the first three entries replaced by `clip((pad*_t + η_t − pad_cmd) / STEP, −1, 1)`.
- `pad_cmd` is the decoder's value before the step. The label therefore always points back to the script's track,
  as in DART.
- With `σ = 0` the executed commands reproduce the script exactly: same native actions and same grip forces.
- `HORIZON = 300` is shorter than the script's 315 steps, so the script is never asked past its end.

### 16.6 Learner (`integrations/friction_rl/ppo.py`, PyTorch, CPU)

- **`ActorCritic(obs_dim, act_dim)`:**
  - actor: Linear(26, 256)–Tanh–Linear(256, 256)–Tanh–Linear(256, 5), which gives the mean;
  - critic: the same shape with output 1;
  - a state-independent `log_std` parameter of length 5.
  - `save(path, model)` and `load(path)` use `torch.save`/`torch.load` of the state dict.
- **`act(model, obs, rng, stochastic)`:** returns `(stored, executed)`.
  - stochastic: `stored = μ + σ ⊙ rng.standard_normal(5)`;
  - deterministic: `stored = μ`;
  - in both cases `executed = clip(stored, −1, 1)`.
- **`bc_fit(model, obs, labels, steps, batch, lr, seed)`:**
  - minimises the MSE of the actor mean against all 5 label dimensions with Adam;
  - logs the loss every 100 steps;
  - afterwards sets `log_std = log(init_std)`.
- **`gae(rewards, values, dones, gamma, lam)`:** standard GAE. A terminal row bootstraps 0.
- **`ppo_update(model, batch, demos, cfg, actor_frozen)`:**
  1. Compute `logp_old` and `values` with the pre-update model.
  2. Compute GAE advantages and returns per episode, then normalise the advantages over the batch.
  3. Run up to `epochs` passes of shuffled minibatches of size `minibatch`. Each pass computes:
     - the clipped surrogate (`clip`);
     - the critic MSE;
     - `bc_weight · MSE(μ[:, :4], demo_labels[:, :4])` on an equal-size random minibatch of demo rows.

     This term covers motion and gripper only; the force output (index 4) is never regularised. It is the
     motion prior during RL.
  4. Use separate Adam optimizers for the actor (`actor_lr`, which includes `log_std`) and the critic
     (`critic_lr`), with gradient-norm clipping at 0.5.
  5. Stop the epochs early when the minibatch approximate KL `mean((ratio − 1) − log ratio)` exceeds
     `1.5 · target_kl`.
  6. With `actor_frozen`, only the critic is updated.
  7. Return a dict: `policy_loss`, `value_loss`, `bc_loss`, `approx_kl`, `epochs_run`.
- No observation or reward normalisation and no entropy bonus.

### 16.7 CLI (`orchestrator/friction_net.py`) and config (`tasks/FrictionTransfer/network.json`)

Every subcommand requires a new `--run-dir` with a `YYYYMMDDHHMM` prefix and `--config`.

- The scene is always `build_scene(0, cfg)` (§15.2 helper). `layout_radius` is 0, so there is one layout.
- Simulator subcommands start `num_envs` clients with `simulator_pool` from `orchestrator/friction_rl.py` (gripper
  scale 1.0). They run episodes through `SimulatorPool.map` (§16.8).

**Subcommands:**

| Command | Arguments | Does | Writes |
|---|---|---|---|
| `demos` | `--friction-scale` | runs `demo_episodes` script episodes. Episode `k` uses `σ = 0` for k = 0 and `demo_noise` otherwise, with seed `[seed, 0, k]`; grip force `script_force` | `demos.npz` (`obs`, `labels`, `episode`), `episodes.jsonl` |
| `bc` | `--demos` | `bc_fit` (no simulator) | `policy.pt`, `bc.jsonl` |
| `evaluate` | `--policy`, `--friction-scale`, `--episodes N`, `--stochastic`, `--render` | N episodes `eval_e{k:02d}` with seed `[seed, 2, k]` | `episodes.jsonl`, `summary.json` |
| `train` | `--init`, `--demos`, `--friction-scale` | PPO iterations (below) | `iterations.jsonl`, `episodes.jsonl`, `evals.jsonl`, `policy_it{i:03d}.pt`, `policy.pt` |

`summary.json` contains:
- `successes` and `N`;
- `stop_reasons` (counts);
- the means of `max_clear`, `slip`, `force_mean_held` and `release_dist`;
- the maxima of `cmd_peak_speed` and `cmd_peak_accel`.

**`train` iteration `i`** (from 0 to `max_iterations − 1`):

1. Run `episodes_per_iter` stochastic episodes `it{i:03d}_e{j:02d}` with seeds `[seed, 1, i, j]`.
2. Call `ppo_update` with `actor_frozen = i < warmup_iterations`.
3. Append to `iterations.jsonl`:
   - `i`, `successes`, `stop_reasons`;
   - the mean `return`, `max_clear`, `slip` and `force_mean_held`;
   - `mu_force_demo`: the mean actor force in newtons over the demo rows labelled closed;
   - `std` (exp `log_std`), the update dict, and `wall_time`.
4. Save `policy_it{i:03d}.pt`.
5. Run a deterministic check of `num_envs` episodes `it{i:03d}_det{k}` when `(i + 1) % eval_every == 0`, or when
   this iteration and the previous one both reach `stop_successes`. Append the result to `evals.jsonl` with the
   `summary` fields.
6. Stop after a deterministic check in which every episode succeeds and the last two iterations each reached
   `stop_successes`.

`policy.pt` is the final iteration's model.

`network.json` has exactly these keys and values. Validate the keys and types only.

```json
{
  "simulator_python": ".venv-robodojo/bin/python", "startup_timeout": 1200, "step_timeout": 120,
  "seed": 0, "num_envs": 2, "layout_radius": 0.0, "cube_shift": [0.03, 0.0],
  "script": {"grasp_dx": 0, "grasp_dy": 0, "grasp_dz": 0.005, "yaw": 0, "grip_cmd": 0, "t_descend": 30,
             "t_close": 15, "lift_clear": 0.0505, "t_lift": 40, "tilt": 0, "t_carry": 90,
             "release_dx": 0, "release_dy": 0, "t_settle": 15},
  "script_force": 15.0,
  "demo_episodes": 24, "demo_noise": [0.002, 0.002, 0.001], "demo_noise_rho": 0.95,
  "bc_steps": 4000, "bc_batch": 256, "bc_lr": 0.001,
  "init_std": [0.03, 0.03, 0.03, 0.1, 0.15],
  "episodes_per_iter": 8, "max_iterations": 40, "warmup_iterations": 2,
  "gamma": 0.99, "gae_lambda": 0.95, "clip": 0.2, "epochs": 8, "minibatch": 512,
  "actor_lr": 0.0002, "critic_lr": 0.0005, "target_kl": 0.02, "bc_weight": 1.0,
  "eval_every": 3, "stop_successes": 7
}
```

### 16.8 Permitted edits to existing code

- `orchestrator/friction_rl.py`:
  - add `SimulatorPool.map(function, jobs)`, which runs `function(client, job)` over the shared queue and returns
    the results in job order;
  - `run` becomes a call to `map`;
  - tighten the `cube_shift` bound in `load_config` from ±0.05 to ±0.03, the simulator's limit.
- `integrations/friction_rl/episode.py`: extract `hand_offsets` (§16.4). Behaviour must not change.

### 16.9 Overseer procedure (GPU)

| Step | Command | Expected |
|---|---|---|
| D | `demos --friction-scale 1.0` | The clean demo succeeds like runs/202611032027_ft4_f100_F15; the noisy demos mostly succeed |
| B | `bc --demos D/demos.npz` | Loss well below 1e-3 |
| E1 | `evaluate` at 1.0: deterministic ×2 (one rendered) and stochastic ×10 | Succeeds, about 15 N |
| E2 | the same at 0.25 | Fails: the cube slips out, as the 15 N script did |
| T | `train --init B/policy.pt --demos D/demos.npz --friction-scale 0.25` | Success rate rises; `mu_force_demo` rises above about 20 N |
| E3 | `evaluate` T/policy.pt at 0.25: deterministic ×2 (one rendered) and stochastic ×10 | Recovered success |

**§16.9 amendment (overseer, D and B).**
- Demonstrations: runs/202611032110_ft5_demos_f100.
  - The clean demo reproduces the 15 N script exactly: success, 275 steps, slip 0.3 mm.
  - 20 of the 23 noisy demos succeed. The 3 drops (and 3 runs with 30–40 mm slip) are noise-induced off-centre
    grasps, up to ±4 mm. The cube pivots about the two-finger axis during settle or lower.
  - All 24 episodes are kept for behaviour cloning. Their labels are the script's corrective commands regardless of
    the outcome.
- Behaviour cloning with 4,000 steps (runs/*_ft5_bc) plateaued at a loss of 0.014. `bc_steps` was raised to 20,000
  (runs/*_ft5_bc20k).
  - Per-dimension MSE: x 6e-3, y 1e-3, z 3e-3, force 3e-5.
  - Gripper MSE is 0.016, which comes from the two transition rows (close and release onsets).

**§16.2 amendment (overseer, after E1/E2 of the first BC policy).**

The first BC policy (runs/202611032120_ft5_bc20k) behaved as follows:
- At 1.0: 2/2 deterministic and 10/10 stochastic.
- At 0.25: 0/2 and 0/10, the expected failure. Records: runs/*_ft5_bc_f100_{det,sto} and runs/*_ft5_bc_f025_{det,sto}.

Two defects make it a poor starting point for RL and a poor replica of the smooth script:
1. **Over-correction.**
   - The lag input `(pad_cmd − pad_obs)/0.01` dominates the network's response to its own command. In the demos it
     mostly encodes velocity.
   - The gain of the network's correction on `pad_cmd` is 2.4–3.6 in the lift and carry, where a gain of 1 is
     ideal. Gains above 2 oscillate.
   - Exploration noise therefore grows into step-to-step oscillation (peak commanded acceleration 3.6–4.4 m/s²).
2. **Coarse timing.**
   - A single clock `t/300` cannot resolve the gripper's step changes.
   - The deterministic policy closes 10 steps late and releases 18 steps late, after the clock-driven withdraw has
     started to lift the cube again.

Revised observation, `OBS_DIM = 33`:
- Remove the lag block.
- Replace the single clock with 11 clock features: `t/300`, then `sin(2πkt/300)` and `cos(2πkt/300)` for each
  k in (1, 2, 4, 8, 16).
- All other blocks are unchanged and keep their order.

The demos are re-collected, because `demos.npz` stores observations.

**§16.9 results with the revised observation (overseer).**
- D: runs/202611032210_ft5b_demos_f100. 21 of 24 demos succeed. The clean demo is identical to the script.
- B: runs/202611032219_ft5b_bc.
  - BC loss 1.7e-4; every per-dimension MSE ≤ 5e-4.
  - The clean demo's gripper closes at t = 70 and last closes at t = 259, exactly as the script does.
  - The correction gain is 0.5–0.8 on the command alone and about 1.0 for a rigid hand shift: stable.
- E1 at 1.0 (runs/*_ft5_bbc_f100_{det,sto}):
  - deterministic 2/2: 275 steps, 0.3 mm slip, 15.0 N, ends 0.2 cm from the centre, peaks 0.132 m/s and
    0.33 m/s²;
  - stochastic 10/10.
- E2 at 0.25 (runs/*_ft5_bbc_f025_{det,sto}):
  - deterministic 0/2, `dropped` at step 110: lift 1.1 cm, slip 27 mm;
  - stochastic 0/10: 4 dropped, 6 no_lift.
  - This is the expected failure, the same as the 15 N script.
- T launched from this BC policy.

**§16.6 amendment (overseer, after T in runs/202611032237_ft5b_ppo_f025, stopped at iteration 10).**

- Symptom: KL of 3–6 per update, an early stop after one minibatch, a 100× rise in BC loss, force 15.0 → 15.3 N,
  and 0/8 successes throughout.
- Cause: new Adam optimizers were created in every `ppo_update` (a sign step of size `lr` on every parameter),
  together with `actor_lr = 2e-4` against position std 0.03.

Fix:
1. `train` creates the actor and critic Adam optimizers once, from the initial model, and passes them to every
   `ppo_update`. Their state persists across iterations. Adam's per-parameter normalisation then damps the noisy
   position gradients while the consistent force gradient keeps full-size steps.
2. `actor_lr = 2e-5`.
3. `ppo_update` also returns `updates`, the number of actor minibatch steps actually taken.

Everything else is unchanged.

**§16.6 amendment 2 (overseer, CPU check before relaunch).**

Persistent optimizers alone are not enough.
- Position std 0.03 makes the KL hypersensitive to tiny shifts in the motion means.
- At `actor_lr = 2e-5`, the KL budget is spent after 1–4 minibatch steps on random motion drift. The force mean moves
  about 0.02 N per iteration.
- This was measured with synthetic batches of 1,100 rows built from demo states.

Revised RL design. Exploration and the policy gradient act only on the gripper and force channels:
1. `ppo.py` gets the constant `EXPLORE = (3, 4)`.
   - `act(..., stochastic=True)` adds Gaussian noise only to those dimensions.
   - The position dimensions execute the actor mean.
   - PPO log-probabilities, ratios and KL use only the `EXPLORE` dimensions.
2. The arm motion remains a network output, trained through the shared layers. The behaviour-cloning term (motion and
   gripper, weight `bc_weight`) keeps it on the 1.0 motion prior. It is no longer explored by RL.
3. `train` sets `log_std = log(init_std)` after loading `--init`, so the RL exploration noise comes from the config.
4. Config: `init_std = [0.03, 0.03, 0.03, 0.1, 0.25]` (force std 5 N), `actor_lr = 1e-4`.

CPU check (synthetic force-dependent reward, three learning rates):

| `actor_lr` | KL per update | Actor steps per iteration | BC loss | Force mean |
|---|---|---|---|---|
| 1e-4 | 0.012–0.018 | up to 24 | stays ≤ 7e-4 | rises steadily |

Disclosure for the report: this restricts what RL may change to the grip and gripper timing. The arm motion is held
near the imitation prior.

**§16.9 T results, first 40 iterations (runs/202611032318_ft5d_ppo_f025).**
- Success was 0/8 at every iteration, though the run is healthy:
  - KL 0.009–0.016 per update, BC loss rising slowly from 2.6e-4 to 1.2e-3;
  - mean return −7.4 → −3.8, held force 15 → 25.6 N, lift 0.5 → 4.95 cm.
- The force output depends on state. RL raised it where the failures occurred, about 30 N in the lift. It still
  decays to 13–17 N early in the carry, where every episode now drops.
- Continuation: a new `train` run from that `policy.pt`, with `tasks/FrictionTransfer/network_continue.json`. It
  equals `network.json` except `actor_lr` 2e-4, `target_kl` 0.04 and `warmup_iterations` 0 (the critic is already
  trained).

**§16.9 final results (overseer, 2026-11-04).**
- T2: runs/202611040011_ft5d_ppo2_f025, continued from T's policy.pt for 40 more iterations (80 in total).
  - First rev4 successes at iteration 63 overall: stochastic 1/8, deterministic check 2/2.
  - The last ten iterations reached 2–6/8. The stop rule never fired.
- E3 (policy runs/202611040011_ft5d_ppo2_f025/policy.pt, friction 0.25):

  | Mode | Ends in plate (rev4) | Clean |
  |---|---|---|
  | Deterministic (runs/*_ft5_rl_f025_det rendered, *_det10) | 10/12 | 0/12 |
  | Stochastic (runs/*_ft5_rl_f025_sto) | 4/10 | 0/10 |

  - Clean = held to the open command, held slip ≤ 10 mm, peaks ≤ 0.25 m/s and 0.5 m/s².
  - Mean held force is 26.6 N (deterministic): about 30–35 N in the lift and most of the carry.
- Mechanism of the rev4 successes (corrected 2026-11-04 after a step-by-step trace review; the earlier wording,
  "the force sags to 5–11 N and the cube slides about 4 cm down inside the fingers", was wrong):
  - The grip holds through the lift and carry: force 23–39 N, held slip ≤ 2.4 mm up to t = 191, and the cube stays
    at a clearance of 4.8–5.0 cm.
  - Over t = 184–195, as the cube nears the plate centre (5.4 → 1.8 cm away), the commanded force ramps down from
    27 to 13.5 N. That is the imitation network's level (15 N) and below the about 20 N needed at 0.25.
  - The cube slides about 2 cm in 9 steps and leaves the closed fingers at t = 200, 0.8 cm from the plate centre.
    It then drops the last about 2 cm onto the plate floor.
  - The gripper command stays closed until t = 260. In the video this looks like a release above the plate. The
    5–8 N commanded at t = 205–235 comes after the cube has gone.
  - Inferred cause: the reward does not separate dropping into the plate from placing into it. Once the
    force-raising frontier reached the plate, no gradient kept the force up there.
- Failures:
  - Deterministic e08/e09 are identical to the successes up to t = 221.
    - At t = 222 the observed hand leaves the commanded pose by 5–23 cm, uncommanded (the command moves ≤ 1.4 cm per
      step). It knocks the cube out of the plate at 2.1 m/s, and the cube ends 49 cm away.
    - Every deterministic episode shows a 2–10 cm tracking excursion from t = 222. Its cause is not established.
    - At t = 223–224 the policy's motion outputs saturate at ±1: with the cube gone, this is a never-demonstrated
      state.
  - Stochastic failures:
    - e01 loses the cube 7.8 cm from the centre (t = 179).
    - The other five leave the plate after landing in it (t = 203–260), at 0.7–1.4 m/s.
  - These also explain the commanded peaks of 0.29–0.39 m/s and 10–18 m/s².
- The simulator is not bit-repeatable across episodes: the same deterministic policy gives different outcomes. Each
  pair of workers is identical.
- Review page: runs/202611040134_ft5_review (scratchpad build_ft5_page.py, ft5_template.html, ft5_clean.py).
