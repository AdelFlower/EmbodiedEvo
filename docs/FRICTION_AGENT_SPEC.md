# Friction agent: autonomous diagnosis, source reconstruction and RL repair (FA)

User request, 2026-11-04:

> The agent first acquires an RL policy in "Source Environment 1.0" (we can use a pre-trained policy here). Upon
> transferring to "Target Environment 0.25," the agent fails; it must then reflect and conduct experiments to identify
> and replicate these dynamic discrepancies. Following this, the agent should perform further RL training within a
> corrected, self-constructed environment—with the flexibility to modify components such as the reward function or
> the underlying dynamics interface—ultimately enabling it to successfully complete the task in the target
> environment.

This is the loop of `new_project_plan.md` (probe, train, deploy, status and submit; an immutable target; a
reproduction gate; budgets; host selection), rebuilt on the friction-transfer network of
`FRICTION_TRANSFER_SPEC.md` §16. It deliberately changes the original plan in these ways:

| Item | Original plan | This spec |
|---|---|---|
| P0 | SAC network | FT5 imitation network `runs/202611032219_ft5b_bc/policy.pt` (imitates the 15 N script at friction 1.0) |
| Training | SAC, later PPO, with a fixed reward | FT5 PPO (§16.6), with the reward replaceable by an agent-written program |
| Scene | 5 random development layouts | One scene (§16; `cube_shift` [0.03, 0]); development *episodes* D0–D3 |
| Success | Native | Strict (held until the gripper opens, ends in the plate); lenient reported too |
| Control branch | Required | Deferred (not in this milestone) |

## 1. Rules for the implementer (normative)

These are the same rules as `FRICTION_TRANSFER_SPEC.md` §2.

- Write no defensive programming. Use no `try/except` except to keep a failed job's evidence (§7.4).
- Validate only at trust boundaries: config keys, CLI arguments, agent requests and reward-program output.
- Add no hashes, checksums or verification files, and never re-read a file you just wrote.
- Add no options, modes or fallbacks beyond those stated here.
- Give every quantity one definition, and reuse existing helpers. The one sanctioned duplicate is
  `frictionlib.py` (§6.2), and a test pins it to the host definitions.
- Write one-line docstrings. Records are JSON or JSON lines. Do not touch `vendor/` or `simulators/`.

## 2. Fixed host setup

- **Scene:** `build_scene(0, cfg)` with `layout_radius` 0 and `cube_shift` [0.03, 0]. This is the FT5 scene.
- **Physics:** four keys go to the worker: `friction_scale`, `mass_scale`, `restitution` and `gripper_scale`.

  | Environment | Values | Who sets them |
  |---|---|---|
  | Nominal source | 1.0 / 1.0 / 0.0 / 1.0 | host |
  | Target | `target_parameters` in the config: friction 0.25, others nominal | host, immutable, never exported |
  | Calibrated source | `friction_scale` ∈ [0.05, 2], `mass_scale` ∈ [0.25, 4], `restitution` ∈ [0, 1] | agent |

  `gripper_scale` is always 1.0. Parameters the agent omits take their nominal values.
- **P0 (network N000):** a copy of `p0.policy`, with `p0.demos` as the motion prior. Both are config paths.
- **Episode:** `run_net_episode` (§16.4) unchanged, including its stop rules, its 300-step horizon and the host
  default reward. §5.1 adds outputs only.
- **Training procedure** (frozen; the agent chooses only parameters, reward and iteration count):
  - FT5 PPO (`ppo_update`, `act`, `EXPLORE` = gripper and force) with the config's `training` block. Its values
    equal `tasks/FrictionTransfer/network_continue.json`, except `warmup_iterations` = 2.
  - The BC term uses the P0 demos (source data; never target data).
  - Each round continues the parent's model and both Adam optimizers. Round 1 creates the optimizers.
    `log_std` is never reset.
  - The first `warmup_iterations` iterations of a round are critic-only (`actor_frozen`) when the parent is N000, or
    when the round's reward program text differs from the parent's. These iterations count against the budget.
  - Each iteration runs `episodes_per_iter` stochastic episodes in the calibrated source. The rewards come from the
    round's reward program (§6), then `ppo_update` runs.

## 3. Episodes, outcomes and success

- **Development episodes D0–D3:** D0 and D1 deterministic (policy mean); D2 and D3 stochastic, seeded
  `[seed, 7, k]` for Dk. Every deployment and the baseline run all four. Probes run any nonempty subset; Dk always
  has the same mode and seed.
- **Evaluation episodes:** `eval_deterministic` deterministic episodes and `eval_stochastic` stochastic episodes,
  seeded `[seed, 8, k]`.
- **Pools and rendering:** rollouts use a pool of `num_envs` simulators in render mode, so videos are produced.
  Training uses state mode. Episodes are assigned through the shared queue (`SimulatorPool.map`). The simulator is
  not bit-repeatable across a worker's successive episodes; disclose this, but do not work around it.
- **Lenient success:** `record["success"]`, the rev4 rule.
- **Held until release** (new pure function `held_to_release(trace)` in `network.py`):
  - `lift` = the first row with both finger contacts, no table or plate contact, and cube clearance > 0.002.
  - `open` = the first row r > `lift` with `closed[r-1]` and not `closed[r]`.
  - The result is true iff both rows exist and both fingers touch the cube on every row in [`lift`, `open` − 1].
  - Re-contact after a drop therefore does not count.
- **Strict success** = lenient success and held until release. It is the host success used for the reproduction
  gate, for selection and for acceptance. Lenient success is reported next to it everywhere.

## 4. Stages and CLI (`orchestrator/friction_agent.py`)

Every command takes a new `--run-dir` with a `YYYYMMDDHHMM` prefix, and `--gpu` (default 0).

- `run --config C [--scripted-agent F]` runs these stages in order, each writing `<stage>/result.json`:
  1. **freeze:** copy `config.json` (with the agent block resolved), P0 to `networks/N000/policy.pt`, and the demos
     to `prior/demos.npz`.
  2. **baseline:** P0 on D0–D3 in the nominal source and in the target (two rollout jobs, render mode).
     `baseline/{source,target}/`.
  3. **calibration:** prepare the workspace (§6.1), start the service (§5) and the agent (§8), and serve until the
     agent submits, exits, or the wall budget ends. Copy the agent sessions and write `selection.json` (§5.4).
- `evaluate --run-dir R` (after calibration):
  - rolls out P0 and the selected network (once if they are the same) at the target on the evaluation episodes;
  - writes `evaluation/results.json` with strict and lenient counts per arm;
  - sets `accepted` = strict successes of the selected network ≥ `required_successes`.
- `trainability --config C --parameters P.json --reward R.py --iterations N --run-dir R` is the host-only oracle
  check:
  - a train job from N000 at exactly P, using reward program R, then a rollout of the result on the evaluation
    episodes at the target;
  - no agent and no budgets; its outputs never enter an agent workspace.
- `summarize --run-dir R` writes `summary.json`: per-stage times, the network table with strict and lenient
  development outcomes, budgets used, evaluation results, model token usage, and the agent's diagnosis. It prints
  the same information.

## 5. Experiment service (`orchestrator/friction_agent_service.py`)

Follow `orchestrator/adaptation_service.py`: one request per connection, one job at a time, `requests.jsonl`
logging every request and response (refusals included), and `poll()` driving job phases. Operations:

| op | fields | effect |
|---|---|---|
| `status` | — | budgets used/limit, seconds left, running job, network table, jobs (id, kind, status), submitted |
| `probe` | `network`, `parameters`, `episodes` (subset of 0–3), `hypothesis`, `prediction` | rollout job in the calibrated source; charges `len(episodes)` diagnostic episodes |
| `train` | `parameters`, `iterations`, `reward` (workspace-relative path or `"default"`), `rationale` | train job, then automatic deployment on D0–D3 at the target; charges 1 round, `iterations`, 4 target episodes |
| `wait` | `job`, `seconds` (1–900) | block until the job leaves `running` or the interval ends |
| `submit` | `diagnosis` | write `diagnosis.json` (schema §8.2); the branch ends |

### 5.1 Validation (refusals consume nothing)

- The usual checks apply:
  - exact field set;
  - nonempty explanation strings;
  - parameter keys and ranges;
  - a known network;
  - distinct episode indices;
  - no running job, branch not ended.
- `iterations` is an integer in [4, `iterations_per_round`], and no more than the iterations remaining.
- `train` also requires the following:
  - **Budgets remain:** rounds and target episodes.
  - **Reproduction gate:** the parent is the latest network. It has development outcomes with at least one strict
    target failure. There is also a completed probe of the parent at exactly these parameters (nominal-filled) that:
    - matches the parent's strict target outcome on every probed episode;
    - includes at least one of the parent's target failures.
  - **Reward program:**
    - `"default"` resolves to `scripts/default_reward.py`.
    - Otherwise the path must resolve inside the workspace (no `..` or absolute paths) to a regular file of at most
      64 KiB.
    - Its text is snapshotted. A dry run in the sandbox (§6.3) on the four baseline *source* traces must succeed.
      The response includes each episode's reward sum.
    - A failed dry run is a refusal, and the error carries the runner's stderr tail (the agent's own code).

### 5.2 Jobs

- **Job directories:** `calibration/jobs/J####/` contain `job.json`, `result.json` and the phase subdirectories
  `episodes/`, `train/` and `deploy/`.
- **Train job:**
  - writes `calibration/networks/N###/` with `policy.pt`, `training.pt` (model, both optimizer states, cumulative
    iterations and episodes), `reward.py` (the snapshot) and `iterations.jsonl`;
  - then deploys.
- **Public copies** go under `workspace/observations/jobs/J####/`:
  - `request.json`;
  - for probes, `episodes/Dk/`;
  - for training, `training/` and `deploy/Dk/`.
- **Network table rows:** id, round, parent, parameters, reward path as requested, iterations, and strict and
  lenient development outcomes (`null` until deployed).

### 5.3 Budgets (config `budgets`)

- **Limits:** `wall_hours` 12, `rounds` 4, `iterations_per_round` 60, `iterations_total` 160, `target_episodes` 16,
  `diagnostic_episodes` 24.
- **History:** the first values were 8 h, 40 per round and 100 in total. The overseer raised them before the formal
  run, after the oracle check (§11 step 3). The oracle first reached strict success at iteration 52 and 8/8 at
  iteration 60. That left too little margin for an agent that also spends rounds on diagnosis or on a wrong reward.
  Training always continues the latest network, so a bad round must be trained back out. The wall limit grew so
  that 160 iterations at about 2 minutes each, plus probes and deployments, finish well before the deadline.
- **Charging:** accepted jobs are charged when they start, even if they fail. Target episodes are charged when
  deployment starts.
- **At the deadline:** the service stops a running job and drops that job's network.

### 5.4 Selection

- **Candidates:** N000 (with its baseline target outcomes) and every deployed network.
- **Order:** the most strict development successes win, then the most lenient ones, then the earlier network. The
  agent's preferred network is recorded but does not select.

## 6. Agent workspace and reward programs

### 6.1 Workspace (`tasks/FrictionAgent/Workspace/` → `calibration/workspace/`)

- **Read-only mounts:**
  - `instruction.md`, with `{{...}}` placeholders filled from the config;
  - `nominal.json` (nominal values and ranges);
  - `scripts/`, containing `adapt.py`, `frictionlib.py`, `reward_runner.py` and `default_reward.py`;
  - `observations/`.
- **Writable:** the agent creates `notes/`, `rewards/` and `submission/`.
- **Observations:** `observations/baseline/{source,target}/Dk/`, plus the job exports.
- **Never present:** target parameters, run paths, P0 provenance, any 0.25 run, host code, host default-reward
  internals beyond `default_reward.py`, and network weights.

### 6.2 Public episode export (in `reward_sandbox.py`, used for observations and for reward programs)

- `trajectory.npz`, with the keys below. Rows are post-step states.

  | Key | Shape |
  |---|---|
  | `commanded_ee_pose` | (T, 7) |
  | `observed_ee_pose` | (T, 7) |
  | `action` (executed, clipped network output) | (T, 5) |
  | `closed` | (T,) |
  | `grip_force_command` | (T,) |
  | `right_grip_force` | (T, 2) |
  | `gripper_opening` | (T,) |
  | `right_finger_contacts` | (T, 2) |
  | `contacts` (table, plate, robot) | (T, 3) |
  | `cube_position` | (T, 3) |
  | `cube_quaternion` | (T, 4) |
  | `plate_position` | (3,) |
  | `initial_observed_ee_pose` | (7,) |
  | `initial_cube_position` | (3,) |
  | `initial_cube_quaternion` | (4,) |
  | `initial_right_finger_contacts` | (2,) |
  | `stop_reason` | 0-d string |

  `run_net_episode` must add the four `initial_*` arrays to `trace.npz` (§9).
- `result.json`, with keys:
  - outcome: `strict_success`, `success_lenient`, `held_to_release`, `stop_reason`, `steps`;
  - motion and grip: `max_clear`, `slip`, `force_mean_held`, `force_min_held`, `release_dist`, `cmd_peak_speed`,
    `cmd_peak_accel`.
- **Videos:** `cam_head.mp4`, `cam_left_wrist.mp4` and `cam_right_wrist.mp4`, hard-linked when they exist.
- **Training exports:**
  - `training/iterations.csv`: iteration, warmup, strict successes, lenient successes, mean return, mean
    `max_clear`, mean slip, mean `force_mean_held`, `approx_kl`, `policy_loss`, `value_loss`;
  - `training/last_iteration/Ek/`: `trajectory.npz` and `result.json` for the final iteration's episodes, plus
    `rewards.npy` (the rewards used).
- **`frictionlib.py`:** NumPy-only copies of `clearance`, `grasp_point` and `hand_offsets`, with `load_trace(path)`.
  A test pins them to the host functions on random inputs.

### 6.3 Reward programs and the sandbox

- **Program:** a Python file defining `reward(trace) -> 1-D array` of length `steps`.
  - `trace` is the dict loaded from `trajectory.npz`, with `stop_reason` as `str`.
  - The program may import `numpy` and `frictionlib`.
  - Termination is fixed by the host; the program sets only the per-step rewards.
- **Runner:** `reward_runner.py PROGRAM IN_DIR OUT_DIR`. For every `IN_DIR/*.npz` it calls `reward` and saves
  `OUT_DIR/<name>.npy`, and it exits nonzero on any exception, wrong shape or non-finite value.
- **Sandbox call:** one `docker run` per batch (one PPO iteration, or one dry run).
  - Image: config `sandbox.image`, which is `embodiedevo-agent:latest`.
  - Flags: `--rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 64
    --memory {sandbox.memory_mb}m --cpus 1 --user 1000:1000`.
  - Mounts: `/code` (read-only) holds the program, `frictionlib.py` and `reward_runner.py`. `/in` (read-only) holds
    the episodes; `/out` is writable.
  - Timeout: `sandbox.timeout` seconds.
- **Host checks after the call:** one array per episode, shape `(steps,)`, finite, and |r| ≤ 1e4.
- **Failure:** during training, any failure fails the job, and the job's error (with the stderr tail) is exported.
- **`default_reward.py`:** reproduces `run_net_episode`'s inline reward exactly from the public trace (§16.4
  reward; it uses the `initial_*` arrays for row 0). A test checks agreement to 1e-9 on fake episodes.

## 7. Job runner (`orchestrator/friction_agent_jobs.py`)

Follow `orchestrator/adaptation_jobs.py`:
- `start_job(dir, spec)` launches `python -m orchestrator.friction_agent_jobs DIR` in a new session;
- `execute_job` writes exactly one `result.json`.

1. **rollout** (spec: network path, physics, episode list of (name, deterministic, seed)):
   - start the pool (§9) in render mode and run the episodes;
   - the result lists every planned episode, with `complete`, both successes, the record, and `episode_dir`.
2. **train** (spec: parent `training.pt` or N000 `policy.pt`, physics, iterations, reward snapshot path, round,
   warmup flag, output dir):
   - run the §2 PPO loop;
   - append one `iterations.jsonl` row per iteration, with the export columns plus wall time;
   - save `policy_it###.pt` per iteration, and the final `policy.pt`/`training.pt`.
   - Training episode seeds are `[seed, 1, round, i, j]`.
3. **Failures:** keep the traceback in `result.json` (`status` failed), and keep partial episodes.

## 8. The agent

### 8.1 Launch

- **Codex:** `launch_agent.agent_argv("codex", ...)` with `model` `gpt-6-astra`, `reasoning_effort` `high` and new
  keyword `service_tier="fast"`. The keyword emits `-c service_tier="fast"`, and it defaults to `None` (no flag), so
  the RoboDojo benchmark is unchanged.
- **Container:** Docker as in `dynamics_adaptation.agent_command`. The socket directory is mounted at
  `/run/embodiedevo`; instruction, `nominal.json`, `scripts` and `observations` are read-only.
- **Prompt:** "Read instruction.md and work autonomously. Use scripts/adapt.py for all experiments. Write analysis
  code under notes/ and reward programs under rewards/. Finish by submitting submission/diagnosis.json with
  scripts/adapt.py submit."
- **Records:** agent sessions are copied to `calibration/agent_sessions/`, and token usage via `codex_token_usage`.

### 8.2 `instruction.md` content (normative points; write it as plain prose for the agent)

- **Task:**
  - Pick up the red cube, lift it until its bottom is about 5 cm above the table, carry it level over the plate
    centre, lower it about 4 cm, and open the gripper.
  - Strict success: the cube is held by both fingers from lift-off until the gripper opens, then rests in the plate
    (centre ≤ 4.5 cm from the plate centre, inside the plate's height band, for 13 steps, with no robot contact).
  - "Ends in plate" (lenient) is also reported.
- **Situation:** N000 succeeds in the nominal source but may fail in a target whose dynamics differ in unknown ways.
  You never send robot actions; the host runs registered networks only.
- **Policy interface:**
  - Observation: the §16.2 list in plain words.
  - Action: five numbers in [−1, 1]:
    - pad displacement of up to 1 cm per step per axis;
    - gripper closed when a₃ > 0;
    - squeeze force 25 + 20·a₄ N while closed (5–45 N).
  - Orientation follows a fixed top-down schedule. 25 Hz; at most 300 steps.
- **Training:**
  - PPO continuing from the latest network.
  - Exploration and meaningful change happen only in the gripper and force outputs; an imitation term keeps the
    motion near the original.
  - Each iteration runs 8 stochastic episodes. The first 2 iterations of a round train only the value function
    when the reward changes.
  - The agent chooses the physics parameters, the reward program and the iteration count; nothing else.
- **Default reward:** the formula in words, plus `scripts/default_reward.py`. How to write a reward program, how to
  test it locally (`python3 scripts/reward_runner.py ...` on any exported trajectory), and the limits.
- **Operations:** examples for each `adapt.py` command; one job at a time; `wait` up to 900 s.
- **Budgets:** the numbers in the placeholders; refusals cost nothing.
- **Rules:**
  - Reproduce before training (enforced; quote §5.1).
  - Record a hypothesis and prediction for every probe.
  - Test at least one competing hypothesis.
  - Use target data for diagnosis and selection only.
  - Keep failed attempts.
- **Data:** the paths and fields of §6.2.
- **Selection rule (§5.4)**, and the diagnosis schema: as in the dynamics adaptation instruction, plus
  `"reward_rationale": str`.

## 9. Permitted edits to existing code

| File | Edit |
|---|---|
| `integrations/friction_rl/network.py` | Add `held_to_release(trace)`. Add the four `initial_*` arrays to the trace, and `held_to_release` and `strict_success` to the record. |
| `orchestrator/friction_rl.py` | Factor `start_pool(run_dir, cfg, physics, mode, gpu)` (a context manager yielding `SimulatorPool`) out of `simulator_pool`, which becomes a thin wrapper. Both existing CLIs behave as before. |
| `orchestrator/launch_agent.py` | Add the `service_tier` keyword (§8.1). |

## 10. Configs

- **`tasks/FrictionAgent/agent.json`:**
  - `simulator_python`, `startup_timeout`, `step_timeout`, `seed`, `num_envs` 2, `cube_shift`;
  - `p0` (`policy`, `demos`);
  - `source_parameters`, `target_parameters`, `parameter_ranges`;
  - `eval_deterministic` 10, `eval_stochastic` 20, `required_successes` 27;
  - `budgets` (§5.3);
  - `training`: the §16.6 PPO keys with `network_continue.json` values, `init_std`, `episodes_per_iter` 8 and
    `warmup_iterations` 2;
  - `sandbox`: `image`, `timeout` 120, `memory_mb` 1024;
  - `agent`: `preset`, `model`, `reasoning_effort`, `service_tier`, `image`, `network` "filtered", `cpus` 4,
    `memory_mb` 8192.
  - `load_config` rejects missing or extra keys, wrong types, and `num_envs` > 2.
- **`tasks/FrictionAgent/smoke.json`:** the same, with `iterations_per_round` 4, `iterations_total` 4, `rounds` 1,
  `target_episodes` 4, `diagnostic_episodes` 1, `episodes_per_iter` 2, `eval_deterministic` 1, `eval_stochastic` 1
  and `wall_hours` 1.
- **`tasks/FrictionAgent/smoke_agent.py`** (test-only host script):
  1. check that training without reproduction is refused;
  2. probe N000 at the true target parameters on one target-failing development episode;
  3. train 4 iterations with `"default"` reward, wait, and submit.

  It makes no diagnosis claim.

## 11. Overseer procedure (GPU; not for the implementer)

1. CPU suites. Then a sandbox check: run the dry run against real Docker on the FT5 traces.
2. Smoke: `run --config smoke.json --scripted-agent smoke_agent.py`, then `evaluate`.
3. Oracle trainability at the true target parameters, with a researcher reward that is aware of the strict rule
   (`tasks/FrictionAgent/oracle_reward.py`, written by the overseer).
   - It runs one train job of 100 iterations, the agent's total budget.
   - It shows whether the frozen procedure can reach strict success at all.
   - Its results are never shown to the agent. If it fails, report to the user before the agent run.
   - **Result** (runs/202611041602_fa_oracle, 100 iterations, 2.85 h of training):
     - Strict successes per iteration of 8: 0 through iteration 51, first at 52, 7/8 at 58, 8/8 from 60 to 99.
     - Mean held force rose from 15 to 29.3 N; mean slip fell from about 7 mm at iteration 60 to 0.4 mm.
     - Iterations took about 60 s while episodes failed early and about 122 s once they ran to success.
     - Frozen evaluation at 0.25: strict 30/30, lenient 30/30; accepted.
4. Formal run with the model agent, then `evaluate` and `summarize`. Report strict and lenient results, and assess
   the diagnosis separately from performance.
   - **Result** (runs/202611041916_fa_run, gpt-6-astra high fast, budgets of §5.3). The agent submitted after 3.24 h.
     It used 2 rounds, 100 iterations, 8 target and 8 diagnostic episodes.
   - **Diagnosis:** three probes of N000 on D0/D2 — friction 0.3 (drops later and higher), mass 4 (never lifts) and
     friction 0.25 (zero-error trajectory match). Estimate: friction 0.25, mass 1, restitution 0, which equals the
     truth. Restitution and joint changes were left untested and reported as untested.
   - **Round 1:** reward `secure_grasp.py` (squeeze shaping saturating at 35 N, −8 on contact loss, +25 on any success),
     40 iterations. N001: target strict 0/4, lenient 4/4 (drops into the plate at about 22.6 N).
   - **Round 2:** reward `continuous_grasp.py` (squeeze shaping force/40 through the carry, −30 on contact loss, +80
     only for success held until release, −20 for lenient-only success), 60 iterations. Strict 8/8 per iteration from
     round iteration 20. N002: target strict 4/4, lenient 4/4.
   - **Frozen evaluation at 0.25, 30 episodes:** N002 strict 30/30, lenient 30/30; N000 strict 0/30, lenient 0/30;
     accepted.
     - N002 held force has a mean of 38.4–39.1 N and a minimum of 23.6–30.1 N. Slip is 0.3 mm.
     - This is well above the roughly 20 N needed. The reward paid for force up to 40 N with no cost for excess.
