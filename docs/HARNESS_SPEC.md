# DynamicsAdapt harness — engineering specification

Status: approved for implementation by milestones (2026-11-11). Research design and
rationale: [PLAN.md](../PLAN.md). Owner split: Claude writes specs, reviews each
milestone and runs GPU/container/model work; Codex implements milestones and runs
CPU tests only.

## 1. Implementation rules (normative)

- No defensive programming: no broad `try/except`, no fallback paths, no automatic
  retries, no speculative options or configurability beyond this spec.
- Validate once at boundaries (config load, request intake, sandbox/model-service
  output, checkpoint load) and trust internal calls afterwards.
- Content hashes identify frozen versions (environment builds, rewards, policies).
  Do not add re-verification layers, checksum ledgers or integrity audits beyond that.
- Explicit rejection instead of silent adaptation: incompatible policies, checkpoints,
  schemas or devices raise a clear error; never rebuild networks, drop optimizers or
  change encodings to make something fit.
- Reuse existing functions (`run_net_episode`, `Decoder`, `ppo_update`, `gae`,
  `Observer`, `build_scene`, `run_sandbox`, `stage_home`); do not copy them.
- One responsibility per module; keep modules near the size guidance in §11.
- Do not modify `vendor/`. Adapters go in `simulators/` or `integrations/`.
- Stop at each milestone boundary (§12) for review. CPU tests only; no simulator,
  Docker, GPU or model calls.

## 2. Architecture

| Layer | Responsibilities |
|---|---|
| Host | Prepare P0, freeze target and layouts, choose resources and protocol, host checks H1–H2, post hoc scoring |
| General harness (`orchestrator/`) | Agent sessions and lifecycle, request scheduling, environment/policy versions, training accounting, evidence export, selection, reports |
| Task adapter (`integrations/cube_to_plate/`) | Fixed task contract (scoring, termination, policy I/O semantics, evidence boundary); source-model integration (build/run/map Agent world models) |
| PPO backend (`integrations/ppo/`) | Load/continue policies, sample, update, checkpoint — task independent |
| Agent (container) | Hypotheses, environment candidates, analysis tools, rewards, requests |

Process and device placement:

| Process | Responsibilities | Boundary |
|---|---|---|
| Host orchestration (CPU) | Agent service, jobs, versions, accounting, evidence, reports | Sends only source specs to source jobs; never target config |
| Trusted GPU job | Batched physics, controller/IK, policy inference, rollout buffer, GAE, PPO, Adam | Never imports Agent programs |
| Reward / lightweight-mechanism sandboxes (CPU, Docker, no network) | Agent NumPy rewards and external-force programs | Numeric IPC in, rewards or forces/torques out |
| Isolated source-model service | Builds of Agent world models; Agent dynamics modules | No policy weights, target config, scorer internals or PPO data |
| Agent container (CPU) | Analysis and authoring | No GPU, weights, repository or private data |

The orchestration layer imports no cube-to-plate module, observation field, parameter
range or scoring function.

### 2.1 Contracts (`orchestrator/harness/contracts.py`)

- `TaskAdapter.describe()` → task contract, source-model modeling interfaces, backend
  capabilities (capabilities are not a list of admissible hypotheses).
- `TaskAdapter.prepare_source(bundle)` → versioned executable source model, or a
  specific `capability_unavailable` gap. Never falls back to the original model.
- `TaskAdapter.create_runtime(environment_ref, num_envs, render)` → batch runtime with
  `reset/step/close`, episode IDs, valid-env masks, termination states, execution errors.
- `PolicyAdapter.load(bundle)`, `PolicyAdapter.act(obs_batch, deterministic)`.
- `PPOBackend.train(train_spec)` → complete checkpoint and metrics.

Runtime data use separate types for (1) adapter-generated policy inputs (actor and
critic separately), (2) public sensors, (3) source training ground truth, (4) host
scoring. Learning data follow `obs_t → action_t → reward_t → obs_{t+1}`; public sensors
and video share the control-step index.

## 3. Host configuration

```text
task_adapter: cube_to_plate
initial_policy: {adapter: deployable, artifact, calibration, training_checkpoint?}
source: {bundle}
target: {private_spec: <path outside Git>}
layouts: {judgment_stream, final_stream, source_eval_stream, training_stream}
training:
  profile: cube_to_plate_ppo_100
  iterations_per_job: 100
  num_envs: 512
  episodes_per_env: 1
  minibatch: 8192
  device: cuda
  warmup_iterations: 2
  source_evaluation: {include_initial: true, every_iterations: 10, num_envs: 512, deterministic: true}
judgment: {layouts: 4, required_strict_successes: 4}
evaluation: {layouts: 30, pass_strict: 27}
agent: {preset: codex, model: gpt-6-astra, reasoning_effort: high, relaunch: {turn_continues: 3, transport_failures: 5, transport_wait_s: 300}}
runtime / sandbox
```

- Remove `p0.demos`, `bc_weight`, `parameter_ranges`, research `budgets` and the
  three-parameter diagnosis schema from the general configuration.
- Source training jobs receive only the source and training specification.
- Policy bundles declare actor/critic observation specs, action spec, preprocessing,
  architecture and weight format; adapters validate them.
- Restore optimizers, counters and random state only from a complete checkpoint of the
  selected policy version; with weights only, create optimizers and record that.
- Freeze protocol version, adapter version, simulator image ID, dependency versions,
  policy and environment identifiers. The Agent cannot override the device.

## 4. Agent workspace and requests

```text
workspace/
  instruction.md      task, contract and loop (no target factors, parameter lists or ranges)
  api/                read-only client and interface documentation
  source_reference/   allowlisted source model description and editable interfaces
  observations/       read-only public results
  environments/       Agent environment candidates
  rewards/  tools/  notes/ (incl. notes/PROGRESS.md)  submission/
```

Tools available: NumPy, frame-by-frame video reading, plotting, local scripts. No
observation-summary model and no robot control.

| Operation | Inputs | Outputs |
|---|---|---|
| `status` | — | Phase, running job progress, policy/environment versions, task counts, cumulative usage, curve references |
| `build` | candidate directory, hypothesis | `environment_id` or build error / `capability_unavailable` |
| `probe` | `policy_id`, `environment_id`, prediction, layout subset of L0–L3 | Source trajectories, source ground truth, video, outcomes |
| `deploy` | registered `policy_id`, purpose | 4 target judgment episodes (L0–L3), public feedback |
| `reproduce` | policy/environment versions, evidence references, comparison, uncertainty | Immutable `claim_id` |
| `train` | `environment_id`, `claim_id`, reward path, rationale (`environment_round_id` to continue) | Registered 100-iteration task `job_id`; automatic judgment after completion |
| `wait` | `job_id`, seconds | Job status and public artifact references |
| `submit` | diagnosis file | Archived diagnosis; experiment end |

- One experiment job at a time. Every request and rejection is archived.
- Side-effecting requests carry `request_id`. Same ID and content returns the original
  record; same ID with different content is rejected. A lost response never
  authorizes re-execution.
- `reproduce` checks only references, a completed source probe and target evidence for
  the same policy and environment versions — never hidden ground truth.
- The first `train` for an environment creates `environment_round_id` bound to the
  environment version, entry policy and claim. Later tasks in the round may continue
  from the latest complete checkpoint of that lineage without new probes. Other
  environments and unrelated policies cannot borrow the claim.
- After a judgment of 4/4, or a 4/4 P0 baseline, adaptation stops; a wrap-up session may
  only read evidence and `submit`.
- Diagnosis schema: conclusion; mechanism hypotheses; environment and claim references;
  evidence references; competing explanations with test status; uncertainty; reward
  rationale; optional preferred policy; optional parameter estimates.

## 5. Evidence and storage

Field allowlist: PLAN §3. Additional requirements:

- Public records state units, frames, sampling times and provenance; initial and
  per-control-step records map to video frames.
- Add a public finger tactile field aggregating all colliding objects in each pad's
  sensing region without identities, computed separately from the private
  cube-contact fields used by strict/native scoring. Name force semantics exactly
  (e.g., normal force, not a 6-axis wrench).
- No target failure subcategories (`dropped`, `no_lift`) in public outputs.
- Training records: episode/replica/policy IDs, `T+1` actor obs, critic obs, `T`
  pre-clip and executed actions, old log-probs, values, rewards, valid and termination
  masks, plus versioned source-model state, contact events, sensors and module outputs
  (with physics-substep records where a dynamics module runs).
- Chunked arrays per iteration with an index of committed ranges; never many small
  files; never discard intermediate iterations; partial writes stay as partial.
- Video for baselines, probes, judgments and final evaluation only.
- Capacity: one task ≤ 15.36 M transitions. Measure bytes per record, disk throughput
  and capacity with the real schema before formal runs.

## 6. PPO backend

- Remove BC loss, demonstrations and `bc_weight`. Remove the `(3, 4)` exploration
  restriction: all five dimensions are sampled and enter the joint log-probability.
- Store pre-clip samples; executed actions pass the existing clip and decoder.
- Continue the model, both Adam optimizers and `log_std` across tasks and environment
  rounds; never reset them.
- Minibatch 8,192; keep learning rates, 8 epochs, gamma, GAE lambda, clipping and KL
  settings. KL early stopping ends only the current iteration's updates.
- Critic-only warm-up for the first 2 iterations of the experiment's first task and of
  the first task after a reward-code change (counted within the 100).
- GAE along each replica's valid timeline; padding and post-termination steps excluded
  from advantage normalization, losses and statistics. Normal ends keep zero
  bootstrap (including horizon truncation). Execution failures are not `done`.
- Old log-probs and values come from the frozen sampling policy.
- Each iteration checkpoint: model, both optimizers, `log_std`, round/task IDs,
  counters, CPU/CUDA and sampling random states, reward/environment versions. The
  resumable pointer moves only after a complete task commits.
- Independent source evaluation (iterations 0, 10, …, 100) saves and restores the
  training random state and never touches the buffer, model or optimizers.
- float32 throughout; no mixed precision.

## 7. Source world models

A candidate environment directory may contain `environment.json` (base version, model
structure, state schema, entry points, resources), `world_model/` (Agent modules and
asset build descriptions), optional `mechanism.py` (external force/torque), and
supporting modules. `build` freezes all of it and produces an immutable version;
later requests cannot execute mutable code outside the snapshot.

Integration paths, implemented in this order:

| Path | What the Agent writes | Who advances the state | Milestone |
|---|---|---|---|
| W1 Declarative native | Physical parameters, materials, mass/CoM/inertia, added bodies, joints, geometry and contact structure as scene/asset overlays | PhysX (native, batched on GPU, no IPC) | M6a |
| W2 External force/torque | `reset(...) → state`, `step(body_state, dt, state, active_mask) → forces[N,B,3], torques[N,B,3], next_state` at every physics substep | PhysX integrates; sandbox computes forces | M6b |
| W3 Dynamics module | `describe()`, `initialize(...)`, `advance(model_state, applied_policy_commands, dt, active_mask) → next_model_state, physical_events` | Isolated source-model service, coupled at substeps | M6c |

Rules for all paths:

- `applied_policy_commands` are read-only outputs of the trusted decoder; modules
  cannot emit robot commands, scores, success flags or images. Trusted sensor/render
  code generates observations from actual state.
- Each state field has exactly one owner (PhysX or one module); mixed execution fixes
  substep order and coupling; no double integration or duplicated forces.
- Forces/torques are world-axis aligned at the center of mass. Updates run at every
  physics substep (10 per control step at dt 0.004), not only at control steps.
- Every reset initializes native and added state before settling; settling uses the
  candidate dynamics; nothing carries across episodes; each replica is independent of
  batch size and order; `active_mask=False` freezes native and added state.
- Builds report the actual state fields, DOFs, modules and backend in use, and the
  host inspects effective mass, materials, drives and bindings after construction.
  Missing capability returns `capability_unavailable` with the gap and keeps the
  candidate; it is not a falsified hypothesis.
- Task identities, roles, layout semantics and success conditions are fixed; source
  models map to the fixed task contract.

## 8. GPU batch backend

Goal: 512 replicas of one built environment in one process on one 4090, with policy
inference and PPO on the same GPU, launched through the existing simulator wrapper.
Training: `num_envs=512`, no cameras or rendering, simulated detection. Probes,
judgments and final evaluation: `num_envs=1`, rendering on, sequential episodes, same
physics backend. Release the training runtime before target judgment.

Remove from the current path: single-env assumptions (`env_0`, `[0]` indexing,
`episode_nums != 1` checks), per-step JSON round trips per environment, per-env Python
scoring and contact traversal, per-env IK and command construction in
`take_action_batch`, fixed CPU physics, and automatic multi-episode resets.

Preserve robot model, collision assets, control timing, physics settings and action
semantics. Compare IK targets, failure handling, joint drives, gripper force control
and tracking error against the single-env path with metrics and tolerances frozen
before the comparison. Use env-local coordinates, isolate collisions, and keep
termination, contact history, grasp counters and mechanism state per replica.
Verify Isaac Sim, Isaac Lab (vendored 2.3.2), CuRobo and PyTorch/CUDA versions inside
the actual image.

M0 measurements and their consequences are recorded in PLAN §11. They fix the design:

- Build the batched scene directly on Isaac Lab (`InteractiveScene` cloning with
  `replicate_physics=True`, articulation and rigid-object views, GPU PhysX at
  dt 0.004, 10 substeps per control step). Reuse RoboDojo assets and robot/scene
  configuration (`RobotManager._get_SceneCfg` works without the environment
  managers). Do not route through `EvalEnv`/`TaskEnv` managers: they assume CPU
  tensors (`env/reward_manager/func_parser.py:45` and ~200 similar conversions).
- Reproduce, in `simulators/cube_to_plate_batch.py`, the parts of the current task
  that matter: scene layout per replica, cube material and mass settings, the
  kinematic plate, gripper position/force drive switching, contact reporting per
  replica (cube–finger, cube–table, cube–plate), and the scoring of
  `simulators/dynamics_scoring.py` as batched tensor logic.
- IK: Isaac Lab's GPU differential IK (damped least squares) batched over all
  replicas, converting the decoder's end-effector command into joint position
  targets each control step (PLAN §13). Single-goal CuRobo calls (~11 ms each) in a
  Python loop are not acceptable. If the M3 parity gates fail, stop and consult the
  user; do not switch IK silently.
- The same backend serves N=1 rendered probes, judgments and final evaluation. The
  existing single-env CPU path remains only as the M3 parity reference and for the
  DP preparation workflow.

## 9. Rewards, accounting and failures

Reward contract: `reward(source_trace) → one finite scalar per valid control step`.
Inputs may include source ground truth. Keep size, finiteness, magnitude and resource
checks. Invoke in chunks of 64 episodes through a persistent isolated executor.
Rewards depend only on the trajectory and frozen program (order-independent). Update
PPO only after every reward of the iteration succeeds. Prechecks use this round's
probe trajectories or its completed training trajectories. Agent program exceptions
are model-code failures.

Accounting: register `job_id` after prechecks; persist the launch record before
starting the process; one registration = 100 planned iterations. Retain started tasks
in the denominator on failure. Record, per task and globally: planned/actual
iterations, completed/partial/unstarted episodes, transitions, evaluation and judgment
counts, elapsed and GPU time. No automatic retries or extensions.

| Situation | Resumable policy | Handling |
|---|---|---|
| Precheck rejected | Unchanged; no launch count | Keep rejection evidence |
| Launch failure or < 100 iterations | Previous complete task | Keep planned amount, partial checkpoints and curves |
| 100 iterations complete | New policy registered and resumable | Final source evaluation, then judgment |
| Source evaluation fails mid-task | Previous complete task | Task marked failed; no filled points |
| Final source evaluation fails after 100 iterations | New policy kept | Evaluation marked incomplete; judgment still runs |
| Judgment complete, not 4/4 | New policy kept | Eligible for ranking; Agent decides next step |
| Judgment interrupted | New policy kept | Planned episodes retained; explicit `deploy` allowed |
| Any complete standard judgment 4/4 | Frozen | Adaptation stops; wrap-up and final evaluation |

## 10. Selection and final evaluation

Select the first policy reaching 4/4. Without recovery, rank eligible policies by their
first complete standard judgment batch: strict, then lenient, then earlier policy.
Final evaluation: P0 and the selected policy, 30 deterministic episodes on F0–F29,
each from a new runtime, same seed table; the same policy is evaluated once. Results
are never returned to the Agent.

## 11. Code locations

| Location | Change (guide size) |
|---|---|
| `orchestrator/harness/` (new package, M1) | General layer: `contracts.py` (~200), `config.py`, `records.py` (versions, rounds/claims, accounting), `service.py` (requests, dedup, stopping), `jobs.py` (scheduling through adapters and the PPO backend), `diagnosis.py`, `experiment.py` (freezing, baselines, selection, final evaluation); no physics constants. Detailed design: §13 |
| `orchestrator/experiment.py`, `experiment_service.py`, `experiment_jobs.py` | Legacy FrictionAgent path; kept unchanged for its callers and removed in M7 once the harness replaces it |
| `orchestrator/evidence.py` (new) | Public/private export, curves, chunked storage (~300) |
| `orchestrator/agent_runtime.py` (new) | Lifecycle record, resume, relaunch policy, token accounting (~200) |
| `orchestrator/host_prep.py` (new) | H1–H2 checks, diagnosis scoring, reproduction fidelity (~200) |
| `orchestrator/program_sandbox.py` (new) | Shared isolation lifecycle for rewards, mechanisms, model service |
| `integrations/ppo/` (new) | GPU continuation, batch learning data, checkpoints, job entry |
| `integrations/world_model/` (new) | Candidate schemas, builds, W1–W3 execution and coupling |
| `integrations/cube_to_plate/` (new) | Task contract, decoder, DP observation/perception, asymmetric critic inputs |
| `simulators/cube_to_plate_batch.py` (new) | Batched replicas, batched control/IK, batched scoring |
| `simulators/dynamics_scoring.py` | Pure scoring logic shared by single-env and batch paths |
| `tasks/cube_to_plate/` (new) | Case config (no target values), workspace templates, default reward |
| `examples/host_preparation/cube_to_plate/` | Relocated P0/DP preparation (demos, BC, `bc_fit`, scripted controllers); moved in M7 together with the legacy removal (decision 2026-11-11) |
| `orchestrator/launch_agent.py` | New mounts; remove `nominal.json` and range exports |

Keep DP/P0 preparation commands working while shared modules move. Carry over the
single-env reset fixes, kinematic-plate render sync, room-reload cleanup, contact
cleanup and initial camera frame capture.

## 12. Milestones

Each milestone ends with a stop for review. "CPU" items are Codex unit tests; "GPU"
items are run by Claude.

**M0 — Feasibility spike (Claude, GPU). Done 2026-11-11.** Results in PLAN §11:
GPU PhysX is blocked inside the vendored environment managers; physics-only Isaac Lab
reaches ≈19,800 env-steps/s at N=512 in 9.7 GB; per-env CuRobo IK is the bottleneck;
P0 fails outright in the target on the current backend.

**M1 — Contracts and mock loop (CPU).** `contracts.py`, request service with dedup,
rounds/claims, task registration, accounting, stopping and failure table, evidence
types, diagnosis schema, end states. Tests: mock adapters with different observation
structures complete a full loop; no cube-to-plate import in the general layer; flat,
rising, falling and noisy synthetic curves never stop or extend a task; 51,200
rollouts and 5,632 evaluation episodes counted separately; dedup and interruption
cases; claims cannot be borrowed across environments or unrelated policies.

**M2 — Cube-to-plate adapter, DP import, evidence and leakage (CPU).** Extract the
task adapter; load DP through its adapter with explicit rejection of incompatible
bundles; public tactile field; evidence allowlist; workspace templates; host-private
target config outside Git (preparation-tool relocation moved to M7). Detailed design: §14. Tests: privileged-field
invariance of actor inputs; target inference uses only image detection and
proprioception; tactile aggregation ignores identity and does not change scoring
signals; no target values or failure subtypes in any exported file, log or error;
the workspace contains no repository files; DP preparation commands still work.

**M3 — GPU batch backend (GPU).** New Isaac Lab scene (§8), batched IK, batched
contacts and scoring. Split into M3a (physics, control, scoring; no rendering) and M3b
(N=1 rendering, camera, image detection, video); detailed design and the frozen gate
tolerances are in §15. CPU tests: batched scoring equals `dynamics_scoring` on
synthetic trajectories; per-replica reset and termination bookkeeping. GPU gates, with
metrics and tolerances frozen before running:
1. Controller parity: for identical joint states and end-effector commands, the
   differential-IK joint targets, unreachable-target handling and tracking error
   match the CuRobo single-env path within the frozen tolerance.
2. Task parity: P0 on DP dev stream 37 (deterministic) is compared with the CPU
   reference `runs/202611110150_probe_parity_cpu_source` (20/20 strict), using stop
   reasons, max clearance, slip, held force and release distance.
3. Target parity: P0 in the target on the same layouts is compared with
   `runs/*_probe_target_check_cpu` (0/20).
4. Replica independence and collision isolation (N=1 vs N=16 with identical
   layouts); 1 → 16 → 256 → 512 scaling.
5. Throughput at N=512 with contacts, IK, policy inference and evidence retention
   is at least 2,000 env-steps/s (PLAN §13); VRAM and host memory are recorded.

**M4 — PPO backend (CPU then GPU).** No BC, five-dimensional exploration, GPU
sampling/GAE/updates, 100-iteration tasks, source evaluation, curves, checkpoints.
CPU tests: masks, GAE boundaries, zero bootstrap, joint log-probs, continuation of
both optimizers and `log_std`, checkpoint/random-state consistency, evaluation
isolation, warm-up rules. GPU gate: one complete 100-iteration task with 512 replicas,
formal evidence retention, 11 source evaluations and the automatic judgment.

**M5 — Host checks and post hoc analysis.** H1–H2 checks on the new backend, diagnosis
scoring rubric and reproduction fidelity tools. GPU gate: P0 succeeds in the source
and fails in the target on L0–L3 before any Agent run.

**M6 — World-model integration.** M6a declarative native overlays (W1), M6b external
forces (W2), M6c dynamics-module service (W3). Engineering capability tests (not
formal targets): a drag law on a body without one; an added internal body/DOF; a
state-extension module with rotational state. Each must change motion and observable
behavior, restore baseline when removed, keep the task contract, and pass
single-env/512-replica, reset, inactive-mask and ownership checks.

**M7 — Agent runtime and end-to-end.** Lifecycle, resume and relaunch per PLAN §10;
workspace instructions; smoke experiment with a scripted Agent; then one supervised
experiment with Astra on the cube-to-plate case.

## 13. M1 detailed design (general layer, CPU)

Added 2026-11-11 for the M1 implementation. The general layer is the new package
`orchestrator/harness/`; it imports nothing from `integrations/`, `simulators/`,
`tasks/` or the legacy `orchestrator/experiment*.py`. Socket transport, the Agent
client, workspace templates and the evidence allowlist filter are later milestones
(M2/M7); M1 exposes `HarnessService.handle(request: dict) -> dict`.

### 13.1 Contracts (`contracts.py`)

Protocols (dependency injection; tests pass mock objects):

- `TaskAdapter`: `describe() -> dict`; `make_layouts(stream: str, seed: int, count: int) -> list`
  (opaque JSON layouts); `prepare_source(bundle_dir: Path) -> PreparedSource | CapabilityUnavailable`;
  `create_runtime(environment_ref, num_envs: int, render: bool) -> Runtime`.
  `environment_ref` is opaque; the initial source and the target are refs supplied by the
  host, built candidates are `PreparedSource.ref`.
- `Runtime`: `reset(layouts: list) -> StepBatch`; `step(actions) -> StepBatch`; `close()`.
  `StepBatch` (one per control step, batch axis first): `actor`, `critic` (adapter policy
  inputs, opaque arrays), `public` (dict of arrays), `source_truth` (dict of arrays, empty for
  the target), `done` (bool[N]), `execution_error` (str|None per env), and, for envs that
  ended this step, `outcome` per env: `{strict: bool, lenient: bool}` plus `host_scoring`
  (dict, host-private).
- `PolicyAdapter`: `load(bundle: PolicyBundle) -> Policy`; `Policy.act(actor_batch, deterministic: bool)`.
- `PPOBackend`: `start(spec: TrainSpec, directory: Path) -> TrainHandle`;
  `TrainHandle.progress() -> dict` (iterations done), `poll() -> TrainResult | None`,
  `stop()`. The backend runs the task (in M4 as a GPU subprocess) and reports failures as
  data in `TrainResult`; it never raises for a job failure.
- `RewardRunner`: `precheck(reward_path: Path, trace_refs: list) -> {ok, error?, summary?}`.
  Agent-code failures are returned as data (model failures), never raised.

Dataclasses: `PolicyBundle` (id, checkpoint ref, `complete_training_state: bool`),
`PreparedSource` (ref, manifest dict), `CapabilityUnavailable` (gap text),
`TrainSpec` (experiment seed, task number, round id, environment ref, entry policy bundle,
reward snapshot path, iterations = 100, warm-up iterations, num_envs, minibatch,
source-evaluation settings, training layout stream; **no target ref**), `TrainResult`
(status `complete|failed`, iterations_completed, checkpoint ref or None, partial checkpoint
refs, curve ref, trace ref of the last completed iteration, evaluation status
`complete|failed_mid_task|final_incomplete`, rollouts completed/partial, transitions,
evaluation episodes, elapsed_s, gpu_s, optimizer_state `restored|created`, error text),
`EpisodeRecord` (layout id, steps, public arrays, source truth, outcome, host scoring,
status `complete|execution_error|cancelled`), `EndState` enum
(`recovered`, `submitted_not_recovered`, `agent_exited`, `host_cancelled`).
Runtime data keep the four kinds of §2.1 in separate fields.

### 13.2 Configuration (`config.py`)

`load_host_config(dict) -> HostConfig` validates the §3 keys once (exact key sets, types,
`iterations_per_job == 100`, `num_envs`, `minibatch`, `warmup_iterations`, judgment 4/4,
evaluation 30/27, source evaluation every 10 with initial). Adapter objects, the initial
source ref and the private target ref are passed separately to the experiment, never read
from the Agent workspace.

### 13.3 Identifiers and records (`records.py`)

Policies `P0000` (P0), `P0001`…; environments `E0000` (initial source), `E0001`…;
jobs `J0001`…; claims `C0001`…; rounds `R0001`…. Host records live under
`<run>/host/` (JSON and JSONL via `orchestrator/files.py`); public results under
`<run>/workspace/observations/`. The **head** is the latest registered policy; the
lineage is linear.

- Environment: id, candidate snapshot directory, hypothesis, status `ready|capability_unavailable`,
  manifest or gap. `build` copies the candidate directory (workspace-relative, under
  `environments/`, regular files only, no symlinks, at most 64 MiB) into
  `host/environments/<id>/candidate/` and calls `prepare_source` on the snapshot.
- Claim (immutable): policy, environment, source probe jobs, target jobs, comparison,
  uncertainty. Intake checks only that every cited job exists and is complete, that at least
  one cited source job is a probe of that policy in that environment, and that at least one
  cited target job is a target batch (baseline, judgment or deploy) of that policy.
- Round: environment, entry policy, claim. A round is continuable iff every policy
  registered after its entry policy was produced by a task of that round.
- Train task: job record plus reward snapshot (copied into `host/rewards/<sha256>.py`,
  hash = reward version), warm-up flag, `TrainSpec`, `TrainResult`, judgment record.

### 13.4 Requests (`service.py`)

All side-effecting requests carry `request_id` (non-empty string). Exact field sets:

| op | fields |
|---|---|
| `status` | — |
| `build` | `request_id`, `candidate`, `hypothesis` |
| `probe` | `request_id`, `policy_id`, `environment_id`, `prediction`, `layouts` (nonempty unique subset of 0–3) |
| `deploy` | `request_id`, `policy_id`, `purpose` |
| `reproduce` | `request_id`, `policy_id`, `environment_id`, `source_jobs`, `target_jobs`, `comparison`, `uncertainty` |
| `train` | `request_id`, `environment_id`, `claim_id`, `reward`, `rationale`, `environment_round_id` (string or null) |
| `wait` | `job_id`, `seconds` (int 1–900) |
| `submit` | `request_id`, `diagnosis` (workspace-relative path under `submission/`) |

- Dedup: same `request_id` and identical request returns the archived response
  (including a rejection); same ID with different content is rejected. Every request,
  response and rejection is appended to `host/requests.jsonl`.
- One experiment job at a time (`probe`, `deploy`, `train` are jobs; `build` and
  `reproduce` are synchronous).
- `probe`: environment must be `ready`; policy must be registered. Mean actions, N=1,
  `render=True`, one episode per chosen layout, sequential.
- `deploy`: any registered policy; 4 target episodes on L0–L3, mean actions, N=1, render.
- `train`, new round (`environment_round_id` null): claim's environment equals
  `environment_id` and claim's policy is the head; creates the round on successful
  registration. Continuation: the round exists, is continuable, its environment equals
  `environment_id` and `claim_id` equals the round's claim. Reward: workspace-relative
  regular file under `rewards/`, at most 64 KiB, snapshotted before the precheck.
  Precheck traces: a new round uses the claim's cited probes; a continuation uses the trace
  ref of the round's latest completed task, or the claim's probes if none completed.
  Precheck rejection: archived, no job, no task count. Warm-up 2 iterations iff the head
  policy is P0 or the reward version differs from that of the task that produced the head.
  The entry bundle is always the head.
- `submit` is rejected while a job is running; the Agent waits for it first.
- After `recovered`, only `status`, `wait` and `submit` are accepted (wrap-up). After any
  other end state nothing is accepted except `status`.

### 13.5 Jobs (`jobs.py`)

Jobs run in one worker thread; `handle` never blocks except `wait`. A generic episode
loop runs a `Policy` in a `Runtime` (reset with one layout, step with mean actions until
`done`), producing `EpisodeRecord`s; it checks a stop event between steps. Probe and
baseline-source records export public arrays, source truth and outcome to the workspace;
target records export public arrays, strict/lenient and status only. Host scoring is
written only under `host/`. A train job: start the backend → poll to completion → apply
the §9 failure table → on a registered new policy run the automatic judgment (4 target
episodes) in the same job. Curves (by reference) and public training records go to the
workspace job directory.

A standard judgment batch is any complete 4-episode target batch on L0–L3 (P0 target
baseline, automatic judgment or `deploy`); an episode with an execution error counts as
not strict/not lenient and still completes the batch; a cancelled batch is interrupted.
Any complete standard batch with 4/4 strict ends the experiment as `recovered`.

### 13.6 Accounting and ending

Global and per-task counters: tasks registered/complete/failed, precheck rejections,
planned and actual iterations, training rollouts completed/partial/unstarted (planned =
100 × num_envs), transitions, source-evaluation episodes, probe episodes and transitions,
target episodes split into baseline/judgment/deploy, builds, claims, elapsed and GPU
seconds. Counters never gate anything. `HarnessService.end(state)` records `agent_exited`
or `host_cancelled` (cancel stops the running job and keeps partial records).

### 13.7 Experiment lifecycle (`experiment.py`) and diagnosis (`diagnosis.py`)

`freeze` writes the config, adapter description and the layout tables
(L0–L3, F0–F29, 512 source-evaluation layouts, training stream seed) under `host/`.
`baselines` runs P0 in the initial source and in the target on L0–L3 (target 4/4 →
`recovered` before any Agent request). `select_policy` and `final_evaluation` implement
§10 (each final episode from a new runtime; results only under `host/`).
Diagnosis schema (exact keys): `conclusion` (str), `mechanism_hypotheses` (nonempty list of
str), `environment_ids`, `claim_ids`, `evidence_jobs` (lists of existing IDs),
`competing_explanations` (list of `{explanation, test_jobs, status ∈ supported|rejected|untested, reason}`),
`uncertainty` (str), `reward_rationale` (str), `preferred_policy` (registered ID or null),
`parameter_estimates` (object of `{name: {value: number, unit: str}}`, may be empty).

## 14. M2 detailed design (cube-to-plate adapter, DP import, evidence, leakage; CPU)

Added 2026-11-11. M2 builds everything of the cube-to-plate adapter that does not need
the simulator. `create_runtime` and the initial-source preparation arrive with the batch
backend in M3; Agent-built sources (`prepare_source`) arrive in M6. Relocating the DP
preparation tools moves to M7 with the legacy removal. Legacy modules stay unchanged;
the adapter imports and reuses them (`Decoder`, `HORIZON`, `ACT_DIM`, `observe`,
`actor_observe`, `Observer`, `ActorCritic`, `build_scene`, `Criteria`/`describe_criteria`,
`reward`).

### 14.1 Package `integrations/cube_to_plate/`

| Module | Content |
|---|---|
| `contract.py` | `describe()` data: action decoder (five dimensions and their meaning, clip, pad floor, force mapping), 25 Hz control, 300-step horizon, `gripper_scale = 1.0`, strict/lenient definitions (text from `describe_criteria`), termination order, policy-input and evidence schema names. No physical source parameters and no target information |
| `layouts.py` | `make_layouts(stream, seed, count)`: entry `k` is `{"stream", "index", "scene_seed", "cube_offset", "plate_offset"}` with `scene_seed = SeedSequence([seed, crc32(stream), k])` and offsets exactly as `integrations/deployable/layout.build_scene` draws them for that seed (DP distribution: cube ±3 cm, plate box x ∈ [0, 3] cm, y ∈ [−3, 0] cm). `training_layout(seed, stream, task, iteration, replica)` gives the training layout for M3/M4. The four host streams must be distinct names |
| `policy.py` | Policy bundle (§14.2) and the DP `PolicyAdapter`: `load(bundle)` validates the manifest and loads `ActorCritic(21, 5, 54)` with `strict=True`; any mismatch raises a clear error (no rebuilding, no partial loads). `act(actor_batch, deterministic=True)` returns the actor mean (float32, batch axis first, shape `[B, 5]`); `deterministic=False` is not used by the host |
| `inputs.py` | Policy inputs per control step: actor = `actor_observe(...)`, critic = actor ⊕ `observe(...)` (54). Detection mode is fixed by run kind: rendered N=1 runs (baselines, probes, judgments, deploys, final evaluation) use image detection with the frozen calibration; non-rendered training and source evaluation use simulated detection |
| `evidence.py` | Field allowlist (§14.3), `public_step(...)`, `source_truth_step(...)`, `public_outcome(...)` builders. Only these builders produce exported records |
| `tactile.py` | `pad_normal_forces(net_force_w[N, 2, 3], pad_normal_w[N, 2, 3]) -> [N, 2]` = `max(0, −F·n)` per pad (N): the unfiltered net contact force on each finger pad from all partners, projected on the pad's inward normal. Pure NumPy/torch-compatible array code; the M3 backend feeds it from unfiltered finger contact sensors, separate from the filtered cube–finger contacts used for scoring |

### 14.2 Policy bundle

A bundle directory holds `policy_bundle.json` and the files it names:
`{"adapter": "deployable", "actor_obs": {"dim": 21, "layout": "dp_actor_v1"},
"critic_obs": {"dim": 54, "layout": "dp_critic_v1"}, "action": {"dim": 5, "decoder": "cube_to_plate_v1"},
"architecture": {"hidden": [256, 256], "activation": "tanh", "log_std": true},
"weights": "policy.pt", "training_state": null | "training.pt", "calibration": "calibration.json"}`.
Exact keys and values are validated at load. Host tool
`python -m integrations.cube_to_plate.make_bundle --policy <pt> --calibration <json> --out <dir>`
copies P0 (`runs/202611101817_dp_a2_train/policy_it000.pt`, calibration3) into a bundle
with `training_state: null` (weights only; M4 then creates optimizers and records that).

### 14.3 Evidence schema (PLAN §3)

Per control step `t` (0 = reset state, then one row after each executed action), units SI:

- Public (source and target): `time_s`, `joint_position` and `joint_velocity` (right arm,
  rad, rad/s), `gripper_opening` (0–1), `executed_action` (5, after clip), `commanded_ee_pose`
  (7, world), `grip_force_command` (N), `ee_pose` (7, from proprioception), `tactile_normal_force`
  (2 pads, N), `video_frame_index`. Outcome: `strict`, `lenient`, `execution_status`
  (`complete` or `execution_error`).
- Source truth (source runs only): cube position/quaternion/linear and angular velocity, plate
  position/orientation, contact flags (cube–table, cube–plate, cube–robot, cube–finger per pad),
  `stop_reason`, slip, lift height and release distance.
- Never public: object poses in target runs, true contacts, `stop_reason` and failure subtypes
  (`dropped`, `no_lift`, `timeout`, `native_end`) in target runs, policy inputs, detection
  truth/error, host scoring, physical parameters of the target.

The Agent default reward (`rewards/default_reward.py` in the workspace) is the DP training
reward rewritten against this trace schema (same terms and constants as
`integrations/deployable/reward.py`).

### 14.4 Host-private target and workspace

- `/local/` is added to `.gitignore`. `load_target_spec(path)` accepts only a path outside the
  repository or under `local/`, validates the exact schema once
  (`{"kind": "cube_to_plate_target_v1", "cube_material": {"static_friction", "dynamic_friction",
  "restitution"}, "cube_mass_kg"}`; numbers, finite, positive except restitution ∈ [0, 1]) and returns
  an opaque target ref. Errors never echo values. An example with placeholder values that are
  not the formal target lives at `tasks/cube_to_plate/target.example.json`.
- General layer `orchestrator/harness/workspace.py`: `prepare_workspace(workspace, template_dir,
  source_reference_files)` creates `instruction.md`, `api/`, `source_reference/`,
  `observations/`, `environments/`, `rewards/`, `tools/`, `notes/PROGRESS.md`, `submission/`
  from an adapter-supplied template directory and an explicit allowlist of files. Nothing else
  is copied; no symlinks into the repository.
- `tasks/cube_to_plate/workspace/`: `instruction.md` (task, contract, request loop, information
  available, PROGRESS.md duty; says only that P0 behaves differently in the source and the
  target; names no physical factor, parameter list or range), `api/README.md` (request
  fields of §13.4 and the §14.3 schema), `rewards/default_reward.py`. The
  `source_reference/` allowlist is a neutral description of the full source model
  (`tasks/cube_to_plate/source_reference/*.md`, written in M2: bodies, joints, actuators and drives,
  geometry, every material, masses, solver and timing, sensors, with nominal values), not a list of
  suspect parameters.

## 15. M3 detailed design and frozen gates

Added and frozen 2026-11-11, before any M3 code or GPU run. Tolerances below may not be
changed after the first gate run; a failed gate is reported to the user (PLAN §13).

### 15.1 M3a scope (physics, control, scoring; no rendering)

`simulators/cube_to_plate_batch.py` (Isaac Lab scene and runtime) plus pure tensor modules
in `simulators/cube_to_plate_batch_*.py` or `integrations/cube_to_plate/` (decoder, IK,
interpolation, scoring) that are CPU-testable. The runtime runs in the simulator image
(`.venv-robodojo/bin/python`, launched with `orchestrator.resources.sim_env`) and
implements the M1 `Runtime` protocol for `num_envs = N`, `render=False`:

- Scene: `InteractiveScene` with `replicate_physics=True`, RoboDojo `dual_x5` robot config via
  `RobotManager._get_SceneCfg`, the static table, the kinematic plate and the dynamic cube with the
  geometry, assets, masses, materials and combine modes the legacy task actually uses (read them
  from `tasks/DynamicsAdapt/scene.json`, `simulators/dynamics_native.py` and RoboDojo; the build
  reports the effective values per §7). GPU PhysX, dt 0.004, 10 substeps per control step,
  solver settings as in the legacy scene. Per-replica env-local coordinates.
- Control: the batched `Decoder` (same arithmetic as `integrations/friction_rl/network.Decoder`)
  produces the commanded EE pose, closed flag and force. IK: batched damped-least-squares IK on a
  torch forward-kinematics/Jacobian model of the right X5 arm parsed from the URDF the scene loads,
  warm-started from the current joint positions and iterated to the CuRobo success thresholds used
  by the legacy solver (fixed maximum iteration count). On IK failure the arm keeps its previous
  joint target (legacy semantics). Arm joint targets then follow the legacy
  `process_control_info` profile (linear interpolation over the first 8 substeps, hold for 2).
  Gripper: legacy position mapping (`gripper_scale` range and mimic) when open; when closed, legacy
  force mode (stiffness 0, effort target −force on the gripper joints); nominal drives restored on
  reset. The idle left arm holds its default pose.
- Reset: per-replica layout (`integrations/cube_to_plate/layouts.py` entries), robot default joint
  state, settle with the candidate dynamics as the legacy reset does, contact history cleared;
  `reset` of a subset of replicas leaves the others untouched.
- Contacts and scoring: per-replica contact sensing for cube–finger (each pad), cube–table,
  cube–plate and cube–robot (filtered), and unfiltered net finger-pad forces for the tactile field
  (§14.1). Batched scoring reproduces `simulators/dynamics_scoring.PlacementScorer` and the
  `run_net_episode` stop logic (success by rest count or native success, `native_end`, `dropped`,
  `no_lift`, `timeout`, in that order) and `held_to_release` / strict.
- Execution errors: non-finite state or PhysX failure in a replica marks that replica's
  `execution_error`; it is never a `done` transition.
- Evidence per §14.3 for every replica and step, held on GPU and copied to host memory per episode
  chunk (storage format is M4's).

CPU tests (Codex): batched decoder equals the legacy `Decoder` on random action sequences; FK and
DLS IK converge on random reachable poses and report failure on unreachable ones; interpolation
profile equals the legacy one; batched scoring and stop logic equal `PlacementScorer` +
`run_net_episode` on synthetic sample sequences (including drop, no lift, native end, timeout,
interrupted hold); per-replica reset/termination bookkeeping; tactile wiring.

### 15.2 M3a GPU gates (Claude), frozen tolerances

References: CPU source `runs/202611110150_probe_parity_cpu_source` (P0, DP dev stream 37,
20 layouts, deterministic, simulated detection: 20/20 strict; means max clearance 50.4 mm, slip
0.27 mm, held force 15.0 N, release distance 4.8 mm) and CPU target `runs/*_probe_target_check_cpu`
(0/20 strict, 0/20 lenient; 18 dropped, 2 no lift).

1. Controller parity. In one process, for every (joint state, EE command) pair visited by gate 2's
   20 episodes, call both the batched DLS IK and the legacy single-goal CuRobo call. Pass: success
   agreement ≥ 99.5 %; where both succeed, p99 |Δq| ≤ 0.01 rad and p99 FK error of the DLS solution
   ≤ 1 mm and ≤ 0.01 rad. Commands outside the decoder workspace fail in both.
2. Task parity (source). P0 on the 20 stream-37 layouts at N=20, deterministic, simulated
   detection with the same detection noise draws as the reference. Pass: strict ≥ 19/20 and
   lenient ≥ 19/20; stop reason equal to the reference in ≥ 19/20 episodes; episode length within
   ±10 steps of the reference in ≥ 18/20; means: max clearance 50.4 ± 3 mm, slip ≤ 1.0 mm, held force
   15.0 ± 1.0 N, release distance ≤ 9.8 mm.
3. Target parity. Same layouts in the target. Pass: strict ≤ 2/20, lenient ≤ 2/20, dropped + no lift
   ≥ 18/20.
4. Replica independence and isolation. The 20 gate-2 layouts run at N=1 (one at a time) and as
   replicas 0–19 of an N=512 batch whose other 492 replicas use source-evaluation layouts: identical strict/lenient outcomes and stop
   reasons, episode length within ±2 steps, final cube position within 2 mm. No contact between
   bodies of different replicas at any step. Scaling 1 → 16 → 256 → 512 recorded (throughput, VRAM,
   host memory).
5. Throughput. N=512, P0 mean-action inference on GPU, contacts, IK, scoring and evidence copied
   to host memory: ≥ 2,000 env-steps/s averaged over one full 300-step iteration (PLAN §13).

Also recorded (not gated): P0 strict rate on the 512 source-evaluation layouts at N=512.

Outcome 2026-11-11: gates 1 and 2 passed; gate 3 failed (14/20 strict vs ≤ 2/20). User decision:
accept the backend for source and target alike and redo H2 on it (PLAN §13). Gate 3 is retained as a
failed, disclosed gate and is not rerun with changed tolerances.

### 15.3 M3b scope and gate (N=1 rendering)

Rendering at N=1: the legacy head camera (`cam_head`, 640×480, pose and intrinsics of the legacy
scene), the room, lighting, background and table/cube/plate visuals the legacy renders use, the
kinematic-plate render pose fix, raw video per camera at 25 Hz, and the reset frame for image
detection. The detector is recalibrated on the new backend with the DP calibration procedure
(streams 20/21) because the renderer is new. Gates: calibration cube p95 ≤ 2 mm and plate p95 ≤ 2 mm;
P0 with image detection on DP render-eval stream 39 (20 layouts, source): strict ≥ 18/20.
