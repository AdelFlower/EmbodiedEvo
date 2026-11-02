# RL Policy Adaptation Through Source-Simulator Calibration — Engineering Specification

Experiment identifier: `dynamics-adaptation-v1` · Specification date: 2026-10-31

This document turns the research plan in [`new_project_plan.md`](../new_project_plan.md)
into an implementable design. If the two documents conflict, this one wins. §2 lists every
difference and gives its reason.

**Roles.** Codex implements one milestone at a time (§12) and runs CPU tests only. The
overseer (Claude Code) reviews each milestone, runs the GPU checks, launches containers and
model runs, and drives the experiment. Codex never starts GPU jobs, Docker containers or
model calls.

---

## 1. Summary

**Question.** Can an agent identify a dynamics difference from a neural policy's mismatch
in a fixed target, calibrate the source simulator, and restore target success through
continued RL in that calibrated source?

**Loop.**

1. P0 is an existing SAC actor trained in the nominal source. At freeze it is converted,
   without changing its deterministic output, into the PPO learner of revision 1 (§2.14).
2. The host runs a shared baseline: P0 on five paired development layouts in both the source
   and the target.
3. The agent forms hypotheses, probes the calibrated source with frozen networks, and chooses
   source parameters. Before every training round it must reproduce, in the calibrated
   source, the target failures of the network it will fine-tune (§2.15).
4. The host continues PPO, with an action-space trust region, from the previous round in
   that calibrated source.
5. The host deploys the new network on the same five target layouts.
6. Steps 3–5 repeat for at most 4 rounds.
7. The host selects a candidate by a fixed rule.
8. A host-scripted control repeats the same schedule at nominal parameters.
9. Frozen 50-layout target evaluation of three networks: P0, the calibration-selected
   network and the control-selected network.

---

## 2. Design review: changes and clarifications relative to the plan

**2.1 P0 is fixed and already qualifies.** No new source evaluation is needed.

| Item | Value |
|---|---|
| Inference weights | `runs/202610302107_dynamics_rl_train_v4/checkpoints/best_rl.pt` |
| Full training state | `runs/202610302107_dynamics_rl_train_v4/checkpoints/best_rl.zip` (SB3 `SACFromDemonstrations`: actor, critic, target critic, optimizers, counters) |
| Counters | `online_steps` 678, `rl_updates` 677 |
| Source evaluation | `runs/202610302128_dynamics_rl_eval`: 50/50, 0 incomplete; its `frozen-policy.pt` is byte-identical to `best_rl.pt` |
| Recipe | robust BC (13 effective actor inputs), then 677 SAC updates at actor learning rate `1e-6` |

**2.2 Disclose P0's prior knowledge and how the target was chosen.** P0 learned its behavior
from demonstrations by the R1 teacher. R1 was written by the host for friction `0.3` in the
DynamicsRepair experiment. Observed results:

- At `0.3`, P0 succeeded on 5/5 layouts (`runs/202610312040_dynamics_rl_transfer_friction_0_3`).
- At `0.25`, P0 failed on its single layout (`runs/202610312058_dynamics_rl_friction_sweep`).
  The grasp lasted control steps 109–178, the cube rose to z = 0.836 and was lost in
  transport.

Researcher ruling (2026-11-01): P0 is used as is. Its demonstrations and online SAC training
ran in the source at friction `1.0`; only the teacher's design targeted `0.3`.

The target `0.25` was chosen after these observations. That is legitimate, because the
target is fixed before formal adaptation, but the report must state it. It also explains why
the transfer gap is narrow.

**2.3 Neither the size of the gap nor whether the target can be recovered has been
established.** The only evidence is one layout. Phase 0 (§11) therefore adds two steps
before any formal run:

- a 10-layout screening of P0 in the target;
- the trainability gate, with a pass criterion and stop rule stated in advance.

**2.4 Adaptation training is a new procedure and needs its own frozen hyperparameters.**

- P0's recipe was tuned to keep the actor close to the demonstrations: actor learning rate
  `1e-6`, log-std ≤ −6 (about 0.6 mm of position noise), BC auxiliary weight 10⁴→10³ and 25%
  demonstration replay.
- The plan removes BC and demonstrations. What remains is SAC with a very slow actor and
  almost no exploration. Nobody knows whether that can change the policy within 100k actions.
- Tuning these values against the trainability check would tune the procedure on the true
  target parameters.
- Recommendation (decision D1, §14): develop the values on a development shift that is not
  the target, freeze them, then run trainability as a pure pass/fail gate. Any change after
  the gate is a recorded protocol revision.
- Outcome: the smoke run confirmed the concern in its strongest form (SAC collapsed P0 even
  at nominal parameters), which led to revision 1 (§2.14).

**2.5 Selection is applied by the host, identically in every branch.** The plan lets the agent
"submit the final network ID" and also fixes a selection rule. The rule decides, so the
control stays matched:

- the candidate with the most target development successes wins;
- ties go to the earlier candidate;
- P0 is a candidate.

The agent's `preferred_network` is recorded and reported but not used.

**2.6 Each round is a training run followed by an automatic target deployment.** With 4
rounds and 20 target episodes, every round is deployed exactly once. A separate `deploy`
operation would give the agent no real choice, only extra failure modes. The client
operations are therefore: `status`, `probe`, `train`, `wait`, `submit`.

**2.7 Long operations run as asynchronous jobs.** A 25k-action round takes about 40 minutes
(about 10.4 actions/s with one SAC update per action, measured in run `202610302039`).

- Operations return a job ID.
- `wait` blocks for at most 15 minutes.
- One job runs at a time.

**2.8 (Superseded by §2.14.)** The SAC replay warm-up no longer applies. PPO is on-policy:
every update uses only the current rollout, so each round's data comes only from that round,
which satisfies the plan's replay-clearing rule.

**2.9 Friction and mass are confounded during the grasp.**

- The gripper is position-controlled, so its normal force N does not depend on cube mass.
- Holding the cube requires μ·2N ≥ m(g+a). Friction ×0.25 and mass ×4 therefore give nearly
  the same grasp slip, and mass ×4.0 is the edge of the allowed range.
- The two differ in how the cube interacts with the table and plate: sliding when pushed, and
  settling on the plate.
- The diagnosis review (§10.3) must check whether the agent told them apart. Target success
  alone cannot show this.

**2.10 Agent information is a whitelist (§8.2).**

- The agent receives: actions, robot and object state, raw videos, outcomes, and the
  success-criteria text.
- It does **not** receive: contact flags, scorer state, shaped rewards, solver velocities,
  live material or mass records, or any target parameter.
- Contact flags are excluded because they are not in the plan's list, and the sim-to-real
  analogy has no tactile sensing (D4).

**2.11 The control is host-scripted and matched beyond the counts.** It uses the same
per-round action counts, per-round learner seeds, online layout streams, deployment layouts
and selection rule. It runs after the calibration branch. It has no diagnostic probes and no
agent time; the report states that difference.

**2.12 In sim-to-sim, the target can be identified exactly.** A probe whose parameters equal
the target's reproduces the target trajectory on the same layout; preflight P3 verifies this.
Calibration is therefore a well-posed search here, and that is no evidence about calibrating
to a real robot. The report must state this limitation.

**2.13 Optional oracle reference arm (D3).** The trainability chain trains at the true target
parameters, so it is an upper bound. Evaluate its selected network on the formal 50 layouts
after both branches are frozen, labelled `oracle`. It is excluded from the agent's budget and
from the agent-vs-control comparison.

**2.14 Revision 1 (2026-11-01): fine-tuning switches from SAC to PPO with an action-space
trust region.**

Evidence, from the scripted smoke run `runs/202611011301_adaptation_smoke`, which used nominal
and friction-0.5 sources only (no target information):

- In both branches, the two online episodes before the first gradient update succeeded.
  Every episode after updates began (transition 500) failed, including at nominal
  parameters, where P0 scores 50/50.
- After about 1,500 SAC updates at actor learning rate `1e-6`, no actor weight had moved
  more than 0.004, but the deterministic action had moved by about 0.4 (normalized units)
  on average.
- A random ±0.004 perturbation of P0's weights moves actions by 0.18 on average. The
  robust-BC network is extremely sensitive to its weights (first-layer clock weight ≈ 200),
  and Adam moves every weight by about the learning rate per step. Learning-rate tuning
  cannot fix this: a rate small enough to keep P0 intact is too small to adapt.

Researcher decision: continue training with PPO instead of SAC; the overseer specifies the
safeguards against collapse. They act on the **action distribution**, not on the weights:

1. **Exact transplant.** P0's actor becomes the PPO policy network. The Gaussian mean is
   `tanh(action_net(latent))`, so the deterministic PPO action equals P0's action. The
   robust gradient mask stays active.
2. **Fixed exploration.** A state-independent `exploration_log_std` that is not learned, with
   no entropy term. This rules out entropy-driven collapse or explosion of the noise.
3. **Value warm-up.** The value network is new at conversion. The first rollout of every
   round updates only the value network. Later rollouts use the warmed value function for
   advantages.
4. **Per-update trust region with backtracking.** SB3's `target_kl` early stopping, then a
   hard guarantee. After each policy update, compute the exact KL between the pre- and
   post-update policies on that rollout's observations:
   `mean_s Σ_d (μ_new − μ_old)² / (2σ²)`. If it exceeds `max_update_kl`, scale the policy
   step back (θ = θ_old + α·Δθ, halving α up to 10 times; α = 0 if still too large) until it
   fits. The value network keeps its full update.
5. **Bounded drift per round as a consequence.** Each update moves the mean by at most
   about `σ·√(2·max_update_kl)` per state. A 25k-action round therefore stays within a few σ
   of its parent, much smaller than the 0.4 collapse observed under SAC. Drift from the
   parent is recorded after every update as evidence.
6. **Selection still protects the result.** P0 and every earlier network remain candidates
   (§2.5), so a degraded network cannot displace a better earlier one.

Preflight P5 (§10.2) checks the safeguards empirically: a full round at nominal parameters
must not degrade P0. The report discloses this revision and its evidence; the plan's "SAC
updates" wording is superseded. Every other plan rule is unchanged: no teacher, no
demonstrations, no BC term, target data never used in training, fixed architecture, and
per-round data only.

**2.15 Revision 1: reproduction before fine-tuning (researcher decision, 2026-11-01).** The
agent must reproduce the target failure in the calibrated source before it may train there.
The service enforces this (§7.6):

- A probe job **reproduces** network N at parameters θ when:
  - all of its episodes completed;
  - on every probed layout, N's success in Sθ equals N's recorded target-development outcome
    on that layout (for N000, the shared target baseline; for N00k, its deployment; an
    incomplete target episode counts as a failure);
  - at least one probed layout is a target failure of N.
- `train` at parameters θ is accepted only if such a probe exists for θ (exact values after
  filling in nominal values) and for the parent network the round will continue from.
- If the parent has no target failure, or no deployment record, `train` is refused: there is
  nothing to reproduce.

The host checks only success or failure, which the agent already sees in both exports. The
agent must still compare the failure **mode** (for example, where and when the cube is lost)
in its hypothesis, prediction and diagnosis. The diagnosis review (§10.3) judges that. In
sim-to-sim, deterministic rollouts make a reproduction repeatable (P3).

---

## 3. Engineering rules for the implementer (normative)

These rules exist because earlier code in this repository grew a lot of defensive and
verification code that hid the logic. Examples: `orchestrator/dynamics_rl.py`'s
cleanup-error ladders, and the sha256 "delivery verification" files under `runs/`.
**New code must not follow that style.** Reuse existing functions as they are and do not add
more of that pattern.

1. **No integrity machinery.** No hashes, checksums, signatures, manifests of manifests,
   "verification" JSON files, or re-reading a file just written to confirm it. Record paths,
   copy files, and use sequential IDs (`J0001`, `N001`) and `time.time()` timestamps.
2. **Validate only at trust boundaries.** There are three:
   - agent requests in the service (§7.6);
   - loading the configuration file (§9);
   - CLI arguments.

   Values produced by our own code are trusted. Do not re-check them with `isinstance`,
   `isfinite`, shape or range checks. Do not repeat a check across client, service, job and
   worker. The client only uses argparse.
3. **Let errors propagate.** Catch an exception only where a record must be written: the
   episode boundary and the job boundary. Write the record once (type, message, traceback),
   then stop or re-raise.
   - No `except BaseException` except to write that record and re-raise.
   - No collected "cleanup error" lists and no nested `try/finally` chains.
   - No catch-and-continue, no fallback values, no silent defaults.
4. **No retries, no automatic reruns, no automatic resume.** A failed episode or job stays
   failed and stays in the record.
5. **No speculative configurability.** Use only the configuration fields in §9 and the CLI
   flags in §7.9. Add no mode switches, feature flags, unused parameters or
   "for future use" hooks.
6. **No compatibility layers.** `dynamics-adaptation-v1` is written once, in `run.json`. Do not
   negotiate versions or support old formats.
7. **Confine edits.** Change existing files only where §7 lists them. Do not refactor,
   reformat or rename existing code.
   - Do not touch `vendor/`, `tasks/RoboDojo/`, `tasks_baseline/`, `workspace-session-v1`, the
     four primitives, or DynamicsRepair code paths, except the shared files named in §7.1.
   - Update existing tests that assert behavior this spec deliberately removes. Do not work
     around them.
8. **Keep it short.** One-line docstrings. Comments only for non-obvious reasons. The size
   guidance below is for new non-test code. Exceeding it needs a sentence of justification in
   the milestone report.

   | Module | Guidance |
   |---|---|
   | §7.1 diff | ≈ 60 lines |
   | `integrations/dynamics_rl/adaptation.py` | ≤ 250 lines (revision 1: PPO) |
   | `orchestrator/adaptation_jobs.py` | ≤ 300 lines |
   | `orchestrator/adaptation_export.py` | ≤ 120 lines |
   | `orchestrator/adaptation_service.py` | ≤ 300 lines |
   | `scripts/adapt.py` | ≤ 100 lines |
   | `orchestrator/dynamics_adaptation.py` | ≤ 720 lines (683 after M5) |
   | **Total** | **≈ 1,500 lines** |

9. **Tests check required behavior.** Write about one test per requirement in §10.1.
   - Mock only the simulator process and Docker.
   - Do not assert on error-message text.
   - Do not test argparse wiring, trivial getters, or that a file you wrote exists.
10. **Add no new dependencies.** The host learner uses `.venv-rl`. The agent image stays
    unchanged (it already has numpy, pillow, imageio and ffmpeg).
11. **Never claim unrun evidence.** CPU tests do not establish GPU physics, trainability or
    agent behavior. Each milestone report lists the commands actually run and their output.
12. **Reuse before writing:**
    - `run_episode`, `simulator`, `save_checkpoint` from `orchestrator/dynamics_rl.py`;
    - `SimulatorClient`, `RemoteEnv`;
    - `scene_plan`, `layout_seeds`;
    - `load_model`, `export_policy`, `load_policy`;
    - `launch_agent.{require_image, ensure_network, stage_home, agent_argv, adapt_for_docker, docker_argv}`;
    - `write_json`, `append_jsonl`, `run_lock`, `codex_token_usage`;
    - `gpu_lease`, `stop_process`;
    - `describe_criteria`, `Criteria`;
    - `nominal_parameters` from `orchestrator/dynamics_repair.py`.

---

## 4. Fixed experimental settings

| Item | Value |
|---|---|
| Task | `red_cube_to_plate` with the DynamicsRL observation/action contract (`features.contract()`), 25 Hz, 500 actions per episode, scoring `dynamics-repair-score-v4` |
| Layouts | cube and plate XY drawn independently and uniformly within ±3 cm (`position_radius` 0.03) |
| Policy | P0 (§2.1): 2×256 MLP, 57-dim encoded observation (13 effective inputs under the robust gradient mask), 7-dim action |
| Fine-tuning | PPO with fixed exploration noise, per-round value warm-up and a backtracking KL trust region (§2.14, §7.3); value network 2×256, new at conversion |
| Source S0 | `friction_scale` 1.0, `mass_scale` 1.0, `restitution` 0.0. Cube 1.0/1.0 static/dynamic friction, 0.35 kg. Effective contact friction 0.5 (`multiply` against 0.5 surfaces). |
| Target T (host-only) | `friction_scale` 0.25, others nominal. Cube 0.25/0.25; effective contact friction 0.125. |
| Editable in Sθ | `friction_scale` [0.05, 2.0], `mass_scale` [0.25, 4.0], `restitution` [0, 1] |
| Budget per branch | 8 h wall time; 4 rounds; ≤25,000 actions per round; ≤100,000 actions in total; 20 target development episodes (= 4 deployments × 5); 20 diagnostic episodes |
| Development layouts | 5, split `adaptation_dev`, shared by baseline, probes, deployments and both branches |
| Final evaluation | 50 layouts, split `adaptation_eval`, fresh simulator per episode, rendered |
| Acceptance | calibration-selected network ≥ 45/50 native successes |

Parameter changes take effect only when a simulator process starts. One job uses one
parameter set; parameters never change within an episode or a job.

---

## 5. Architecture

```
host: .venv-rl/bin/python -m orchestrator.dynamics_adaptation <command> ...
 ├─ runner (main thread): stages, selection, control, evaluation, summary
 ├─ service thread (calibration stage only): Unix socket <-> workspace client
 │    └─ job subprocess: python -m orchestrator.adaptation_jobs <job-dir>
 │         ├─ SB3 learner on CPU (train jobs)
 │         └─ SimulatorClient -> existing orchestrator.dynamics_rl_worker (GPU)
 └─ agent container (existing image, Docker): workspace + socket dir; no GPU, no simulator
```

- Jobs are subprocesses started with `start_new_session=True`. Stopping one at the deadline is
  a process-group kill, and a crash in a job cannot take down the service.
- The baseline, the control, trainability and evaluation use the same job module, called
  synchronously by the runner.
- The container mounts only `<run>/calibration/workspace` and the socket directory.
  Everything else under `<run>` is host-only, including `config.json` with the target
  parameters.

---

## 6. Run directory layout

```
runs/<YYYYMMDDHHMM>_adaptation/
  run.json             experiment, command, started_at, agent_kind (model | scripted)
  config.json          resolved configuration, including target parameters (host-only)
  layouts.json         development layouts [{index, layout_seed, scene}]
  source/              copied source files (existing new_run convention)
  networks/N000/       policy.pt (P0, copied), sac-training.zip (P0's SAC state, copied),
                       training.zip (PPO learner converted from P0 at freeze), network.json
  baseline/source/     rollout job: P0, S0, 5 development layouts, rendered
  baseline/target/     rollout job: P0, T, 5 development layouts, rendered
  calibration/
    workspace/         agent-visible tree (§8.1)
    requests.jsonl     every agent request and response, including refusals
    jobs/J0001/        job.json (request + resolved spec), result.json, sub-job dirs
    networks/N001/     policy.pt, training.zip, network.json, checkpoints/
    selection.json
    result.json        end reason, budgets used, timings, model usage
    agent.log, agent_sessions/
  control/
    rounds/R1/train/, rounds/R1/deploy/, ...
    networks/N001/ ...
    selection.json
    result.json
  evaluation/          created by `evaluate`
    P0/ calibration/ control/ [oracle/]   each: E000..E049 episode dirs + result.json
    results.json
  diagnosis-review.json  written by the researcher (§10.3)
  summary.json           written by `summarize`
```

`network.json`:

```json
{
  "id": "N002",
  "parent": "N001",
  "round": 2,
  "parameters": {...},
  "round_actions": 25000,
  "online_steps": 50678,
  "policy_updates": 48,
  "created_at": 0.0
}
```

`online_steps` is cumulative, starting from P0's 678. `policy_updates` counts the PPO
updates in the chain that changed the policy (N000 = 0; value-only updates are not counted).
P0's SAC history stays in N000's `sac-training.zip`.

---

## 7. Components

### 7.1 Simulator parameterization (modify existing files)

- **`simulators/dynamics_rl.py`, `DynamicsRLTask.__init__`.** Remove the two RL-only
  restrictions: "mass and restitution must be nominal" and "non-nominal friction requires
  render mode". Range validation stays in `experiment_settings`. After each `reset_episode`,
  raise `RuntimeError` if the live cube mass (`self.native.cube.get_mass()`) differs from
  `0.35 * mass_scale` by more than 1e-4 kg. Live material is already verified after every
  reset by `prepare_scoring`.
- **`orchestrator/dynamics_rl_worker.py`.** Replace `--friction-scale` with `--parameters`, a
  JSON object of the three keys. Pass it into the task settings and echo it in the ready
  message.
- **`integrations/dynamics_rl/client.py`.** `SimulatorClient(..., parameters=dict)` replaces
  `friction_scale`. Record the parameters in `transport-config.json` and drop the state-mode
  restriction.
- **`orchestrator/dynamics_rl.py`.** Adapt the call sites
  (`parameters={"friction_scale": f, "mass_scale": 1.0, "restitution": 0.0}`). The CLI behaves
  exactly as before.

### 7.2 Layout splits (`integrations/dynamics_rl/data.py`)

Add these streams to `layout_seeds`: `"adaptation_dev": 71`, `"adaptation_online": 73`,
`"adaptation_eval": 79`, `"trainability_eval": 83`.

Online layouts for round r, used by every branch:
`layout_seeds("adaptation_online", round_actions + 1, 1000 * seed + r)`.

The learner seed for round r is `1000 * seed + r`.

### 7.3 Continued training (revision 1: PPO)

**Revert** the M2 change to `integrations/dynamics_rl/learning.py` (zero-BC path,
`update_after_transitions`, and the `custom_objects` argument of `load_model`) unless
something outside the old SAC adaptation path still uses it. SAC adaptation is gone;
`learning.py` keeps serving DynamicsRL and P0 loading only.

**Module `integrations/dynamics_rl/adaptation.py`** (rewritten; ≤ 250 lines) holds all PPO
code:

```python
class TanhGaussianPolicy(ActorCriticPolicy): ...  # mean = tanh(action_net(latent_pi))
class AdaptationPPO(PPO): ...                     # param groups, warm-up, trust region, mask
def convert_p0(sac_training_zip, training, seed, output_zip): ...
def load_round_model(training_zip, env, training, seed): ...
def train_round(model, actions, *, on_episode, run_dir, checkpoint_interval): ...
def save_network(model, directory, metadata): ...   # training.zip + policy.pt + network.json
```

`TanhGaussianPolicy`:

- `net_arch = dict(pi=[256, 256], vf=[256, 256])`, ReLU, separate policy and value networks.
- The Gaussian mean is `tanh(action_net(latent_pi))`. The state-independent `log_std` is
  filled with `exploration_log_std` and is not trainable (`requires_grad=False`, not in the
  optimizer).
- One Adam optimizer with two parameter groups: policy (`mlp_extractor.policy_net`,
  `action_net`) at `actor_learning_rate`, and value (`mlp_extractor.value_net`, `value_net`)
  at `value_learning_rate`. `AdaptationPPO._update_learning_rate` keeps each group's own rate
  (SB3 would otherwise overwrite both with one schedule).
- The value network sees the full 57-dim observation. The robust mask applies to the policy
  network's first layer only.

`convert_p0` (called once per run directory at freeze, and by `trainability`):

- Load P0's SAC zip with the existing `load_model`. Build `AdaptationPPO` on a spaces-only
  stub environment (`Box(-inf, inf, (57,))`, `Box(-1, 1, (7,))`) with the `training` values
  and `seed`.
- Copy `actor.latent_pi` into `mlp_extractor.policy_net` and `actor.mu` into `action_net`.
  The value network keeps its seeded initialization.
- Set `num_timesteps` to P0's `online_steps` (678) and `policy_updates` to 0. Save to
  `output_zip`.

`load_round_model`:

- `AdaptationPPO.load(training_zip, env=env, custom_objects=...)`, with every `training`
  value passed explicitly so that the frozen config, not the zip, decides: `n_steps`,
  `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`, `target_kl`,
  `max_grad_norm`, `ent_coef=0`. Then set the learning rates, `exploration_log_std`,
  `max_update_kl` and `value_warmup_epochs` on the model.
- Set `_last_obs = None`, so SB3 resets the new environment instead of continuing a stale
  observation. Call `set_random_seed(seed)`. Mark the next rollout as the round's first.

`AdaptationPPO.train()` (one call per rollout):

1. **First rollout of the round:** run `value_warmup_epochs` epochs of value-only regression
   (MSE to the rollout's returns, minibatches of `batch_size`, value group only). Make no
   policy update. Record `value_only: true`.
2. **Later rollouts:** snapshot the policy-group parameters, then run SB3's `PPO.train()`
   (clipped objective, `target_kl` early stopping, advantage normalization, gradient
   clipping), with the robust gradient-mask hook registered for its duration.
3. **Trust region:** compute the exact `KL = mean_s Σ_d (μ_new − μ_old)² / (2σ²)` over the
   rollout's observations (σ = exp(`exploration_log_std`)). If `KL > max_update_kl`, set the
   policy group to `θ_old + α·(θ_new − θ_old)`, halving α from 1 until the KL fits (at most 10
   halvings; α = 0 if it still does not fit). Record α as `step_fraction`. Value parameters and
   optimizer state are left as they are.
4. If `step_fraction > 0`, increment `policy_updates`.

Per-update record appended to `rl-metrics.jsonl` (replacing the SAC 100-action records):
`online_steps, value_only, epochs_completed, policy_loss, value_loss, approx_kl, exact_kl,
step_fraction, clip_fraction, explained_variance, drift_from_parent, elapsed_seconds`.
`drift_from_parent` is `mean_s max_d |μ_now − μ_parent|` on that rollout's observations, with
the parent's means computed once per round.

`train_round`:

- `model.learn(actions, reset_num_timesteps=False, callback=...)`. The service and runner
  guarantee that `actions` is a multiple of `n_steps`, so a round collects exactly `actions`
  transitions.
- Appends `online-episodes.jsonl` per episode, as before. Writes evidence-only
  `checkpoints/step_XXXXXX.zip` at rollout ends where the round's action count is a multiple
  of `checkpoint_interval`.
- A partial episode at the round's end is recorded `complete: false`, as before.

`save_network` writes `training.zip`, `policy.pt` and `network.json` (§6). `policy.pt` uses the
existing `POLICY_FORMAT` payload, built from `mlp_extractor.policy_net` and `action_net`, so
`load_policy`, rollouts and deployments are unchanged. The payload's `rl_updates` holds
`policy_updates`, and `metadata.learner` records `algorithm: "ppo"`, the `training` values and
the masked columns.

Required properties, each with a test in §10.1:

1. **Exact conversion.** On random observations, `convert_p0`'s deterministic action equals P0
   `policy.pt` within 1e-6. Masked columns are zero, `log_std` equals `exploration_log_std`
   and is not trainable.
2. **Chain inheritance.** After load, the policy, value and optimizer state equal what the
   previous round saved. `num_timesteps` and `policy_updates` continue; a round adds exactly
   `actions` to `num_timesteps`.
3. **Warm-up.** After a round's first rollout, the policy parameters are unchanged and the
   value parameters have changed.
4. **Trust region.** With a learning rate large enough to violate it, every recorded update has
   `exact_kl ≤ max_update_kl` (recomputed independently in the test) and some
   `step_fraction < 1`.
5. **Mask.** The masked first-layer columns are still zero after a round.
6. **Export.** `policy.pt` loads with `load_policy` and matches the model's deterministic
   action within 1e-6.
7. **Learning rates.** The optimizer has exactly the two parameter groups at their configured
   rates after `train()`, and `log_std` is in neither.

### 7.4 Jobs (`orchestrator/adaptation_jobs.py`)

Invocation: `python -m orchestrator.adaptation_jobs <job-dir>`.

- The job reads `<job-dir>/spec.json` and writes `<job-dir>/result.json`.
- Exit status is 0 when the job ran to its end, even if episodes failed physically.
- Exit status is nonzero only when the job itself crashed. `result.json` then has
  `status: "failed"`, `error` and `traceback`.

Every spec contains: `kind`, `gpu`, `simulator_python`, `startup_timeout`, `step_timeout`,
`parameters` (all three keys).

| `kind` | Extra fields | Behavior |
|---|---|---|
| `rollout` | `network` (path to `policy.pt`), `layouts` (list of `{index, layout_seed, scene}`), `fresh_simulator_per_episode` (bool) | Rendered. Episodes run in order with `run_episode`. One simulator per job unless fresh-per-episode. If an episode raises: record it `complete: false` with the error, mark the remaining layouts `not_started`, stop. |
| `train` | `parent` (path to `training.zip`), `round`, `actions`, `seed`, `training` (§9), `output` (network dir) | State mode, one simulator, `RemoteEnv` over the round's online seeds. Runs `load_round_model` → `train_round` → `save_network`. |

`result.json` for each kind:

- `rollout`:

  ```json
  {"status": "complete|failed",
   "episodes": [{"layout": 0, "success": false, "complete": true, "actions": 500,
                 "error": null, "episode_dir": "..."}],
   "successes": 0, "elapsed_seconds": 0.0}
  ```

  Not-started layouts appear with `"complete": false, "not_started": true`.
- `train`:

  ```json
  {"status": "...", "round_actions": 25000, "online_steps": 0, "rl_updates": 0,
   "online_episodes": 0, "online_successes": 0, "elapsed_seconds": 0.0}
  ```

Host-side helpers in the same module:

- `start_job(job_dir, spec) -> Popen` (asynchronous, used by the service);
- `run_job(job_dir, spec) -> dict` (synchronous, used by the runner).

### 7.5 Agent export (`orchestrator/adaptation_export.py`)

- `export_episode(host_episode_dir, agent_dir)` reads the worker's `transitions.jsonl` and
  `result.json`. It writes `trajectory.npz` (§8.2) and `result.json` (§8.3), and hard-links the
  three `raw_cam_*.mp4` files as `cam_head.mp4`, `cam_left_wrist.mp4` and
  `cam_right_wrist.mp4`.
- `export_training(job_dir, agent_dir)` writes `training.json`, `online-episodes.csv` and
  `losses.csv` (§8.4).
- Exported target records contain no `parameters` field. Exported probe records echo the
  agent's own parameters.

### 7.6 Experiment service (`orchestrator/adaptation_service.py`)

`AdaptationService(branch_dir, config, layouts, deadline)` serves newline-delimited JSON on
a Unix socket, using the same framing as `orchestrator/dynamics_client.py`. It handles one
connection at a time and appends every request and response to `requests.jsonl`. A refused
request returns `{"ok": false, "error": "..."}` and consumes nothing.

**Operations**

| `op` | Fields | Effect |
|---|---|---|
| `status` | — | Budgets used and remaining, wall time remaining, current job, network table (id, round, parameters, round actions, target development successes), job list. Never advances anything. |
| `probe` | `network`, `parameters`, `layouts` (subset of 0–4), `hypothesis`, `prediction` | Rollout job of a registered network in Sθ on the chosen development layouts. Charges `len(layouts)` diagnostic episodes. |
| `train` | `parameters`, `actions`, `rationale` | Train job continuing the chain from the latest network created by a successful train job (N000 initially). On success: register `N00k`, then automatically start the deployment rollout of `N00k` in T on all 5 development layouts. Charges 1 round, `actions` actions, 5 target episodes. |
| `wait` | `job`, `seconds` (1–900) | Blocks until the job is finished (both sub-jobs for `train`) or `seconds` elapse; returns the job record. |
| `submit` | `diagnosis` (§8.6) | Records the diagnosis and ends the branch: later `probe`/`train`/`submit` are refused. Refused while a job is running. |

**Request validation** (the only layer that validates agent input):

- parameter keys must be a subset of the three; values finite and within the ranges in §4;
  omitted keys take nominal values;
- the network ID must exist;
- layouts must be unique integers in 0–4, and at least one;
- `hypothesis`, `prediction` and `rationale` must be nonempty strings;
- `actions` must be an integer in [2,000, min(25,000, remaining actions)] and a multiple of
  `training.n_steps`;
- the request must fit the remaining budget;
- no job may be running, and wall time must remain;
- `train` additionally requires a reproduction probe (§2.15) for its exact nominal-filled
  parameters and for its parent network (the latest registered network). The service checks
  this from its own job records: the probe's completed episode results and the parent's
  recorded target-development outcomes per layout (baseline target for N000, deployment for
  later networks). The `status` network table gains `target_dev_outcomes` (per-layout success,
  or `null` without a deployment record) so the agent can see what has to be reproduced.

**Accounting.**

- Rounds, actions and diagnostic episodes are charged when a job is accepted and are never
  refunded.
- Target episodes are charged when the deployment starts.
- A failed train job registers no network and starts no deployment. The next round continues
  from the latest registered network.

**Deadline.** The runner owns the deadline: agent start + `wall_hours`. When it passes, the
runner removes the container, kills the running job's process group, and records that job as
`stopped_by_deadline`. A stopped train job registers no network.

**Pre-registration.** The hypothesis, prediction and rationale are written to `job.json` with
the submission timestamp before the job starts. That record is the evidence that predictions
came before results.

### 7.7 Workspace client (`tasks/DynamicsAdaptation/Workspace/scripts/adapt.py`)

Standard library only. It reads `EMBODIEDEVO_SOCKET`, prints the JSON response, and exits
with status 1 when `ok` is false.

```bash
python scripts/adapt.py status
python scripts/adapt.py probe --network N000 --parameters notes/p.json --layouts 0,2 \
  --hypothesis "..." --prediction "..."
python scripts/adapt.py train --parameters notes/p.json --actions 25000 --rationale "..."
python scripts/adapt.py wait J0003 --seconds 600
python scripts/adapt.py submit submission/diagnosis.json
```

### 7.8 Workspace template (`tasks/DynamicsAdaptation/Workspace/`)

`instruction.md` uses placeholders filled at workspace creation, following the style of
`prepare_workspace` in `dynamics_repair.py`. Required content, in this order:

1. **Goal.** P0 succeeds in the nominal source but may fail in a target whose dynamics
   differ in unknown ways. Restore target success by calibrating the source simulator and
   continuing RL training there. The agent never sends robot actions; the host runs only
   registered networks.
2. **Policy interface.** A state-based MLP. Effective inputs: cube position, plate position,
   right end-effector position, plate-minus-cube, and remaining action allowance. Output:
   absolute right end-effector target pose and gripper opening at 25 Hz. Training is PPO
   with a fixed small exploration noise and a per-update trust region, so each round changes
   the policy gradually. The architecture, reward, training algorithm and hyperparameters
   are fixed and cannot be changed.
3. **Parameters.** The three cube parameters, their meaning and ranges, and a pointer to
   `nominal.json`. No statement about which parameter differs.
4. **Operations, budgets and selection rule.** The five operations, the budgets (filled in),
   and the selection rule (§2.5). Tell the agent that `wait` blocks for up to `--seconds`, so
   its shell timeout must be longer than that value.
5. **Data.** Where the baseline and job outputs appear (§8), with the trajectory field list.
6. **Success criteria.** `describe_criteria(Criteria(**criteria))`.
7. **Rules.**
   - **Reproduce before training (enforced).** `train` at parameters θ is refused unless a
     completed probe at exactly θ, of the network the round will continue from, matched that
     network's target-development success or failure on every probed layout and included at
     least one of its target failures. Check failure modes, not only outcomes, in your
     analysis.
   - Record the hypothesis and prediction before every probe.
   - Test at least one competing hypothesis.
   - Target data is for diagnosis and selection only.
   - Failed attempts are kept.
   - Finish with `submit`.
8. **Diagnosis schema** (§8.6).

The Codex prompt passed to `agent_argv` reads:

> Read instruction.md and work autonomously. Use scripts/adapt.py for all experiments.
> Write analysis code under notes/. Finish by submitting submission/diagnosis.json with
> scripts/adapt.py submit.

### 7.9 Runner CLI (`orchestrator/dynamics_adaptation.py`, run with `.venv-rl/bin/python -m`)

Every command requires `--run-dir`, whose name starts with `YYYYMMDDHHMM`, and `--gpu`
(except `summarize`). Wrap GPU work in `gpu_lease`.

| Command | Arguments | Behavior |
|---|---|---|
| `preflight` | `--config --run-dir --gpu` | GPU checks P1–P4 (§10.2). Writes `preflight.json` with `passed`. New run directory. |
| `trainability` | `--config --run-dir --gpu [--parameters FILE]` | N000 is prepared exactly as in `freeze` (P0 copy plus `convert_p0`). Oracle chain at `--parameters` (default: the configured target). Rounds × actions per round at those parameters; after each round, deploy on the 5 development layouts at the same parameters; select (§7.10). Stop early once a candidate has 5/5, since later rounds cannot change the selection. Evaluate the selected network on 50 `trainability_eval` layouts at the same parameters, fresh simulator per episode. Writes `trainability.json` with `passed = successes ≥ required_successes`. New run directory. |
| `run` | `--config --run-dir --gpu --preflight DIR [--scripted-agent FILE] [--stage NAME]` | Stages in order: `freeze` (§6 files, P0 copy, `convert_p0` into `networks/N000/training.zip`, layouts), `baseline`, `calibration`, `control`. Requires `preflight.json` `passed: true`. Each stage writes `<stage>/result.json` when it completes. `--stage NAME` runs only that stage in an existing run, and only if its directory does not yet exist and all earlier stages have completed. A stage that started and failed is not rerun; the researcher records any rerun decision separately. |
| `evaluate` | `--run-dir --gpu [--oracle TRAINABILITY_DIR]` | §7.12. Requires completed `calibration` and `control`; refuses if `evaluation/` exists. |
| `summarize` | `--run-dir` | §7.13. Reads only. |

`--scripted-agent FILE` runs a host-side Python script instead of the container. Its working
directory is the workspace and `EMBODIEDEVO_SOCKET` points at the service. It is used only for
the M4 end-to-end smoke test, and it records `agent_kind: "scripted"`. Formal runs use the
Codex container configured through `agent_task` (§9), exactly as `dynamics_repair` does.

**Calibration stage sequence:**

1. Create the workspace.
2. Export the baseline into it.
3. Start the service thread.
4. Launch the container (or scripted agent) and record the deadline.
5. Wait until the agent exits, the deadline passes, or the agent submits. After a submission,
   wait up to 120 s for the agent to exit, then remove the container.
6. Copy the Codex sessions, record token usage, write `selection.json` and `result.json`.

`result.json.end_reason` is one of `submitted`, `agent_exited_without_submission`,
`deadline`.

### 7.10 Selection

```python
def select(candidates):
    # candidates: [(network_id, target_dev_successes)], in creation order, P0 first
    return max(enumerate(candidates), key=lambda c: (c[1][1], -c[0]))[1][0]
```

- P0's entry uses `baseline/target` successes.
- Each round network's entry uses its deployment successes, with incomplete episodes counted
  as failures.
- Networks without a deployment, because their train job failed or was stopped, are not
  candidates.
- `selection.json` lists every candidate and the winner.

### 7.11 Control stage

The control reads the calibration branch's train jobs in order. For each one whose train job
completed:

- train round r with **nominal** parameters, the same `actions`, round index and seed
  derivation, continuing the control's own chain from N000;
- deploy on the 5 development layouts in T only if the calibration job's deployment started
  (its `deploy/` sub-job exists). Otherwise the control network is registered but, like its
  calibration counterpart, is not a candidate. This keeps target queries matched.

The control is not subject to the reproduction rule (§2.15); it has no probes.

Calibration rounds that failed or were stopped are listed in `control/result.json` and not
replicated. Selection follows §7.10. The control has no time limit; its elapsed time is
recorded.

### 7.12 Final evaluation

The policies, in order:

1. P0;
2. the calibration-selected network;
3. the control-selected network;
4. optionally, the oracle-selected network from `--oracle`.

Each is evaluated on the 50 `adaptation_eval` layouts in T, rendered, with a fresh simulator
per episode, as one rollout job per policy (`fresh_simulator_per_episode: true`). A network
selected by more than one arm (only P0 can be) is evaluated once and referenced by the other
arms.

`results.json` holds, per arm: network, successes, incomplete, the per-episode list, and
elapsed time. It also holds `accepted`, which is true when the calibration arm has
≥ `required_successes`. Every planned episode stays in the denominator. Nothing is retried.

### 7.13 Summary

`summary.json` and a printed table answer the plan's two questions separately.

1. **Was task performance restored?** Final 50-layout successes per arm, the per-round
   table for each branch, and costs. Per-round table columns: parameters, actions, online
   successes, target development successes. Costs: wall time per stage, GPU job time by kind,
   model tokens, actions, updates, target episodes, diagnostic episodes.
2. **Is the diagnosis supported?** The agent's diagnosis (self-report) and the host review
   from `diagnosis-review.json` (default `not_reviewed`). Neither changes the other.

The summary also cites the P0 source-evaluation record (§2.1) and the trainability result if
`config.trainability_run` is set.

---

## 8. Agent-visible data

### 8.1 Workspace tree

```
workspace/
  instruction.md          read-only
  nominal.json            read-only: nominal_parameters() + ranges
  scripts/adapt.py        read-only
  observations/           read-only to the agent; written by the host
    baseline/source/L0..L4/   episode dirs (§8.2–8.3)
    baseline/target/L0..L4/
    jobs/J0001/request.json   the agent's own request, as accepted
    jobs/J0001/episodes/L0/   (probe) or deploy/L0..L4/ + training/ (train)
  notes/                  writable: analysis code, hypotheses
  submission/             writable: diagnosis.json
```

### 8.2 `trajectory.npz`

T is the number of actions. State arrays have T+1 rows (initial state plus one row per
action).

| Field | Shape | Source |
|---|---|---|
| `step` | (T+1,) | `sample.step` |
| `action` | (T, 7) | normalized network output |
| `applied_action` | (T, D) | decoded native command sent to the controller |
| `right_ee_pose` | (T+1, 7) | `public_state.right_ee_pose` |
| `right_arm_joint_position` | (T+1, n) | `public_state.right_arm_joint_state` |
| `right_arm_joint_velocity` | (T+1, n) | `right_arm_joint_velocity` |
| `right_gripper_joint` | (T+1, k) | `public_state.right_ee_joint_state` |
| `right_gripper_opening` | (T+1,) | `sample.gripper_opening.right` |
| `right_finger_center` | (T+1, 3) | `sample.gripper_position.right` |
| `cube_position`, `cube_orientation` | (T+1, 3), (T+1, 4) | `sample` |
| `cube_linear_velocity`, `cube_angular_velocity` | (T+1, 3) each | `sample` (measured, not solver) |
| `plate_position`, `plate_orientation` | (T+1, 3), (T+1, 4) | state |

Excluded: the encoded observation, `reward`, `native_reward`, `scoring`, all contact fields,
solver velocities, snapshots, materials and mass.

### 8.3 Episode `result.json`

```json
{"layout": 0, "success": false, "complete": true, "terminated": false,
 "truncated": true, "actions": 500, "error": null}
```

### 8.4 Training export

- `training.json`: `{round, network, parent, parameters, round_actions, online_episodes, online_successes, elapsed_seconds, status}`.
- `online-episodes.csv`: `episode, success, complete, actions`.
- `losses.csv`: `online_steps, value_only, policy_loss, value_loss, exact_kl, step_fraction, drift_from_parent` (one row per PPO update).

### 8.5 `status` response

```json
{"ok": true, "wall_seconds_remaining": 0,
 "budgets": {"rounds": [0, 4], "actions": [0, 100000], "target_episodes": [0, 20],
             "diagnostic_episodes": [0, 20]},
 "running": null,
 "networks": [{"id": "N000", "round": 0, "parameters": {...},
               "target_dev_successes": 3,
               "target_dev_outcomes": [false, false, true, true, true]}],
 "jobs": [{"id": "J0001", "kind": "probe", "status": "complete"}],
 "submitted": false}
```

In `budgets`, each pair is `[used, limit]`.

### 8.6 `submission/diagnosis.json`

```json
{
  "estimated_target_parameters": {"friction_scale": 1.0, "mass_scale": 1.0, "restitution": 0.0},
  "conclusion": "text",
  "evidence_jobs": ["J0002"],
  "competing_hypotheses": [
    {"hypothesis": "text", "test_jobs": ["J0003"],
     "outcome": "rejected | not_rejected | untested", "reason": "text"}
  ],
  "limitations": "text",
  "preferred_network": "N002"
}
```

The service checks that the keys are present and of the right types (trust boundary). An
invalid submission is refused, and the agent may fix it and resubmit.

---

## 9. Configuration (`tasks/DynamicsAdaptation/adaptation.example.json`)

```json
{
  "agent_task": "RoboDojo",
  "simulator_python": ".venv-robodojo/bin/python",
  "seed": 0,
  "position_radius": 0.03,
  "p0": {
    "policy": "runs/202610302107_dynamics_rl_train_v4/checkpoints/best_rl.pt",
    "training": "runs/202610302107_dynamics_rl_train_v4/checkpoints/best_rl.zip",
    "source_evaluation": "runs/202610302128_dynamics_rl_eval"
  },
  "source_parameters": {"friction_scale": 1.0, "mass_scale": 1.0, "restitution": 0.0},
  "target_parameters": {"friction_scale": 0.25, "mass_scale": 1.0, "restitution": 0.0},
  "parameter_ranges": {"friction_scale": [0.05, 2.0], "mass_scale": [0.25, 4.0],
                       "restitution": [0.0, 1.0]},
  "development_layouts": 5,
  "evaluation_layouts": 50,
  "required_successes": 45,
  "budgets": {"wall_hours": 8, "rounds": 4, "actions_per_round": 25000,
              "actions_total": 100000, "target_episodes": 20, "diagnostic_episodes": 20},
  "training": {
    "actor_learning_rate": 3e-6,
    "value_learning_rate": 3e-4,
    "exploration_log_std": -4.6,
    "n_steps": 1000,
    "batch_size": 250,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "target_kl": 0.01,
    "max_update_kl": 0.02,
    "max_grad_norm": 0.5,
    "value_warmup_epochs": 20,
    "checkpoint_interval": 5000,
    "learner_threads": 4
  },
  "trainability_run": null,
  "startup_timeout": 1200,
  "step_timeout": 120
}
```

- `load_config` rejects unknown keys and wrong types. That is the whole of config validation.
- `agent_task` resolves model, image and isolation through `load_task`, as in
  `dynamics_repair.load_config`.
- The `training` values are revision-1 placeholders. σ = exp(−4.6) ≈ 0.01 normalized
  (≈ 2.5 mm in x, ≈ 1.8° in rotation). They are frozen only after P5 and the D1 development
  run. `checkpoint_interval` and every allowed `actions` value must be multiples of
  `n_steps`.
- The smoke configuration `tasks/DynamicsAdaptation/smoke.json` is a complete copy of the
  example with only these values changed: budgets of 1 round × 2,000 actions, 2 diagnostic
  episodes, 5 target episodes and 1 h, and `evaluation_layouts: 2` with
  `required_successes: 2` so the smoke `evaluate` stays short. Its `training` block equals the
  example's.

---

## 10. Validation

### 10.1 CPU tests

Run in both environments. In `.venv`, the SB3 tests skip.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv-rl/bin/python -m unittest discover -s tests -v
```

| File | Requirement covered |
|---|---|
| `test_dynamics_rl_simulator.py` (update) | state mode accepts non-nominal parameters; out-of-range values rejected; mass mismatch after reset raises |
| `test_adaptation_training.py` (needs SB3; reuse the fake env from `test_dynamics_rl_training.py`) | §7.3 properties 1–7 (PPO, revision 1); `_last_obs` reset; chain across two rounds |
| `test_adaptation_jobs.py` (fake `SimulatorClient`) | rollout records an incomplete episode plus `not_started` and stops; train job writes `network.json`, `training.zip`, `policy.pt` |
| `test_adaptation_export.py` | `trajectory.npz` has exactly the §8.2 fields; exported target records have no `parameters`; no host-only key appears anywhere under the workspace |
| `test_adaptation_service.py` (fake job starter) | parameter validation; refusals consume nothing; each budget cap; one job at a time; deadline refusal; submit ends the branch; a failed train job registers no network and the next round continues from the previous one; hypothesis/prediction recorded before the job starts; reproduction rule (§2.15): train refused without a probe, with a probe at other parameters or of another network, with an outcome mismatch, with no probed target failure, and when the parent has no target failure; accepted after a reproduction; `actions` not a multiple of `n_steps` refused |
| `test_adaptation_runner.py` | selection rule (ties → earlier, P0 included, undeployed networks excluded); control schedule equals the completed calibration rounds (actions, seeds, layouts); run-dir prefix enforced; `--stage` refuses an existing stage directory; control deploys only where the calibration deployment started |

### 10.2 GPU preflight (`preflight`, run by the overseer)

Parameter sets: A = nominal; B = `{friction 0.5, mass 2.0, restitution 0.5}`; T = target.

| Check | Procedure | Pass |
|---|---|---|
| P1 Live parameters | A and B, state and render mode, 3 resets each on different layouts; read `cube_physx_shape_materials` and `cube_mass` from the host snapshot after every reset | all materials equal the configured values (float32); mass within 1e-4 kg |
| P2 Render/state parity | P0 on one development layout under A and B in both modes | cube trajectory max abs difference ≤ 1e-4 m; same success |
| P3 Cross-process identity | two fresh state-mode simulators at T, P0 on one layout | trajectories identical (≤ 1e-6 m) |
| P4 Chain on GPU (revision 1) | `convert_p0`, then 2 rounds × 2,000 actions under B (train jobs), save, reload | conversion matches P0 `policy.pt` (≤ 1e-6); counters cumulative; mask columns zero; every update's `exact_kl` ≤ `max_update_kl`; `policy.pt` loads |
| P5 Nominal stability (revision 1) | `convert_p0`, then one 25,000-action train job at **nominal** parameters, then deploy the result on the 5 development layouts at nominal | deployment 5/5; no update with `exact_kl` > `max_update_kl`; final `drift_from_parent` reported; online success rate under exploration noise reported per 5,000 actions |

Also rerun the existing reset regression after the M1 changes:
`.venv-rl/bin/python -m orchestrator.dynamics_rl preflight`. Earlier passes do not cover the
changed adapter.

### 10.3 Diagnosis review (researcher, outside the agent workspace)

`diagnosis-review.json`:

```json
{"calibration": {"assessment": "supported | ambiguous | unsupported",
                 "evidence_jobs": ["J0002"], "notes": "text"}}
```

The rubric follows DynamicsRepair. For this experiment, `supported` additionally requires all
of the following:

- the agent named a decrease in cube friction;
- it gave pre-registered predictions for at least one probe that separated friction from mass
  (§2.9);
- the predicted direction matched the observations.

An estimate near the target with no such discrimination is `ambiguous`. Target success is
never evidence by itself.

---

## 11. Experimental procedure

**Phase 0 — before any formal run (host only, no model calls)**

1. **Preflight** (§10.2) and the reset regression. Required: both pass.
2. **Screening:** measure P0's transfer gap at the target with the existing CLI.

   ```bash
   .venv-rl/bin/python -m orchestrator.dynamics_rl rollout \
     --config tasks/DynamicsRL/train.example.json \
     --checkpoint runs/202610302107_dynamics_rl_train_v4/checkpoints/best_rl.pt \
     --friction-scale 0.25 --episodes 10 \
     --run-dir "runs/$(date +%Y%m%d%H%M)_adaptation_screening" --gpu 0
   ```

   Stop rule (D2): if P0 succeeds on ≥ 9/10 layouts, there is no meaningful gap. Stop and
   revisit the target before any formal run, and record the decision.
3. **Procedure development (D1, option A).** First pass P5 (nominal stability). Then run
   `trainability --parameters DEV_SHIFT.json` with candidate `training` settings on a
   development shift that is not the target, on which P0 partially fails as at the target (10–30% frozen-P0 success over about 10,000 training actions; see the §12 status). Freeze the chosen
   settings in the formal config.
4. **Trainability gate:** `trainability` at the true target with the frozen config.
   - **Pass** (≥ 45/50): proceed.
   - **Fail:** the formal run does not start. The result is reported as a property of the
     training procedure, not of the agent. Any revision to the procedure is versioned and
     disclosed.

**Phase 1 — formal run:** `run` (freeze → baseline → calibration → control).

**Phase 2:** `evaluate` (with `--oracle` if D3 is accepted), then the researcher writes
`diagnosis-review.json`, then `summarize`.

**Time estimates on the single RTX 4090**, based on about 10.4 training actions/s, about
70 s per rendered episode, and about 4 min per fresh-simulator episode:

| Step | Estimate |
|---|---|
| Preflight + reset regression | ~1 h |
| Screening (10 fresh episodes) | ~40 min |
| Trainability (≤100k actions + ≤4×5 deployments + 50 evaluation episodes) | ~6.5 h |
| Formal baseline | ~20 min |
| Calibration branch | ≤ 8 h |
| Control | ~3.5 h |
| Final evaluation, 3 arms (+ oracle) | ~10 h (~13 h) |

GPU work is serial; never run two commands that use the simulator at the same time.

---

## 12. Milestones

Codex stops after each milestone and reports: files changed, line counts, CPU test commands
and output, and any deviation from this specification with its reason. The overseer reviews
the diff against §3, runs the GPU checks, and either approves or returns findings.

| # | Deliverable | CPU acceptance | Overseer GPU check |
|---|---|---|---|
| M1 | §7.1, §7.2; `preflight` command with P1–P3 | updated simulator tests | P1–P3 + reset regression |
| M2 | §7.3 | `test_adaptation_training.py` | P4 |
| M3 | §7.4, §7.10; `trainability` command | jobs and selection tests | screening, then trainability (Phase 0 steps 2–4, after D1) |
| M4 | §7.5–7.9: export, service, client, template, `run` freeze/baseline/calibration, `--scripted-agent`, `smoke.json` | export, service and runner tests | scripted end-to-end smoke with `smoke.json` |
| M5 | §7.11–7.13: control, evaluate, summarize; `docs/DYNAMICS_ADAPTATION.md` user guide; update the stale plan pointer at the top of `docs/DYNAMICS_REPAIR.md` | runner tests | control and evaluate on the smoke run |
| R1 | Revision 1 (§2.14, §2.15): PPO in `adaptation.py` (§7.3) with the `learning.py` revert; `freeze`/`trainability` conversion; reproduction rule in the service, status and template; `actions` multiple of `n_steps`; PPO export columns; control deploy matching (§7.11); scripted smoke agent updated; `adaptation.example.json`/`smoke.json` training blocks; user guide | updated training, service, runner and export tests | P4 (revision 1), P5, scripted smoke rerun |
| M6 | — | full suite in both environments | formal Phases 1–2 |

M4 and M5 can be implemented while the overseer runs the M3 GPU work.

**Status, 2026-11-01.** M1–M5 approved. GPU: P1–P3 and the reset regression passed
(`runs/202611011145_adaptation_preflight_m1`, `runs/202611011205_*`); the SAC P4 passed
(`runs/202611011213_adaptation_p4_m2`); screening 0/10 (`runs/202611011222_adaptation_screening`);
SAC smoke `runs/202611011301_adaptation_smoke` exposed the collapse that led to revision 1.
Before R1, the M1 `preflight.json` (P1–P3) stays the run's `--preflight` evidence; P4 and P5
are overseer runs recorded beside it.

**Status, 2026-11-01 (after R1).**

R1 approved. The revision-1 GPU checks passed:
- P4: `runs/202611011453_adaptation_p4_ppo`. The conversion difference was 0.0 and the KL bound held.
- P5: `runs/202611011503_adaptation_p5_ppo`. After 25k nominal actions the deployment stayed 5/5.
- Scripted smoke: `runs/202611011549_adaptation_smoke_r1`. Training was refused before a
  reproducing probe and accepted after one.

D1, first development shift:
- The screen (`runs/202611011644_adaptation_devshift_screen`) found P0 at 0/5 with friction 0.5 × mass 2.5.
- Candidate log-std −4.6 was run there (`runs/202611011659_adaptation_d1_devshift_ls-4.6`).
- Result: 0 online successes in 200 training episodes and 0/5 on every round's deployment.
  N000 was selected and scored 0/50.
- Training only moved the policy at random. N004 lifts the cube about 6 mm, where P0 lifts it
  about 40 mm before it slips.
- Diagnosis: the shaped reward is potential-based, so a fitted value function cancels it. With
  no success anywhere in the rollouts there is no learning signal, so this shift cannot rank
  training settings.

D1 shift criterion, as revised:
- At the target, P0 is a partial failure (3/5 on the development layouts). The development
  shift must also be a partial failure, with P0 at 1–4/5 on the same layouts.
- Any shift whose P0 result is 0/5 or 5/5 is rejected.
- The first shift and its run are kept as a recorded attempt.

D1, second shift screen (`runs/202611012312_adaptation_devshift_screen2`):
- P0 scored 0/5 at friction 0.6 × mass 2.5 and at friction 0.55 × mass 2.5.
- P0 scored 1/5 at friction 0.5 × mass 2.0, succeeding only on L4.
- Mass hurts P0 beyond the friction/mass ratio. At the target (ratio 0.25, mass 1) P0 scores
  3/5; at the same ratio with mass 2.0 it scores 1/5. So the chosen shift is not physically
  equivalent to the target.
- The development shift is friction 0.5 × mass 2.0. Candidate log-std −4.6 is running there
  (`runs/202611012339_adaptation_d1_devshift2_ls-4.6`).
- Result: the deployments went 1/5, then 0/5 for rounds 2–4. Training saw 0 successes in 200
  episodes. Selection fell back to N000, which scored 1/50.

D1 diagnostics, run with P0 frozen (actor learning rate 0) on the round-1 training layouts:
- `runs/202611020551_adaptation_d1_noise_sweep`, at shift 2: P0 succeeded in 0/20 episodes at
  each of log-std −9, −6 and −5.3. So shift 2 is about 2% for P0 across layouts, and the 1/5
  on the development layouts came from one easy layout (L4).
- `runs/202611020639_adaptation_d1_noise_nominal`, at nominal: 44/44 at log-std −9, but 15/28
  at −4.6. The training environment is sound, and σ = 0.01 exploration noise alone halves
  P0's success rate. P0 was trained at log-std −6.

D1 criterion, as revised again:
- Five layouts are too few to place a shift. Shifts are now screened by frozen P0's success
  at log-std −9 over about 10,000 training actions, and must fall within 10–30%. The target is
  about 20% (3/5 on the development layouts, 0/10 on the screening layouts).
- Candidate 2 is `exploration_log_std` −6 with `target_kl` 0.16 and `max_update_kl` 0.32.
  KL scales with 1/σ², so this gives the same per-update bound on how far the action means
  can move as the −4.6 configuration.
- Shift screen at log-std −9: friction 0.45 × mass 1.5 scored 16/28 (57%), too easy. Mass ×3.5
  alone scored 1/20 (5%), too hard. Friction 0.4 × mass 1.5 scored 5/22 (23%) and was chosen.
  Runs: `runs/202611020715_*`, `runs/202611020733_*`, `runs/202611020750_*`.
- Candidate 2 on that shift: `runs/202611020807_adaptation_d1_devshift3_ls-6`.
  - Rounds 1–4 completed. The deployments scored 0, 0, 1 and 0 out of 5, against 1/5 for P0.
    Training episodes succeeded 4 times in 200.
  - Selection fell back to N000.
  - The run was stopped at the user's request on 2026-11-02 during the final evaluation, after 11 of
    50 episodes (2 successes). It has no `trainability.json` and is kept as a stopped attempt.

Video review (`runs/202611021159_adaptation_review/index.html`):
- P0 executes nearly the same timed trajectory regardless of the physics, with the gripper fully
  closed during the carry.
- With reduced grip the cube slips out during the carry, at steps 151–196 instead of the release
  at step 211. Success depends on whether the slip happens within about 2 cm of the plate centre.
- To adapt, the carry itself has to change. That is a qualitative trajectory change.

---

## 13. Reporting requirements

- Give two separate answers (§7.13): performance restored, and diagnosis supported.
- Disclose:
  - P0's provenance and teacher (§2.2);
  - how the target was chosen and the screening result;
  - the trainability result and any procedure revision;
  - the exact-identification limitation (§2.12);
  - the friction–mass confound (§2.9);
  - cost differences between the branches (the control has no probes and no agent time).
- Report all attempts, including incomplete ones, in their planned denominators. Classify
  failures only from evidence: crashes, timeouts and missing reports stay `unclassified`
  unless there is evidence for a specific cause.
- Results from five layouts are development signals, not estimates of success rates.
- Runtime artifacts stay under `runs/` and are never committed.

---

## 14. Decisions requiring researcher sign-off

**Signed off 2026-11-01:** D1 option A (tune on a non-target development shift, then gate at
the target); D2–D5 as recommended. Revision 1, same day: PPO fine-tuning with overseer-designed
safeguards (§2.14), and reproduction before every training round (§2.15).

| ID | Decision | Recommendation | Blocks |
|---|---|---|---|
| D1 | How to freeze the adaptation `training` values. Option A: tune on a non-target development shift, then gate at the target. Option B: tune using the trainability check at the target and disclose that the procedure was developed on the target. | A. Revision 1: the candidate set is over the PPO values, mainly `exploration_log_std` (e.g. {−5.3, −4.6, −3.9}) and `max_update_kl`, fixed after P5. | Phase 0 step 3 |
| D2 | Screening stop rule | Stop if P0 ≥ 9/10 at the target | Phase 0 step 2 |
| D3 | Oracle reference arm in the final evaluation | Include it, labelled `oracle`, outside the comparison | `evaluate` |
| D4 | Exclude contact flags from agent data | Exclude (§2.10) | M4 |
| D5 | Host-applied selection; deployment merged into `train` (§2.5–2.6) | As specified | M3/M4 |

## 15. Out of scope for v1

- Multiple targets or multiple dynamics types.
- Real robots.
- Visual policies.
- Changes to P0's architecture, observation encoder, reward or algorithm.
- Hyperparameters chosen by the agent.
- Concurrent GPU jobs.
- Resuming an interrupted agent branch.
- Inclusion in RoboDojo formal Test scores.
