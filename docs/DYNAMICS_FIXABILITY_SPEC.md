# Fixability study: can RL repair a source policy under reduced friction?

Status: specification for Codex, written 2026-11-02 by the overseer (Claude).
The overseer runs every GPU stage; the implementer writes code and CPU tests only.

## 1. Question and design

The agent experiment (`docs/DYNAMICS_ADAPTATION_SPEC.md`) is paused. Its trainability gate
failed: PPO starting from P0 could not repair P0's failures even when training used the true
shifted physics. The video review showed the cause. P0 replays a timed trajectory and ignores
the cube's state, and its successes under low friction depend on lucky slip timing.

This study answers a narrower question first:

> Starting from a policy that is **reliable** in the source (friction 1.0), and that **fails**
> after a friction drop, can RL in the failing environment restore reliable task success?

The study has four stages, each a separate run by the overseer:

| Stage | What | Physics | Output |
|---|---|---|---|
| S | Train a reliable source policy: noisy-teacher demonstrations → BC → PPO | friction 1.0 | `S*` network, source gate |
| T | Deploy the source policy unchanged | friction 0.3 and 0.25 | Transfer result, target selection |
| F | Scripted teacher at the target, normal and slowed carry | target | Feasibility reference (not training data) |
| A | PPO from the source network, in the target | target | Adapted network, fixability verdict |

The design has four principles:

- **One learner, one reward, one hyperparameter set.** Stages S and A use the identical PPO
  configuration. A differs from S only in the friction value and in its initial network. Do not
  tune anything for the target.
- **Reliability means "succeeds under its own exploration noise".** P0 was 50/50 when
  deterministic, but only about 55% with σ = 0.01 noise. That is chance, not reliability.
  The source gate therefore requires a high success rate both deterministically and when
  sampling from the training distribution.
- **The actor must see state, not time.** The actor input excludes the remaining-allowance
  clock (column 56) and the previous action (columns 49–55). P0 was dominated by its clock
  input, with a clock weight of about 200. A clock-free actor has to decide phase from the
  cube, hand and gripper state. This is the kind of policy that can react to slip. The critic
  sees the full 57-dimensional observation.
- **Held-out evaluation, paired layouts.** The same 50 `fix_eval` layouts are used at every
  friction, so source, transfer and adapted results compare scene by scene. Checkpoints are
  chosen only on `fix_dev` layouts.

## 2. Engineering rules (normative)

§3 of `docs/DYNAMICS_ADAPTATION_SPEC.md` applies unchanged. In short:

- No hashes, checksums or verification files, and no re-reading of files you just wrote.
- Validate only at trust boundaries (the config file and CLI arguments).
- Let errors propagate, and write the record once at the episode or job boundary.
- No retries, automatic reruns, fallbacks, mode switches, compatibility shims or speculative
  options.
- Reuse existing functions. One-line docstrings.
- Tests check required behaviour and mock only the simulator process.

AGENTS.md boundaries apply:

- Every command takes an explicit `--run-dir` with a `YYYYMMDDHHMM` prefix and refuses a
  populated directory.
- Failed and partial attempts stay in the record.
- Nothing under `runs/` goes into Git.

## 3. Files

| File | Change | Size guide |
|---|---|---|
| `integrations/dynamics_rl/fixability.py` | New: policy, PPO subclass, BC, exported-actor loader | ≤ 260 lines |
| `integrations/dynamics_rl/parallel.py` | New: threaded `VecEnv` over N `RemoteEnv`s, and a parallel episode runner | ≤ 140 lines |
| `orchestrator/dynamics_fixability.py` | New: CLI `demos`, `bc`, `train`, `evaluate`, `feasibility` | ≤ 450 lines |
| `tasks/DynamicsRL/fixability.example.json` | New: the §9 configuration, verbatim | — |
| `integrations/dynamics_rl/data.py` | Edit: add the split streams `fix_demonstration` 101, `fix_online` 103, `fix_dev` 107, `fix_eval` 109 to `layout_seeds` | +1 line |
| `docs/DYNAMICS_FIXABILITY.md` | New: user guide (commands, artifacts, gates) | short |
| `tests/test_dynamics_fixability.py` | New: §10 | — |

Do not modify `adaptation.py`, `adaptation_jobs.py`, the adaptation service or orchestrator,
`learning.py`, `teacher.py`, `features.py`, the worker or `client.py`. Import from them.

## 4. Parallel simulators (`parallel.py`)

The overseer measures capacity on the GPU and sets `num_envs` (§9). A single persistent
simulator runs about 11 actions/s.

- `simulator_pool(run_dir, config, gpu, parameters, mode)` is a context manager. It starts
  `num_envs` `SimulatorClient`s concurrently, using one thread per `start()`. Their directories
  are `<run>/<prefix>_<name>_w<k>`. Closing the pool closes every client.
  - Reuse `orchestrator.dynamics_rl.simulator()` to construct each client.
  - All clients use the same parameters; a pool never mixes frictions.
- `ThreadedVecEnv(envs)` is an SB3 `VecEnv` over `RemoteEnv`s.
  - `step_wait` runs every `env.step` in a `ThreadPoolExecutor`. The RPCs block on sockets,
    so threads give real parallelism.
  - It auto-resets exactly like `DummyVecEnv`, setting `terminal_observation` and
    `TimeLimit.truncated`.
  - Exceptions propagate. There is no per-env recovery.
- `run_episodes(clients, layouts, policy, *, deterministic)` takes layouts from a shared queue,
  with one thread per client. Each thread runs `orchestrator.dynamics_rl.run_episode`-equivalent
  logic.
  - It returns one record per layout, in layout order, with fields `layout`, `success`,
    `complete`, `actions`, `error` and `episode_dir`.
  - An episode exception is written to that episode's record, then re-raised after the other
    threads finish their current episode.
  - `policy.predict(obs, deterministic)` must support `deterministic=False` (§5.4), so do not
    call `run_episode` unchanged if it hard-codes `deterministic=True`. Factor out the minimum,
    or write the loop in this module.

## 5. Learner (`fixability.py`)

### 5.1 Policy `FixabilityPolicy(ActorCriticPolicy)`

- **Actor:** MLP 49 → 256 → 256 → 7 with ReLU. Its input is `obs[:, :49]`. The mean head is
  linear (unsquashed). SB3's normal clipping to [−1, 1] applies when stepping the environment.
- **Critic:** a separate MLP, 57 → 256 → 256 → 1 with ReLU, on the full observation.
- **Standard deviation:** state-independent and **fixed**: `log_std = config.log_std` (7
  values), `requires_grad=False`. The exploration noise is part of the experimental condition,
  identical in S and A (§1).
- **Optimiser:** Adam with two parameter groups, actor and critic, with learning rates
  `actor_learning_rate` and `value_learning_rate`. `AdaptationPPO._update_learning_rate` shows
  the pattern.

### 5.2 `FixabilityPPO(PPO)`

Collapse safeguards. Each is required, and each has a test.

1. **Value warm-up.** The first `value_warmup_rollouts` rollouts after constructing or loading
   the learner update only the critic, with the MSE between value and GAE returns over
   `n_epochs`. This covers the random critic after BC and the dynamics change at the start of
   stage A.
2. **SB3 `target_kl`** early stopping inside an update.
3. **Exact-KL backtracking.** Snapshot the actor parameters and the old mean on the rollout's
   observations, then run `super().train()`.
   - Compute the exact mean-shift KL `Σ_d (μ_new − μ_old)² / (2σ_d²)`, averaged over the
     rollout. σ is fixed, so this is the full KL.
   - While it exceeds `max_update_kl`, halve the actor step back toward the snapshot. Allow at
     most 10 halvings, then revert fully (fraction 0).
   - Log `exact_kl` and `step_fraction`. `AdaptationPPO.train` already implements this, so
     reuse the idea without the robust-column mask.
4. Advantage normalisation, `clip_range` and `max_grad_norm`, from SB3.
5. **Reward scaling.** The learner multiplies environment rewards by `reward_scale`, using a
   `VecEnv` wrapper or `step_wait`. Episode records keep the raw reward.

`rl-metrics.jsonl` gets one row per rollout. Each row has `online_steps`, `value_only`,
`policy_loss`, `value_loss`, `approx_kl`, `clip_fraction`, `explained_variance`, `exact_kl`,
`step_fraction`, the rollout's completed episodes and successes, and `elapsed_seconds`.

### 5.3 BC

- `bc_train(observations, labels, config) -> FixabilityPPO`.
- The loss is the MSE between the actor mean and the teacher's clean action label.
- It runs `bc_updates` Adam steps at `bc_learning_rate` with batch `bc_batch_size`, uniformly
  over all successful demonstration transitions.
- The critic is left untrained (warm-up handles it).
- Write `bc.jsonl` every 1,000 updates.

### 5.4 Exported actor

- `save_network(model, directory, metadata)` writes `training.zip` (the SB3 learner),
  `policy.pt` and `network.json`.
- `policy.pt` holds:
  - `format: "dynamics-fixability-actor-v1"`;
  - the actor state dict;
  - `log_std`;
  - `actor_columns: 49`;
  - `metadata`.
- `load_actor(path)` returns an object with `predict(obs, deterministic)`. The deterministic
  output is the clipped mean. The stochastic output is the clipped `mean + σ·ε`, drawn with
  `np.random.default_rng(seed)`, where `seed` is passed when the object is constructed.

## 6. Demonstrations (`demos`)

- The teacher is `R1Teacher`, unchanged, at friction 1.0 only (the CLI rejects anything else).
- There are `demonstration_attempts` planned layouts from split `fix_demonstration`. They run
  in parallel on the pool, without retrying a failed layout.
- **Noise injection (DART-style).** For attempt *i*, the noise level is
  `demonstration_noise[i % len(demonstration_noise)]`. Each level is a pair
  `[position_m, rotation_deg]`.
  - At each step the *executed* action is the teacher's action plus iid Gaussian noise. The
    noise is added to the position dims (`position_m` in metres, converted to normalised
    units per axis using `ACTION_LOW` and `ACTION_HIGH`) and to the rotation-vector dims
    (`rotation_deg`, converted to the normalised rotation units, radians/π). The result is
    clipped to [−1, 1].
  - The gripper dimension is never perturbed.
  - The *recorded label* is the clean teacher action. The noise RNG is seeded from the layout
    seed.
- Per-attempt artifacts:
  - `transitions.npz`, holding the executed actions as `actions` and adding a `labels`
    array;
  - the episode summary;
  - the noise level.
- `demonstrations.json` lists every attempt with its outcome.
- BC uses only complete, successful attempts.

## 7. `train`

Arguments: `--config`, `--init <training.zip or BC directory>`, `--friction-scale f`,
`--run-dir`, `--gpu`.

1. Load the learner from `--init`. Write `config.json`, `run.json` (init path, friction, start
   time) and the layout plans.
2. Open the pool at `{"friction_scale": f, "mass_scale": 1.0, "restitution": 0.0}`.
3. **Round 0.** Evaluate the initial network deterministically on `dev_episodes` `fix_dev`
   layouts. Save it as `networks/R00/`.
4. For round r = 1 … `max_rounds`:
   - Train `rollouts_per_round × num_envs × n_steps` actions.
   - Online layouts come from split `fix_online`, seeded by (config seed, round, env index), so
     each env walks its own layout sequence.
   - Partial episodes at the round end are closed as incomplete, as `train_round` does.
   - Save `networks/Rrr/`, then evaluate it deterministically on the same `fix_dev` layouts,
     using the same pool.
   - Append a row to `rounds.jsonl` with `round`, `online_episodes`, `online_successes`,
     `online_success_rate`, `dev_successes` and `elapsed_seconds`.
   - **Stop early** after a round where `dev_successes == dev_episodes` **and**
     `online_success_rate ≥ stop_online_success_rate`. The online rate is measured on the
     round's stochastic training episodes, counting complete episodes only.
5. **Selection** (`selection.json`): choose the network with the most `dev_successes`, then
   the highest round `online_success_rate` (round 0 counts as −1), then the earliest round.
   Write the selected id and the full table.
6. Stop on `max_train_hours` (checked between rounds and between rollouts), with the status
   recorded. A stopped run still selects among the networks it completed.

The pool stays alive across rounds. Evaluation uses `run_episodes` on the pool's clients
between training phases. The `VecEnv` must not hold an active episode when that happens:
close partial episodes first. Starting the next round resets the `VecEnv`.

## 8. `evaluate` and `feasibility`

- **`evaluate`.** Arguments: `--config`, `--network <dir with policy.pt>`, `--friction-scale`,
  `--episodes` (default 50), `--stochastic` (sample with the network's σ, seed = config seed),
  `--render K`, `--run-dir`, `--gpu`.
  - It evaluates on the first `episodes` layouts of split `fix_eval` with the pool in `state`
    mode.
  - Then, with `--render K > 0`, it runs the first K of those layouts again in one `render`
    mode simulator, to record videos.
  - It writes `evaluation.json` with per-layout records, successes, the mode and the friction.
  - It accepts a BC or PPO network, or P0's `best_rl.pt` (loaded with `load_policy`, which is
    deterministic only), so P0 can be compared on the same layouts.
- **`feasibility`.** Arguments: `--config`, `--friction-scale`, `--carry-time-scale s`,
  `--episodes` (default 20), `--run-dir`, `--gpu`.
  - It runs `R1Teacher` on `fix_dev` layouts with the step counts of the `lift`,
    `carry_and_tilt` and `lower_to_plate` moves multiplied by `s` (rounded).
  - Build the scaled moves in this module, for example with a subclass whose `__init__`
    replaces `self.moves`. Do not edit `teacher.py`.
  - The teacher's total must stay ≤ 500 actions; reject an `s` that exceeds it.
  - The results are reference only and never become training data.

## 9. Configuration (`tasks/DynamicsRL/fixability.example.json`)

```json
{
  "simulator_python": ".venv-robodojo/bin/python",
  "startup_timeout": 1200,
  "step_timeout": 120,
  "seed": 0,
  "num_envs": 2,
  "learner_threads": 4,
  "demonstration_attempts": 192,
  "demonstration_noise": [[0.0, 0.0], [0.005, 2.0], [0.01, 4.0]],
  "bc_updates": 50000,
  "bc_learning_rate": 0.0003,
  "bc_batch_size": 512,
  "log_std": [-3.9, -4.0, -3.5, -4.5, -4.5, -4.5, -3.0],
  "reward_scale": 0.1,
  "n_steps": 512,
  "batch_size": 512,
  "n_epochs": 5,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "clip_range": 0.2,
  "target_kl": 0.03,
  "max_update_kl": 0.05,
  "max_grad_norm": 0.5,
  "actor_learning_rate": 0.00005,
  "value_learning_rate": 0.0003,
  "value_warmup_rollouts": 2,
  "rollouts_per_round": 24,
  "max_rounds": 12,
  "dev_episodes": 20,
  "stop_online_success_rate": 0.9,
  "max_train_hours": 10
}
```

`log_std` corresponds to about 5 mm of position noise per axis, 2° of rotation and 0.05 of
normalised gripper command. `load_config` validates types and the obvious ranges once:

- positive counts;
- `num_envs ≥ 1`;
- 7 `log_std` values;
- noise levels as non-negative pairs.

Capacity measured 2026-11-02: 1 simulator ≈ 9.5 actions/s, 2 ≈ 17.9 actions/s; 3 exceed 24 GB VRAM (≈ 7.5 GB each) and time out. The overseer therefore set `num_envs` 2 and, to keep rounds ≈ 25k actions, `rollouts_per_round` 24 and `max_rounds` 12 (≈ 295k actions per stage). Every other value is frozen
before stage S and applies unchanged to stage A.

## 10. Tests (CPU, `tests/test_dynamics_fixability.py`)

Mock only `SimulatorClient`/`RemoteEnv`, for example with a tiny fake env that has the same
response schema. Write about one test per requirement:

1. The actor ignores columns 49–56: perturbing them leaves the mean unchanged. The critic
   uses them.
2. `log_std` stays fixed through BC and PPO updates.
3. Value warm-up: the actor parameters are unchanged after the first `value_warmup_rollouts`
   rollouts, and the critic changes.
4. Exact-KL backtracking: with a huge actor learning rate, the logged `exact_kl` is
   ≤ `max_update_kl` and `step_fraction` < 1. The test must fail if the backtracking is
   removed.
5. Reward scaling reaches the learner, while the episode records keep raw rewards.
6. `ThreadedVecEnv` matches `DummyVecEnv` on a deterministic fake env, including auto-reset
   and `terminal_observation`.
7. `run_episodes` returns records in layout order, and records and re-raises an episode
   exception.
8. Demonstration noise: the labels equal the clean teacher actions; the executed actions
   differ in position and rotation only; the gripper is untouched; there is no noise at level
   `[0, 0]`.
9. BC reduces the label MSE on a small synthetic dataset.
10. Exported-actor round trip: deterministic output equals the learner's clipped mean, and
    stochastic output is reproducible for a given seed.
11. The `train` round loop stops early under the §7 rule. Selection follows the §7.5 order.
12. The feasibility teacher scales exactly the three named moves, and rejects totals over
    500.

## 11. Overseer procedure (GPU; not for the implementer)

Pre-registered decision rules. Every run keeps its directory, including failures.

1. **Capacity.** Measure throughput with 1, 4 and 8 concurrent simulators, then set
   `num_envs`.
2. **S1.** Run `demos` at 1.0, then `bc`.
3. **S2.** Run `evaluate` on BC at 1.0 for 20 episodes: report only.
4. **S3.** Run `train --init BC --friction-scale 1.0`, then select the source network `S*`.
5. **Source gate**, on the 50 `fix_eval` layouts at 1.0:
   - deterministic **≥ 48/50**; **and**
   - stochastic **≥ 45/50**.

   Failing the gate means stop and report. Any changed retry is a new, labelled run, decided
   by the user.
6. **T.** Run `evaluate S*` deterministically at 0.3 and at 0.25 on the same 50 layouts, plus
   `--render 5` at 1.0, 0.3 and 0.25 for videos.
   - The **target** is the highest friction in {0.3, 0.25} where S* scores **≤ 25/50**.
   - If neither qualifies, stop and report: the source policy does not fail.
   - P0 is evaluated on the same layouts for comparison.
7. **F.** Run `feasibility` at the target with `s = 1` and `s = 3`, 20 episodes each: reference
   only.
8. **A.** Run `train --init S*/training.zip --friction-scale target` with the frozen
   configuration, then select A*.
9. **Verdict.** Run `evaluate A*` at the target on the 50 `fix_eval` layouts.
   - Deterministic **≥ 45/50** means **fixable by RL**.
   - Otherwise, report the measured improvement over S* at the target, paired by layout.
   - Also report A* stochastic at the target and A* deterministic at 1.0 (retention), plus
     `--render 5` videos at the target.
