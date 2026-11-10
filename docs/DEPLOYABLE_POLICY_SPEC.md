# Deployable pick-and-place policy (DP) — engineering spec

Status: approved design decisions from the user on 2026-11-09.
Owner: Claude (spec, review, GPU stages); Codex (implementation of milestone DP1).

## 1. Goal

Retrain the cube-to-plate network policy so that its **actor observation contains
only quantities a real robot can measure**, while keeping the task, action decoder,
25 Hz control, 300-step horizon, physics and success rules of the current network
runner. A programmatic script is used only to build a trajectory prior (demos → BC);
PPO then improves the network.

User decisions (2026-11-09):

| Question | Decision |
|---|---|
| Object perception | One-shot detection: cube and plate XY detected once at reset from the head camera |
| Variation | Random layouts (cube ±3 cm XY; plate x ∈ [0, +3] cm, y ∈ [−3, 0] cm — amendments A1/A2, §5), nominal physics (friction 1.0, mass 1.0, restitution 0) |
| Workflow | Claude writes this spec; Codex implements DP1; Claude reviews and runs GPU stages |

The FrictionAgent case study (`orchestrator/experiment*.py`, its 33-dim observation,
budgets, protocol) is **not changed**. DP is a separate pipeline that reuses the
episode runner, decoder, PPO and simulator transport through small, default-preserving hooks.

## 2. Privileged vs deployable signals

Current actor input (`integrations/friction_rl/network.py:observe`, 33 dims) uses:

| Signal | Status |
|---|---|
| clock features (11), commanded pad position (3) | deployable (controller state) |
| gripper opening (1) | deployable (gripper joint encoder, `gripper_opening.right`) |
| cube position relative to pad, every step (3) | **privileged** (simulator ground truth) |
| plate position relative to pad (3) | **privileged** (ground truth) |
| cube clearance, quaternion, linear velocity (8) | **privileged** |
| finger contacts (2), table contact, plate contact | **privileged** (PhysX contact reports) |

Measured grip force (`right_grip_force`) is computed from contact impulses and is
also privileged; it is not used.

## 3. Observations

### 3.1 Actor (deployable), `ACTOR_OBS_DIM = 21`

```
clock(11)        t/300, sin/cos(2π·{1,2,4,8,16}·t/300)          # identical to observe()
pad_cmd(3)       (pad_cmd − [.15,−.15,.85]) / [.2,.2,.1]          # identical to observe()
cube_det(3)      (ĉ − pad_cmd) / .05,   ĉ = [ĉx, ĉy, TABLE_Z + CUBE_HALF]
plate_det(3)     (p̂ − pad_cmd) / .3,    p̂ = [p̂x, p̂y, PLATE_Z]   (PLATE_Z = 0.7828, nominal)
opening(1)       state["sample"]["gripper_opening"]["right"]
```

`ĉx, ĉy, p̂x, p̂y` are the one-shot detection made at reset (§4) and held fixed
for the episode. No previous action, measured-minus-commanded pose or opening rate
is included: the FT5 BC history showed lag-type inputs make BC copy its own
previous output (spec history: FT5 "lag input" fix).

**Signature-level guarantee.** `actor_observe(t, pad_cmd, opening, detection)`
takes explicit arrays only — never the simulator state dict. The observer
extracts exactly `gripper_opening.right` from the state for the actor.

### 3.2 Critic (training only, privileged), `CRITIC_OBS_DIM = 54`

`concat(actor_obs (21), observe(state, t, pad_cmd) (33))`. The critic is never
used at deployment, so asymmetric actor–critic is permitted.

## 4. One-shot detector

### 4.1 Image detector (deployment and validation)

Input: the first head-camera RGB frame of an episode (`cam_head`, 640×480,
pinhole, fixed pose `[0, −0.41, 1.308]`, pitch 30°), before any action.

1. Classify pixels: **red** (cube) = high saturation, hue near red, value above a
   threshold; **white** (plate) = low saturation, high value. Thresholds are
   constants in `detector.py`, chosen by Codex from the reference frames and
   checked by the calibration gate.
2. **Workspace ROIs** (overseer revision 2026-11-09, replacing a ≥90%-frequency static
   mask, which would also have masked the plate: the plate is ~60 px across but a ±3 cm
   shift moves it only ~±16 px). Fixed pixel boxes cover each object's reachable
   layouts plus margin and exclude the robot's red accents and white housings at its
   fixed initial pose: cube `(380, 215)–(466, 286)`, plate `(250, 175)–(371, 281)`.
3. Detection = centroid of **all** class pixels inside the object's ROI (robust to the
   plate's ridge pattern splitting the white region); fewer than 40 (cube) / 400
   (plate) pixels is a failed detection.
4. Map centroid → table XY with a **per-object homography** (cube: centroid → cube
   centre XY; plate: centroid → plate centre XY), least-squares/DLT fit on the
   calibration fit set. A plane homography is exact for a pinhole camera; fitting
   a separate one for the plate absorbs its 3-D bowl appearance.

Real-robot analogue: a one-time camera calibration plus colour segmentation. The
sim ground truth is used **only** to fit and validate this calibration, never at
policy runtime.

### 4.2 Simulated detector (training, demos, state-mode evaluation)

State mode disables cameras, so training uses
`ĉ_xy = c_xy + clip(N(0, σ_c² I), ±3σ_c)` and `p̂_xy = p_xy + clip(N(0, σ_p² I), ±3σ_p)`,
drawn **once per episode** from the episode's own RNG stream. Ground truth at reset
is read in exactly one function, `simulated_detection(state, rng, sigma)`.

`σ_c, σ_p` come from the calibration rule (pre-registered):
`σ = max(1.5 × validation RMS error per axis (max over x, y), floor)`, floors
2 mm (cube) and 3 mm (plate).

## 5. Layouts and seeds

`layout_radius = 0.03`, `cube_shift = [0, 0]`, `plate_box = [[0, 0.03], [−0.03, 0]]` (amendment A2):
cube offset uniform ±3 cm XY, plate offset uniform in the box (x range, y range) around
the nominal scene, sampled by `integrations/deployable/layout.py::build_scene`.

**Amendment A1 (2026-11-10, user-approved).** The original plate range was ±3 cm XY.
The demo gate failed (`runs/202611091650_dp_demos` (deleted): strict 54/64, clean k=2 not strict).
Diagnosis: for plate offsets toward −x / +y the right arm cannot track the scripted
carry/lower poses (cmd–observed lag up to 111 mm, then a 20–71 mm one-step snap that
flips the cube). In that run, plates inside the new box scored 28/29 strict versus
26/35 outside. The plate range was therefore restricted to the reachable box; gates
and all other settings are unchanged. The calibration (`runs/202611091633_dp_calibrate3`)
was fit on the ±3 cm superset and is reused; the ROIs still cover every layout.
Stage S and the demos are rerun on the new layouts.

**Amendment A2 (2026-11-10, user-approved).** A1 used `plate_box = [[−0.01, 0.03], [−0.03, 0.01]]`.
Stage S passed (`runs/202611101703_dp_a1_script_dev2` (deleted), strict 20/20), but the demo gate failed
again (`runs/202611101711_dp_a1_demos` (deleted): strict 58/64, clean k=1 dropped), by the same lag-and-snap
mechanism. Pooling both demo runs (128 episodes): the lag occurs only when the commanded pad
lowers at world x < −0.006 and y > −0.09. The plate box was narrowed to x ∈ [0, +3] cm,
y ∈ [−3, 0] cm, where all 58 prior episodes were strict (max lag 1 mm).
**Disclosure:** this box was chosen after seeing those results. It covers one quarter of the
original ±3 cm plate area. Gates and all other settings are unchanged; stage S and the demos
are rerun again.

**Result (2026-11-10).** All gates passed under A2: stage S 20/20, demos 64/64 strict,
BC dev 20/20. PPO ran 80 iterations; every dev check was 20/20, so selection chose
iteration 0, identical to the BC policy. Acceptance: 50/50 strict on stream 38 and
20/20 strict on stream 39 (image detector). Retained runs: `202611091633_dp_calibrate3`,
`202611101751_dp_a2_script_dev`, `202611101759_dp_a2_demos`, `202611101816_dp_a2_bc`,
`202611101817_dp_a2_train` (selected `policy_it000.pt`), `202611102129_dp_a2_selected_eval`,
`202611102143_dp_a2_selected_render_eval`. At the user's request all other runs,
including the failed and diagnostic attempts cited above, were deleted on 2026-11-10.
Layout seed for `(stream, k)` is
`int(np.random.SeedSequence([cfg.seed, stream, k]).generate_state(1)[0])`, passed
to `build_scene`. Each episode's detection/action RNG uses `[cfg.seed, stream, k, 1]`.

| Stream | Use | Count |
|---|---|---|
| 20 | calibration fit | 40 |
| 21 | calibration validation | 40 |
| 30 | demonstrations | 64 |
| 31 | PPO training episodes (k = iteration·8 + j) | 8 / iteration |
| 37 | development layouts | 20 |
| 38 | final evaluation, simulated detector | 50 |
| 39 | final evaluation, image detector (render) | 20 |

Physics everywhere: `SOURCE_PARAMETERS` (nominal).

## 6. Trajectory prior (script → demos → BC)

- The existing `PrimitivePolicy` with the P0 script parameters
  (`tasks/DynamicsAdapt/p0.example.json` `script`) and 15 N force-mode grip, but it
  receives a **detection-substituted state**: `cube_position = ĉ` (with nominal z)
  and `plate_position = p̂`. The script therefore uses only the actor's information.
- Demos: 64 episodes on stream-30 layouts; episodes 0–7 have no motion noise, the
  rest DART AR(1) noise `[2, 2, 1] mm`, ρ = 0.95 (as in P0). Labels are the
  script's clean corrective labels (`ScriptController` semantics). Saved
  `demos.npz` holds `obs` (actor), `critic_obs`, `labels`, `episode`.
- BC: existing `bc_fit` on actor obs → labels, 20k steps, batch 256, lr 1e-3,
  `init_std = [.02, .02, .02, .1, .25]`.

## 7. PPO

Reuse `ppo_update` with these default-preserving generalisations:

- `ActorCritic(obs_dim, act_dim, critic_dim=None)`; critic input `critic_dim or obs_dim`.
- `ppo_update` uses `batch["critic_obs"]` for the critic when present, else `batch["obs"]`.
- `act(..., explore=EXPLORE)` and `ppo_update(..., explore=EXPLORE)`; DP passes
  `cfg.training.explore`. The BC term on outputs `[:4]` against the demo prior stays.

Config (pre-registered): `explore = [0,1,2,3,4]`, `episodes_per_iter = 8`,
`warmup_iterations = 2` (critic only), `gamma .99`, `gae_lambda .95`, `clip .2`,
`epochs 8`, `minibatch 512`, `actor_lr 2e-4`, `critic_lr 5e-4`, `target_kl .04`,
`bc_weight 1.0`, `max_iterations 80`, `dev_every 10`.

**Reward** (host code, in-process — no Agent, so no sandbox): the strict-aware
formula of `tasks/DynamicsAdapt/oracle_reward.py` (lift and carry progress, −0.25/mm
slip, −5 on losing the cube while closed, +10 strict success, −3 lenient-only
success, −3 drop/no-lift). The reward may use privileged trace data because it is
training-only. Ported to `integrations/deployable/reward.py`; a test asserts equality
with the oracle file on fixture traces.

**Development checks**: at iteration 0 (the BC policy) and after every `dev_every`
iterations, run the 20 development layouts deterministically. Candidate selection:
most strict successes, then most lenient, then the earlier checkpoint.

**Pre-registered fallback**: if, after warm-up, the online strict success rate is
more than 25 points below the BC policy's iteration-0 online rate for 3 consecutive
iterations, stop the run (record it, keep it) and start a new run from BC with
`explore = [3, 4]`.

## 8. Stages and gates (overseer, GPU)

| Stage | Command | Gate |
|---|---|---|
| C calibrate | `deploy calibrate` (render, resets only) | 40/40 validation detections; cube error p95 ≤ 4 mm; plate p95 ≤ 8 mm |
| S script check | `deploy evaluate --script` on dev, simulated detector | strict ≥ 18/20 |
| D demos | `deploy demos` | clean episodes 0–7 all strict successes |
| B BC | `deploy bc` then `deploy evaluate` on dev | report only (PPO proceeds if ≥ 6/20 strict or online ≥ 30%) |
| P PPO | `deploy train` | — |
| E final | `deploy evaluate` on streams 38 and 39 for BC and selected | **Accept**: selected ≥ 45/50 strict (stream 38) and ≥ 18/20 strict (stream 39, image detector) |

A gate failure stops the pipeline and goes back to the user; thresholds are not
tuned after seeing results. Report strict and lenient, stop reasons, detection
errors and the BC→PPO difference.

## 9. Implementation (milestone DP1, Codex)

### 9.1 Hooks in existing code (default behaviour must stay bit-identical)

1. `run_net_episode(client, scene, episode_dir, controller, observer=None)`.
   With an observer: after reset call `observer.reset(state, episode_dir)`; each step
   `actor_obs, critic_obs = observer(state, t, decoder.pad_cmd)`; the controller receives
   `actor_obs`; the batch adds key `critic_obs`; `episode.json` adds
   `observer.record()` (detection, truth, error). Without an observer nothing changes.
2. `ppo.py`: `critic_dim`, `explore` and `critic_obs` as in §7.
3. Worker (render mode only): write the first `cam_head` frame of each episode
   losslessly to `<episode_dir>/reset_cam_head.png` before the reset reply. The
   host-side image observer reads that file.

### 9.2 New code

| Path | Content (guide size) |
|---|---|
| `integrations/deployable/observation.py` | constants, `actor_observe`, `Observer` (simulated or image detection) (~100 lines) |
| `integrations/deployable/detector.py` | pixel classification, static mask, components, homography fit/apply, `simulated_detection`, `sigma_rule` (~150) |
| `integrations/deployable/reward.py` | strict-aware reward port (~50) |
| `integrations/deployable/script.py` | detection-substituted `ScriptController` subclass (~40) |
| `orchestrator/deployable.py` | CLI `calibrate`, `demos`, `bc`, `train`, `evaluate`; config loading (~300) |
| `tasks/Deployable/config.json` | config (§5–§7, P0 script, simulator settings, detector σ path) |
| `orchestrator/__main__.py` | add `deploy` command dispatching to `orchestrator.deployable` |
| `docs/RUNNING.md` | short "Deployable policy" section with the §8 commands |

CLI rules: every command takes a new `--run-dir` with a `YYYYMMDDHHMM` prefix and
writes `config.json`, `run.json` and a terminal `result.json` (as `orchestrator/p0.py`).
`train` and `evaluate --detector simulated` read σ from `--calibration <calibration.json>`.
`evaluate` options: `--policy` or `--script`, `--stream {dev,eval,render_eval}`,
`--detector {simulated,image}` (image ⇒ render mode), `--stochastic`.
`train` writes `iterations.jsonl`, `dev.jsonl`, `policy_it*.pt`, `training.pt`,
`selection.json`; `--demos` and `--policy` (BC) are required.

### 9.3 Tests (CPU, `tests/test_deployable.py`)

- Default paths unchanged: the existing suite passes unmodified; `run_net_episode`
  without observer produces identical batches/records to before (existing fakes).
- Privilege audit: run an episode with a fake client and a `policy_controller` whose
  observer is checked: `actor_observe` output is invariant when every privileged
  field of the state (cube pose/velocity, contacts, plate position, grip force)
  is changed after reset, and changes when opening or detection changes.
- Detector: synthetic images (drawn red square / white disc on brown background plus
  static red/white "robot" blobs) → static mask excludes robot blobs; homography
  recovers a known projective map to < 0.1 mm; sigma rule.
- Simulated detection is reproducible per seed and clipped at 3σ.
- Reward port equals `tasks/DynamicsAdapt/oracle_reward.py` on fixture traces.
- Asymmetric PPO: one update runs with `critic_obs` of a different width; explore
  `[0..4]` vs `(3, 4)` changes only the log-prob dimensions.
- CLI: run-dir rules, config validation, a fake-pool `train` for 3 iterations with
  `dev_every 2` writes the listed files and a correct selection.

### 9.4 Normative rules for the implementation

- No defensive programming: no broad `try/except`, no fallback paths, no retries,
  no hash/checksum "verification", no speculative options beyond this spec.
- Validate configuration once at load; trust internal calls afterwards.
- Reuse existing functions (`run_net_episode`, `Decoder`, `ScriptController`,
  `bc_fit`, `ppo_update`, `gae`, `start_pool`, `build_scene`, `summary`) rather than copying them.
- Do not modify `vendor/`, the FrictionAgent experiment modules or the 33-dim `observe`.
- Keep modules near the guide sizes; one responsibility per module.
- CPU tests only; do not launch simulators, Docker or GPU work.

## 10. Disclosures for the final report

Sim-only evaluation; single robot/camera configuration; detector thresholds and
homographies fitted on sim frames; no camera noise, latency or lighting
randomisation; the cube's yaw is not randomised; nominal physics only; force-mode
gripper (real grippers need a force-controllable mode); training reward and
termination use privileged simulator data (training only).
