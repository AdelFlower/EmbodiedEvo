# DynamicsAdapt research protocol

This document defines the retained FrictionAgent case study. DynamicsAdapt asks
whether an Agent can diagnose a hidden dynamics difference, reproduce its effects
in a source simulator, and recover target task performance through continued RL.
The refactor changes project organization, not this experimental procedure.

## Environment and policy

The task is to lift a red cube, carry it over a plate, lower it and release it.
The current experiment uses a single fixed layout: layout radius zero and cube
shift `[0.03, 0.0]`. It is a sim-to-sim study using RoboDojo/Isaac Sim, not a claim
of real-robot transfer or generalization across layouts.

| Parameter | Nominal source | Example fixed target | Agent-editable source range |
|---|---:|---:|---:|
| `friction_scale` | 1.0 | 0.25 | [0.05, 2.0] |
| `mass_scale` | 1.0 | 1.0 | [0.25, 4.0] |
| `restitution` | 0.0 | 0.0 | [0.0, 1.0] |
| `gripper_scale` | 1.0 | 1.0 | Fixed at 1.0 |

Nominal cube static/dynamic friction is `1.0/1.0`, nominal mass is 0.35 kg, and
parameters affect the actual simulator properties. The host fixes target
parameters before the experiment and never exports them to the Agent. Omitted
source parameter keys take nominal values; parameters remain fixed within a job.

P0 is the FT5 imitation policy trained from source demonstrations, historically a
15 N scripted grasp. `p0.policy` and `p0.demos` identify local weights and source
motion-prior data. Scripts are used only for optional P0 preparation. Formal
probes, deployments and evaluation execute registered neural networks.

The policy has separate actor/critic MLPs with two 256-unit tanh layers. It receives
33 state features, including time features, commanded pad position, relative cube
and plate positions, cube clearance/orientation/velocity, opening and contacts.
Its five outputs specify pad displacement (up to 1 cm per axis per step), gripper
closure and squeeze force `25 + 20*a[4]` N while closed. The decoder clips actions
to [-1, 1] and uses a fixed orientation schedule. Control is 25 Hz with a 300-step
network episode horizon.

## Outcomes and episodes

Keep the native success signal, lenient success, strict success and reward
distinct. The current network runner's lenient success is its `success` stop:
13 consecutive supported rest steps after a release, or native successful
termination. A rest step requires plate contact, no table/robot/finger contact,
opening at least 0.8, the cube's lowest corner within the configured plate height
band, and its centre within 4.5 cm of the plate centre.

Strict success additionally requires uninterrupted two-finger contact from the
first airborne lift (clearance over 2 mm, no table/plate contact) until the first
subsequent closed-to-open command. Re-contact after losing the cube does not
qualify. The approximately 5 cm lift is the task instruction and a diagnostic
measurement; `lifted5` is not an additional strict-success predicate. Preserve
the existing stop order, including native termination, a table drop after a
5 mm lift, no such lift by step 160, and the 300-step timeout.

Every baseline and deployment uses D0–D3 on the same scene. D0/D1 use the policy
mean; D2/D3 sample exploration with seed `[seed, 7, k]`. A probe uses any nonempty
subset. Final evaluation uses 10 deterministic and 20 stochastic episodes by
default, seeded `[seed, 8, k]`. These are held-out episode streams on the same
layout, not 30 held-out layouts. Simulator pools reuse workers between episodes;
even deterministic policies are not bit-repeatable across successive resets.

## Agent experiment loop

1. Freeze the host configuration, P0 and source motion prior.
2. Run P0 in both source and target on D0–D3 and export paired raw evidence.
3. Let the Agent record hypotheses/predictions and probe source parameters.
4. Continue PPO from the latest network in the selected source, with the submitted
   reward program. Automatically deploy the resulting network to target D0–D3.
5. Repeat within budgets, then record the Agent's diagnosis and select a network.
6. Evaluate P0 and the selected frozen network; a shared selection of P0 is run once.

The public workspace client supports `status`, `probe`, `train`, `wait` and
`submit`. `train` includes automatic deployment; there is no separate `deploy`
operation. One experiment job runs at a time. Requests and refusals are retained.

Training requires a completed probe of the current parent at exactly the proposed
nominal-filled parameters. Every probed strict outcome must match that parent's
target outcome, including at least one target failure. A parent with no target
failure or no deployment record cannot be trained. The Agent must record a
prediction before each probe, compare failure modes, and test a competing
hypothesis; matching binary outcomes alone does not establish a diagnosis.

Candidates are N000 and deployed networks. The host selects most strict development
successes, then most lenient successes, then the earlier network. The Agent's
preferred network is recorded but does not override this rule. There is no matched
fixed-source control branch in this case study.

Default limits are 12 hours, four rounds, 4–60 iterations per round, 160 iterations
total, 16 target deployment episodes and 24 diagnostic episodes. The shared
baseline is separate. Accepted training/probe jobs consume their budgets at
start, including failed jobs; target episodes are charged when deployment starts.
Refusals consume no experiment count but take wall time. At the deadline the
active job stops and its candidate is removed from the network table; its files
remain as evidence.

## PPO and reward programs

Each iteration collects eight stochastic source episodes. PPO explores the
gripper and force outputs; a BC term on the frozen P0 source demonstrations keeps
motion close to the prior. Every round continues the latest model, both Adam
optimizers and cumulative iteration/episode counters; `log_std` is not reset.
Starting from N000 creates the optimizers. The first two iterations are
critic-only when starting from N000 or when the reward program text changes.

Architecture, optimizer settings, decoder, termination and success checks are
host-controlled. The Agent chooses only source parameters, reward program and
iteration count. Target observations support diagnosis and candidate selection;
they never become PPO transitions, demonstrations or gradient-update data.

Reward programs define `reward(trace)` and return one finite value per step with
magnitude at most 10,000. Programs are limited to 64 KiB and may use NumPy and the
public `frictionlib`. The host snapshots the text, dry-runs it on four baseline
source trajectories, and runs it in a separate Docker sandbox with no network,
one CPU, 1,024 MiB memory and 120 seconds per batch by default. Only output is
writable. Program errors reject the dry run or fail an accepted training job;
they do not trigger automatic retries. Reward changes cannot change termination.

The supplied default reward rewards increased held clearance and progress toward
the plate, penalizes slip, adds 10 on lenient success, and subtracts 3 on drop or
no-lift termination. Its exact implementation is the public
`Workspace/scripts/default_reward.py`.

## Isolation, artifacts and conclusions

The Agent sees instructions, nominal physics, whitelisted object/robot/contact
trajectories, outcomes, raw videos and exported training metrics. Instructions,
scripts, nominal values and observations are read-only; notes, rewards and
submission are writable. Target parameters, material metadata, weights, simulator
sources, private host state, historical answer-bearing runs and oracle evidence
remain outside its workspace. No model summarizes observations or videos.

The separate host-only `trainability` command trains from P0 with researcher
parameters and reward, then evaluates on the target. Its costs and results are
separate from Agent performance and its outputs never enter the Agent workspace.
It is not a matched control. The scripted smoke Agent deliberately knows the
target; it validates execution only.

Acceptance is at least 27 strict successes out of the 30 configured final
episodes. Report strict and lenient outcomes, native records where available,
all planned attempts, actual budgets, timing and model usage. Preserve failed,
partial and unstarted episodes. Crashes, timeouts and missing reports remain
unclassified unless causal evidence identifies the failure; confirmed
infrastructure reruns need separate records preserving the original attempt.

Report performance recovery separately from diagnosis support. Assess predictions,
parameter interventions, competing explanations and raw behavior; success does
not by itself prove the inferred dynamics. Disclose the single scene, simulated
state/contact access, source demonstration prior, restricted action exploration,
pool reuse, absent matched control and any untested hypotheses.
