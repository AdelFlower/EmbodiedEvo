# RL Policy Adaptation Across Dynamics Through Source Simulator Calibration

This document defines the research objective, implementation plan, and acceptance criteria for the experiment. The initial policy P0 may directly reuse an RL policy already trained in the source environment. The adaptation loop and independent evaluation will be implemented subsequently.

## 1. Research objective and experimental boundaries

The research question is: **Can an agent identify dynamics differences from policy mismatch in a target environment, calibrate the source simulator, and restore target task success through further RL training?**

The complete loop is:

**Load P0, already trained in the source environment → deploy to the fixed target → have the agent analyze the mismatch → modify source simulator parameters → continue fine-tuning the RL network → redeploy to the target → freeze and evaluate independently.**

- Policies in both the source and target environments are neural networks that produce an action from the observation at every step. The agent analyzes experiments and calibrates the simulator, ultimately submitting simulator parameters, a network checkpoint, and diagnostic evidence.
- The agent may write code for data analysis and experiment orchestration. Scripted control policies, teacher takeover, and manual trajectory modifications are not allowed in the formal adaptation loop.
- The final task objective is success in the target environment. Initial source performance establishes P0's task competence; adapted policies are not required to retain their original source performance.
- The first version uses one task and one target with hidden dynamics, fixed by the researcher before formal adaptation, to validate the complete loop. This is a sim-to-sim experiment modeling a sim-to-real problem. It does not establish transfer to a real robot or generalization across types of dynamics changes.
- Reuse EmbodiedEvo's agent execution, isolation, and logging mechanisms in an independent experiment. Preserve the original benchmark, `workspace-session-v1`, its four primitives, and the Playground/Test knowledge transfer rules.

## 2. Environments, initial policy, and permissions

### Environment setup

Reuse the existing cube-to-plate task, native IK, 25 Hz control frequency, 500-action episode limit, and existing success checks. Success requires a valid grasp, airborne transport, release, and stable placement without robot support.

Randomize the cube and plate XY coordinates independently and uniformly within ±3 cm of their nominal positions. Apart from the specified dynamics difference, keep assets, control interfaces, initial-state distributions, and scoring rules identical.

| Environment | Purpose | Parameter permissions |
|---|---|---|
| Nominal source S0 | Prepare P0, provide initial behavior references, and run the fixed-parameter control | Nominal configuration remains fixed |
| Calibrated source Sθ | Test dynamics hypotheses and train subsequent networks | The agent may modify parameters within the permitted scope |
| Target T | Provide deployment feedback and final evaluation | Parameters are fixed in advance by the host and cannot be modified by the agent |

### Source and target parameters

**Set source friction to `1.0` and friction in the manually constructed target failure environment to `0.25`.**

| Setting | Nominal source S0 | Constructed target failure environment T |
|---|---|---|
| Cube friction multiplier `friction_scale` | `1.0` | `0.25` |
| Actual cube static / dynamic friction | `1.0 / 1.0` | `0.25 / 0.25` |

Only the cube's static and dynamic friction differ. Both environments use a cube mass of `0.35 kg` and restitution of `0.0`; all other physics settings, layout distributions, control interfaces, and success checks remain identical.

The researcher constructs and fixes the target in advance. Its true dynamics parameters are hidden from the agent. The agent may adjust only the calibrated source environment; it cannot select or modify the target.

### Editable parameters in the calibrated source

The agent may modify three parameters in the calibrated source environment:

- `friction_scale`: scales both static and dynamic cube friction.
- `mass_scale`
- `restitution`

Changes take effect only before a new experiment or training round and remain fixed within an episode. They must affect the actual PhysX collision materials and mass properties, with verification performed by the host.

### Initial network P0

**An existing RL policy trained in the source environment at friction `1.0` may be used directly as P0 in the formal experiment.** Collecting new demonstrations or retraining the source policy is not required.

Freeze P0's network weights, normalization parameters, and observation/action configuration, and record its training provenance on the host. Use frozen inference weights for deployment and the corresponding full training checkpoint for continued fine-tuning. The calibration branch and the fixed-source-parameter control start from the same P0 and training state.

P0 must have succeeded in at least 45 of 50 independent source layouts. Existing source evaluation records may be reused when they correspond to the same frozen policy, task, and scoring settings; perform additional evaluation only when matching records are unavailable. This is an initial competence check, not a source-performance retention requirement for subsequent policies.

### Information available to the agent

The agent receives actions, object and robot state trajectories, raw videos, and task outcomes. It may compare behavior across environments with the same initial layout.

Keep true target parameters, collision material metadata, private rewards, internal scoring state, simulator source code, and historical records containing the answer on the host. The experimental agent starts with an independent context and does not receive the target parameters disclosed in this research document.

## 3. Adaptation loop, training, and control condition

### Agent adaptation process

1. **Provide a shared baseline.** Run P0 in both the nominal source and target environments on the same five fixed development layouts—the initial cube and plate placements—and provide paired trajectories, videos, and task outcomes.
2. **Form hypotheses.** The agent records hypotheses about the dynamics difference, supporting evidence, competing explanations, and predicted observations.
3. **Test in the source.** The agent modifies source parameters within the permitted scope and runs diagnostic rollouts with existing frozen RL networks to determine whether behavior more closely matches the target. Test at least one competing hypothesis.
4. **Continue fine-tuning.** Train in the source environment with the calibrated parameters, continuing from the previous round's network to produce the next network.
5. **Deploy to the target.** Run each round's candidate on the same five target development layouts. Use the feedback to inform the next parameter decision.
6. **Select and freeze.** Select the candidate with the most target development successes, breaking ties in favor of the earlier candidate. Include P0 among the candidates so that a degraded policy need not be selected.

Entry into the experiment does not require P0 to fail all three of three initial trials. Preserve outcomes with no observed mismatch, no adaptation needed, failed adaptation, and successful adaptation. Do not modify the fixed target during formal adaptation to manufacture failures.

### Rules for continued fine-tuning

- The researcher freezes the training procedure, network architecture, observation encoder, action decoder, reward, and hyperparameters before the formal experiment. The agent cannot modify them.
- Each round inherits the preceding round's actor, critic, target networks, optimizers, and cumulative training counters. Final candidate selection does not alter this continuous training chain.
- Clear replay data at the beginning of every round in both branches. Do not mix transitions from previous dynamics or original demonstrations into the new round's critic replay.
- Disable the original behavior cloning (BC) auxiliary loss and demonstration sampling during adaptation. Perform SAC updates using actual transitions collected in the current calibrated source environment; do not invoke the teacher.
- Use target data only for diagnosis and candidate selection. Do not use it in gradient updates, demonstration datasets, or replay buffers.
- Keep the network's two hidden layers of 256 units each and the shared observation/action interfaces fixed throughout the formal experiment. The training service must save full training state and export standalone inference weights.
- Keep the native success signal distinct from the training reward. Reward shaping and termination handling are fixed by the host and cannot be modified by the agent.

### Control condition and budgets

Include a control that continues RL training with fixed source parameters. It starts from the same P0, keeps source parameters at their nominal values, and matches the calibration branch's actual training rounds, action count, update count, target query count, and candidate selection rule.

Each adaptation branch is limited to:

- Eight hours in total, including analysis, diagnosis, training, and development validation.
- At most four training rounds, with at most 25,000 online actions per round and 100,000 online actions in total.
- Five target validation episodes per round, with at most 20 in total. Record the shared initial baseline separately.
- At most 20 diagnostic rollout episodes in the calibrated source, counted against the branch's time budget.

Account separately for P0 preparation, training procedure development, trainability checks, and final evaluation. If time limits prevent matched budgets, disclose the actual resource differences rather than claiming a strictly matched-budget comparison.

## 4. Repository changes and interfaces

Add an independent `orchestrator.dynamics_adaptation` entry point with `preflight`, `run`, `evaluate`, and `summarize`. Every invocation must explicitly provide a `--run-dir` whose name begins with `YYYYMMDDHHMM`.

`preflight` checks software interfaces and simulation physics consistency. `run` starts by loading P0 and providing the shared baseline. Neither includes the additional trainability check arranged separately by the researcher, nor requires its result as a startup input.

The agent uses a restricted experiment client for the following operations:

- `probe`: submit source parameters, an existing network ID, a hypothesis, and a prediction to obtain a diagnostic rollout.
- `train`: submit the next round's source parameters, run the fixed training procedure, and receive a network ID.
- `deploy`: run a registered network on the target development layouts and return raw feedback.
- `status`: query budgets and completed experiments.
- `submit`: submit the final network ID, its associated source parameters, and diagnostic evidence.

These are independent experiment service interfaces, not additional robot primitives. The agent does not directly send robot actions; the host deploys only registered neural networks.

Implementation priorities:

- Extend the existing RL adapter to support permitted parameter changes in the calibrated source. Enforce target immutability in the host service.
- Explicitly fix `friction_scale = 1.0` for the source and `friction_scale = 0.25` for the target in the new adaptation entry point's host configuration.
- Retain the host-side `.venv-rl`, private Unix socket, and simulation worker in the existing image. Do not modify `vendor/`.
- Support continued training from checkpoints, replay clearing between rounds, target data isolation, and cumulative budget accounting.
- Training may disable camera capture. Diagnostic rollouts and final evaluation must provide raw videos; first verify that both modes preserve the same physics and scoring behavior.
- Preserve the old code-policy repair and reference policy archives as historical records. Do not use them as part of the formal adaptation workflow or as agent knowledge.

## 5. Validation, acceptance, and deliverables

### Software and physics validation

Run CPU validation with `python -m unittest discover -s tests -v`. Cover parameter permissions, target immutability, action and observation consistency, training-chain restoration, replay clearing, target data isolation, actual parameter updates, checkpoint saving and loading, cumulative budgets, and preservation of failed attempts.

GPU validation checks effective materials and mass, restoration after randomized resets, consistency between rendering modes, and matching source/target behavior under identical parameters. CPU mocks cannot establish these physical conclusions.

### Additional pre-experiment check: RL trainability

**Conduct this check separately before the formal experiment; it is not part of the agent framework design or agent workflow.** Set the source simulator to the true target parameters—cube static/dynamic friction `0.25/0.25`, with all other settings unchanged—and check whether the fixed RL training procedure can learn an effective policy starting from P0.

This check distinguishes the training procedure's capability from the agent's diagnostic and calibration capability. It does not select or modify the target environment. Do not provide its parameters, trained policies, trajectories, or logs to the experimental agent, or use them as initialization or training data for formal adaptation.

Record its cost and results separately from the agent's budget and performance. A failed check is a result of the training procedure validation, not an agent diagnosis or adaptation failure.

### Final target evaluation

Freeze all candidate selections, networks, normalization parameters, action configurations, and diagnostic conclusions before generating 50 target layouts unused in training or selection.

Evaluate each of the following on the same set of layouts:

- Initial P0.
- The selected network from the agent calibration branch.
- The selected network from the fixed-source-parameter control.

Every attempt uses a fresh simulator and runs only the frozen network. Within-episode reset, online updates, and teacher takeover are prohibited. Allow approximately three to four additional hours per policy and report the actual elapsed time.

**The calibration branch must achieve at least 45 native successes out of 50 attempts to meet the target task acceptance criterion.** There is no requirement for success in the original source environment.

Retain every attempt. Mark crashes, timeouts, and missing reports separately while keeping them in the denominator of planned attempts. Do not delete or automatically rerun them.

### Research conclusions and deliverables

Report two questions separately:

1. **Was task performance restored?** Report target success counts, per-episode results, and actual costs for P0, the calibrated policy, and the control policy.
2. **Is the diagnosis supported?** Assess whether predictions recorded before experiments, parameter interventions, competing hypotheses, and behavioral evidence support the conclusion. An independent host-side review labels it supported, ambiguous, or unsupported. Task success does not automatically establish a correct diagnosis.

Deliver the complete replacement plan, training and execution code, source parameter records, full training checkpoints, standalone inference weights, training curves, per-episode trajectories, videos, diagnostic evidence, and reproducible commands. Store all runtime artifacts under the corresponding `runs/` directory and retain failed results alongside successful ones.
