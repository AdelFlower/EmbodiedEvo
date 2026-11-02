# Restore target success through source calibration

P0 (N000) succeeds in the nominal source but may fail in a target whose dynamics
differ in unknown ways. Restore target success by calibrating the source simulator
and continuing RL training there. You never send robot actions: the host runs only
registered networks.

## Policy interface

The policy is a state-based MLP. Its effective inputs are cube position, plate
position, right end-effector position, plate-minus-cube, and remaining action
allowance. Its output sets the absolute right end-effector target pose and gripper
opening at 25 Hz. Episodes have 500 actions. Training uses PPO with fixed small
exploration noise and a per-update trust region, so each round changes the policy
gradually. Architecture, reward, training
algorithm and hyperparameters are fixed and cannot be changed.

## Source parameters

`friction_scale` multiplies the cube's static and dynamic friction, in [0.05, 2.0].
`mass_scale` multiplies its mass, in [0.25, 4.0]. `restitution` sets its coefficient
of restitution, in [0, 1]. See `nominal.json` for nominal values and ranges.
Omitted parameters take nominal values. Parameters stay fixed throughout each job.

## Operations, budgets and selection

Use only `scripts/adapt.py` for experiments:

```bash
python scripts/adapt.py status
python scripts/adapt.py probe --network N000 --parameters notes/p.json --layouts 0,2 --hypothesis "Explanation being tested" --prediction "Expected observable change"
python scripts/adapt.py train --parameters notes/p.json --actions 2000 --rationale "Why train in this source"
python scripts/adapt.py wait J0001 --seconds 600
python scripts/adapt.py submit submission/diagnosis.json
```

Probes run frozen registered networks on a nonempty subset of the development
layouts. Training continues from the latest successfully trained network, then
automatically deploys on all {{DEVELOPMENT_LAYOUTS}} target development layouts.
One job may run at a time. `wait` blocks for up to `--seconds` (1–900); set your
shell timeout longer than that value. Status queries consume nothing.

Limits: {{WALL_HOURS}} hours, {{ROUNDS}} training rounds, {{ACTIONS_PER_ROUND}}
actions per round (minimum 2,000, in multiples of {{N_STEPS}}), {{ACTIONS_TOTAL}} total training actions,
{{TARGET_EPISODES}} target episodes, and {{DIAGNOSTIC_EPISODES}} diagnostic episodes.
Accepted attempts consume their budgets even when they fail; target episodes are
charged when deployment starts. No experiments are accepted after submission.

The host selects the network with most target development successes. Ties go to
the earlier candidate; P0 is included. Incomplete episodes count as failures.
Your preferred network is recorded but does not determine selection.

## Data

Read `observations/baseline/source/L0...` and `observations/baseline/target/L0...`.
Job outputs are under `observations/jobs/J0001/`: your accepted `request.json`,
`episodes/L0...` for probes, or `training/` and `deploy/L0...` for training.
Episode outputs include `result.json`, three raw `cam_*.mp4` videos when available,
and `trajectory.npz`. State arrays have T+1 rows, including the initial state;
action arrays have T rows. Fields are `step`, `action`, `applied_action`,
`right_ee_pose`, `right_arm_joint_position`, `right_arm_joint_velocity`,
`right_gripper_joint`, `right_gripper_opening`, `right_finger_center`,
`cube_position`, `cube_orientation`, `cube_linear_velocity`,
`cube_angular_velocity`, `plate_position`, `plate_orientation`.
`action` is the normalized network output. `applied_action` concatenates left pose,
right pose, left gripper joints, right gripper joints. Training exports contain
`training.json`, `online-episodes.csv` and `losses.csv`. Failed or unstarted episodes
may lack trajectories or videos.

## Success criteria

{{SUCCESS_CRITERIA}}

## Rules

**Reproduce before training (enforced).** Training at parameters θ is refused unless
a completed probe at exactly θ, of the network the round will continue from,
matched that network's target-development success or failure on every probed
layout and included at least one of its target failures. Check failure modes,
not only outcomes, in your analysis. `status` lists `target_dev_outcomes` for each
network; `null` means it has no deployment record. A network without a target
failure or deployment record cannot be trained further.

Record a hypothesis and prediction before every probe. Test at least one competing
hypothesis. Target data is for diagnosis and selection only. Keep failed attempts;
never automatically retry. Write analysis code and notes under `notes/` and your
diagnosis under `submission/`. Instruction, scripts, nominal values and observations
are read-only. Finish with `submit` after jobs have ended.

## Diagnosis schema

Write `submission/diagnosis.json`, then submit it:

```json
{
  "estimated_target_parameters": {"friction_scale": 1.0, "mass_scale": 1.0, "restitution": 0.0},
  "conclusion": "Your conclusion and uncertainty",
  "evidence_jobs": ["J0001"],
  "competing_hypotheses": [
    {"hypothesis": "Alternative explanation", "test_jobs": [],
     "outcome": "untested", "reason": "Why it remains unresolved"}
  ],
  "limitations": "What the observations cannot establish",
  "preferred_network": "N000"
}
```

Competing-hypothesis outcomes are `rejected`, `not_rejected` or `untested`.
