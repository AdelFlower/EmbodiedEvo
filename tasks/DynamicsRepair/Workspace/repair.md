# Unknown Dynamics Change: Diagnose and Repair the Control Policy

You are a coding agent controlling a robot. A policy that previously ran in the source environment has failed after transfer to the current target environment. An unknown dynamics change has occurred. Use raw feedback to propose and test hypotheses, and repair the complete code policy so it completes the same task in the fixed target environment.

First read `primitives/primitives.md` and the read-only `source_experience/`. They contain the source policy, source-stage experience, and raw target-trial feedback. The host fixes the target environment parameters. You cannot modify them, the robot, the task goal, the success criteria, or action budgets.

## Workspace and robot interface

The robot interface still contains only `get_instruction()`, `get_observation()`, `step(action)`, and research Playground-only `reset()`. You may edit `submission/`, `skills/`, `lessons/`, and `scratch/`. `instruction.md`, `primitives/`, `scripts/`, `observations/`, `source_experience/`, and host-provided nominal configuration are read-only.

Within a trial, iterate on code using `python scripts/env.py exec submission/solution.py`. Python variables and environment state persist. Each exec runs all submitted source from the first line; included files execute in order in the same namespace. Control code, including comments, must use ASCII. Later exceptions do not roll back actions already taken.

## Success criteria

{{SUCCESS_CRITERIA}}

## Diagnostic and target trials

```bash
python scripts/dynamics.py status
python scripts/dynamics.py start --kind target
python scripts/env.py exec submission/solution.py
python scripts/dynamics.py end --conclusion "Observed behavior and explanations not yet ruled out"
```

Each `start` ends the current trial, creates a new simulator from the same construction state, and clears the Python namespace. Target trials accept policies but no parameter overrides. Diagnostic copies start from the public nominal configuration. Their results test hypotheses and cannot substitute for success in the target environment.

Three general diagnostic controls are available: `friction_scale` scales both static and dynamic friction of the manipulated object, `mass_scale` scales its mass, and `restitution` sets its material's coefficient of restitution. Omitted fields retain their nominal values. Allowed ranges are `[0.05, 2.0]` for `friction_scale`, `[0.25, 4.0]` for `mass_scale`, and `[0, 1]` for `restitution`; values must be finite. Baseline values are in read-only `nominal.json`. Parameters take effect before a new diagnostic trial and remain fixed within that trial.

Before each diagnostic trial, write a parameter JSON file in `scratch/` and state your hypothesis and prediction. For example, a control experiment with no parameter changes can use an empty object `{}`:

```bash
python scripts/dynamics.py start --kind diagnostic --parameters scratch/parameters.json --hypothesis "The explanation to distinguish in this trial" --prediction "The observable behavior predicted before execution"
python scripts/env.py exec submission/solution.py
python scripts/dynamics.py end --conclusion "Actual observations, comparison with predictions, and next steps"
```

Your experimental condition may restrict you to nominal diagnostic copies. If nonempty overrides are unavailable, repair the policy within the granted permissions rather than trying to bypass the restriction. All branches may try the fixed target environment, but target access is limited.

Control relevant variables when choosing experiments, and compare at least one competing hypothesis. Similar failure behavior does not identify a unique cause. Record ambiguity when observations cannot distinguish explanations, and do not infer an unsupported diagnosis retrospectively from successful repair. Use existing action capabilities; the interface offers no direct grip-force or actuator-torque control.

## Record, publish, and finish

Promptly record hypotheses, trial identifiers, raw feedback locations, conclusions, and reasons for policy changes in `lessons/`. Save reusable control code in `skills/`. Preserve failed attempts and uncertain conclusions.

Write your final diagnosis to `lessons/diagnosis.json`:

```json
{
  "conclusion": "The category and direction supported by current evidence; explicitly state uncertainty",
  "evidence_trials": [],
  "competing_hypothesis": "Alternative explanations compared and their outcomes",
  "limitations": "Causes not yet ruled out and observation limitations"
}
```

Use actual trial identifiers in `evidence_trials`. This is your summary of the evidence; it is not itself proof that the diagnosis is established.

Assemble the final policy into a complete program that runs in a fresh simulator with an empty Python namespace, and update `submission/policy.json`:

```json
{"entry": "submission/solution.py", "includes": []}
```

List required Python files under `skills/` or `submission/` in `includes`, in execution order. The final policy must not call reset or depend on earlier incremental snippets or manually executed preparatory actions. The host freezes and independently evaluates this policy. Evaluation feedback is not used for further policy changes.

```bash
python scripts/dynamics.py end --conclusion "Final policy and diagnosis published; describe limitations"
python scripts/env.py finish
```

## Budgets

| Resource | Branch limit |
|---|---:|
| Execution requests | {{EXECUTION_BUDGET}} |
| Total episodes | {{EPISODE_BUDGET}} |
| Target episodes within that total | {{TARGET_EPISODE_BUDGET}} |
| Cumulative native actions | {{TOTAL_ACTION_BUDGET}} |
| Native actions per episode | {{NATIVE_ACTION_LIMIT}} |

Diagnostic and target trials share cumulative budgets. New trials and resets consume episodes; target resets also consume target episodes. Reset preserves current Python variables and trial parameter overrides, restoring only the same initial state and the per-episode action allowance. It does not replenish execution requests or cumulative actions.

`exec`, `instruction`, `observe`, and standalone reset requests consume execution slots. Status queries consume no slots and do not advance physics. Check live `status` before and after execution, and stop actions after `terminated` or `truncated`. Disconnections, timeouts, and missing reports do not mean actions never occurred. Do not automatically resend requests or delete failed attempts.
