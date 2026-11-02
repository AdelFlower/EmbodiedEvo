# Source Environment: Learn a Complete Control Policy

You are a coding agent controlling a robot. Complete the task in the current environment and produce a complete Python policy that runs independently from the initial state. First read `primitives/primitives.md`, obtain the task instruction through `get_instruction()`, inspect raw images and robot state, and iterate on the control code.

This is a research Playground. The robot interface contains only `get_instruction()`, `get_observation()`, `step(action)`, and `reset()`. You may edit `submission/`, `skills/`, `lessons/`, and `scratch/`. `instruction.md`, `primitives/`, `scripts/`, `observations/`, `source_experience/`, and host-provided configuration are read-only. Do not rewrite raw observations.

## Success criteria

{{SUCCESS_CRITERIA}}

## Start and execute

```bash
python scripts/dynamics.py status
python scripts/dynamics.py start --kind source
python scripts/env.py exec submission/solution.py
```

Each `exec` executes the whole file from its first line against the current robot state. Python variables and functions persist within a trial. During exploration, submit incremental snippets to avoid accidentally replaying earlier actions. Use `--include skills/helper.py` for helper code; includes execute source in order and are not Python imports. Only `.py` files under `submission/` or `skills/` may be submitted. Control code, including comments, must use ASCII.

After every execution, read the returned budgets, success signal, stdout/stderr, and raw observation directory, and inspect images and videos. Actions taken before an error are not rolled back. Do not assume a timeout or disconnection means nothing executed, and do not automatically resend the request.

```bash
python scripts/env.py status
python scripts/env.py reset
python scripts/dynamics.py end --conclusion "Observed results of this execution"
```

`reset()` restores the same trial's initial state and preserves Python variables, but consumes another episode. A new `start` ends the current trial, creates a new simulator, and clears the Python namespace. It does not restore global budgets. Use only `--kind source` during source learning.

## Record and publish

Write reusable code to `skills/` and promptly record observed failures, successful techniques, and unresolved issues in `lessons/`. These files provide source experience for later independent research stages. Do not rely on conversation content that has not been saved to files.

When finished, assemble the final policy into a complete program that runs in a fresh simulator with an empty Python namespace, and update `submission/policy.json`:

```json
{"entry": "submission/solution.py", "includes": []}
```

List required Python helper files in `includes` in execution order. Avoid unintended top-level robot actions in helpers. The final policy must not call reset or depend on variables or actions from previous submissions. The last successful snippet may not be a complete policy; first check the full entry point in a new source trial.

End the current trial, then finish the session:

```bash
python scripts/dynamics.py end --conclusion "Complete policy published; describe known results and limitations"
python scripts/env.py finish
```

## Budgets

| Resource | Phase limit |
|---|---:|
| Execution requests | {{EXECUTION_BUDGET}} |
| Total episodes, including initial starts and resets | {{EPISODE_BUDGET}} |
| Cumulative native actions | {{TOTAL_ACTION_BUDGET}} |
| Native actions per episode | {{NATIVE_ACTION_LIMIT}} |

`exec`, `instruction`, `observe`, and standalone reset requests consume execution slots. Resets inside code still consume episodes. Status queries consume no slots and do not advance physics. Use live `status` and execution feedback to plan remaining actions, and stop taking actions after `terminated` or `truncated`. Resetting, switching trials, or reconnecting cannot restore cumulative budgets.
