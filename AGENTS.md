# EmbodiedEvo development boundaries

This repository contains the RoboDojo reproduction source. Read README.md and
docs/PROTOCOL.md before changing execution or evaluation behavior.

- Preserve `workspace-session-v1`: persistent Python sessions and exactly four
  primitives: `get_instruction()`, `get_observation()`, `step(action)`, and
  Playground-only `reset()`. Keep the primitive manual handwritten.
- Only `skills/` and `lessons/` transfer from Playground to Test. Freeze them
  before selecting Test scenes. Each Test gets a fresh agent, workspace and
  simulator; knowledge is read-only, and reset and cross-scene learning are forbidden.
- Preserve task goals, native action limits, success checks and raw feedback.
  Do not add a model to summarize observations or videos.
- Keep ledgers, private rewards, scene plans and simulator sources outside the
  agent workspace. The agent image has no simulator or GPU access.
- Treat `vendor/` as read-only after provisioning. Put compatibility adapters
  in `simulators/` or `integrations/`; do not redirect to another project checkout.
- Every invocation needs an explicit `--run-dir` with a `YYYYMMDDHHMM` prefix.
  Never copy results, credentials, environments or machine configuration into Git.
- Preserve failed attempts. Model-written code failures count as model failures.
  Only confirmed infrastructure failures are candidates for an explicitly recorded
  rerun; crashes, timeouts and missing reports alone do not establish causation.
- Use `python -m unittest discover -s tests -v` for CPU validation. GPU experiments,
  container builds and model calls are separate operations requiring actual resources.
- Keep the third-party recipe text, license and checksums unchanged. Record its use
  as supplied task knowledge, separately from learned Skills/Lessons.

# Rules for Changes and Communication

1. **Limit each code change to one script.** If a change requires edits to multiple files, list every file, explain why each edit is necessary, and obtain the user's explicit approval before editing.
2. **Research each change and get approval before editing.** Read the relevant code and configuration, and check where the code is called. Identify the file and logic you plan to change. Explain the current behavior, why the change is needed, and what will happen afterward. Wait for the user to approve the plan before making any edits.
3. **Keep validation focused.** Handle only edge cases relevant to the request. Avoid defensive code for hypothetical situations. Never perform hash checks. Obtain the user's explicit permission before running any test, including a single test, a regression suite, or a smoke test. The test commands above are for reference.
4. **Write for a beginner.** Show a relevant code snippet and point to concrete evidence, such as a configuration value or the line that calls a function. Explain in plain language how the evidence supports your conclusion. Define technical terms when you first use them.
