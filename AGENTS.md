# DynamicsAdapt development boundaries

This project studies Agent diagnosis of dynamics differences and reinforcement
learning recovery. Read README.md and docs/PROTOCOL.md before changing execution,
training or evaluation behavior. FrictionAgent is the sole retained research
case; the public CLI is `dynamicsadapt` / `python -m orchestrator`.

- Keep the formal loop neural-policy driven: the Agent requests experiments and
  writes analysis/reward programs; it does not directly send robot actions.
- Preserve the current 33-dimensional observation, five-action decoder, 25 Hz
  control, 300-step network horizon, physics, native scoring, strict/lenient
  outcomes and termination rules unless a research change is explicitly requested.
- The Agent may change only calibrated-source friction, mass and restitution,
  its reward program and the requested training iterations. The target is fixed
  by the host and hidden from the Agent. Gripper scale remains 1.0.
- Continue the latest PPO network and both optimizers. Keep the source-only BC
  prior, reproduction gate, warm-up rules, budgets and host candidate selection.
  Target trajectories must not enter training gradients or demonstration data.
- Keep target parameters, weights, simulator sources, oracle evidence and host
  records outside the Agent workspace. Preserve raw trajectories and videos;
  do not add a model to summarize observations.
- Keep the Agent container separate from the GPU simulator. Run Agent-written
  rewards only through the existing isolated reward runner and validate outputs.
- Treat `vendor/` as read-only after provisioning. Put compatibility adapters in
  `simulators/` or `integrations/`; never redirect to another checkout.
- Preserve the existing simulator wrapper and deployment environment variables.
  Never copy credentials, environments, runtime profiles, assets or results into Git.
- Every experiment/P0 invocation requires an explicit `--run-dir` with a
  `YYYYMMDDHHMM` prefix. Preserve existing runs and create new attempts separately.
- Retain failed and partial attempts in their planned denominator. Agent-written
  code failures are model failures. Crashes, timeouts and missing reports alone
  do not prove infrastructure causation. Only confirmed infrastructure failures
  may receive a separately documented rerun that preserves the original evidence.
- Use `python -m unittest discover -s tests -v` for CPU validation. GPU checks,
  container builds and model calls are separate operations requiring resources.
- Tests and test reports are local-only and ignored by Git. Do not force-add them.
- Keep `SOURCE_MANIFEST.json` as historical provenance, not a current file manifest.
  Preserve source attribution and the historical recipe text/license/checksum
  records in Git history. Recipes are not part of the active Agent knowledge.

The former Playground/Test workflow, Skills/Lessons transfer and
`workspace-session-v1` code-execution protocol are historical functionality; they
are not requirements for this research project.
