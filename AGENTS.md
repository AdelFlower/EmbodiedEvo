# DynamicsAdapt development boundaries

DynamicsAdapt is a general, policy-driven Agent harness for diagnosing dynamics
differences between a source simulator and a fixed target, revising the source
simulator, and recovering target performance by continuing reinforcement learning.
Read [PLAN.md](PLAN.md) (research design and protocol) and
[docs/HARNESS_SPEC.md](docs/HARNESS_SPEC.md) (engineering spec and milestones)
before changing execution, training or evaluation behavior. Cube-to-plate is the
first task adapter; its details must not become assumptions of the general layer.

Migration status: the FrictionAgent code (`orchestrator/experiment*.py`,
`integrations/friction_rl/`, `tasks/DynamicsAdapt/`) is the legacy implementation
being migrated by the HARNESS_SPEC milestones. Until a milestone replaces a module,
keep that module working for its existing callers. `docs/PROTOCOL.md` describes
the legacy protocol only.

Research boundaries:

- Keep the formal loop neural-policy driven. The Agent requests experiments and
  writes environment candidates, analysis tools and reward programs; it never
  sends robot actions. Probes, deployments and evaluations use policy mean actions.
- The task contract is host-fixed per task adapter: for cube-to-plate, the
  five-action decoder, 25 Hz control, 300-step horizon, native scoring,
  strict/lenient outcomes, termination order and `gripper_scale = 1.0`.
  Policy observation layouts belong to the task/policy adapters, not the harness.
- The Agent may revise the source world model (parameters, added laws, added
  states/degrees of freedom, geometry/contact structure, dynamics programs) through
  versioned candidates built by the host. It cannot change the target, scoring,
  termination, observation meaning, action decoding or robot command entry points.
- Continue the latest complete PPO training state (model, both optimizers,
  counters). The policy lineage is linear. No BC term or demonstration data enters
  adaptation training. Target trajectories never enter gradients, demonstrations
  or normalization statistics.
- The Agent decides when to train, revise, or submit. Do not add research budgets
  or automatic early stopping; record all usage. Stopping rules are in PLAN.md.
- Keep target specifications, weights, simulator sources, host records, oracle and
  reference runs outside the Agent workspace. Repository documents and Git history
  must never be mounted into the Agent container. Do not name the formal target
  values in tracked files (host-private configs live outside Git).
- Preserve raw trajectories and videos; do not add a model to summarize observations.

Execution boundaries:

- Keep the Agent container (CPU, no GPU, no weights) separate from GPU simulation
  and PPO jobs. Agent-written rewards and dynamics programs run only in the isolated
  runners defined by HARNESS_SPEC and are validated at their boundary.
- Treat `vendor/` as read-only after provisioning. Put compatibility adapters in
  `simulators/` or `integrations/`; never redirect to another checkout.
- Preserve the existing simulator wrapper and deployment environment variables.
  Never copy credentials, environments, runtime profiles, assets, weights or results
  into Git.
- Every experiment, host-preparation or probe invocation requires an explicit
  `--run-dir` with a `YYYYMMDDHHMM` prefix. Preserve existing runs; create new
  attempts separately.
- Retain failed and partial attempts in their planned denominator. Agent-written
  code failures are model failures. Crashes, timeouts and missing reports alone do
  not prove infrastructure causation. Only confirmed infrastructure failures may
  receive a separately documented rerun that preserves the original evidence.

Implementation rules (apply to every contributor, including Codex):

- No defensive programming: no broad `try/except`, fallback paths, automatic
  retries, or speculative options beyond the spec. Validate inputs once at a
  boundary (config load, request intake, sandbox output) and trust internal calls.
- Content hashes are identifiers for frozen versions, not verification machinery.
  Do not add checksum re-verification layers.
- Reuse existing functions rather than copying them; one responsibility per module;
  keep modules near the size guidance in HARNESS_SPEC.
- Stop at each HARNESS_SPEC milestone boundary for review.
- Use `python -m unittest discover -s tests -v` for CPU validation. GPU checks,
  container builds and model calls are separate operations requiring resources.
- Tests and test reports are local-only and ignored by Git. Do not force-add them.
- Keep `SOURCE_MANIFEST.json` as historical provenance, not a current file manifest.
  Preserve source attribution and the historical recipe records in Git history.

The former Playground/Test workflow, Skills/Lessons transfer and
`workspace-session-v1` code-execution protocol are historical functionality; they
are not requirements for this research project.
