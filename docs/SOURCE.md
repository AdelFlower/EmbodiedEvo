# Source provenance and historical evidence

DynamicsAdapt is derived from EmbodiedEvo. It reuses the original project's
isolated CLI Agent launcher, staged provider home, workspace/socket isolation,
process lifecycle, raw evidence and model-usage recording. The active research
loop is the later FrictionAgent experiment: hidden dynamics diagnosis, calibrated
source reconstruction, Agent-written rewards and PPO recovery.

The simulator backend remains RoboDojo/Isaac Sim with local compatibility
adapters. Preserve these attributions when describing or redistributing the work.
The original EmbodiedEvo project reference is
[EmbodiedEvo: Self-improving Coding Agents for Physical Intelligence](https://arxiv.org/abs/2610.18867).
A project-wide license has not been selected; dependency or historical recipe
licenses do not license the entire project.

## Refactor boundary

The complete pre-refactor tree is recorded at Git commit
`37a151311bb3989596bb79a33ccc18121408b817`. It includes the original Playground/Test
benchmark, code-policy repair, SAC/PPO adaptation prototypes, fixability study,
friction search, historical specifications and supplied recipes. These are no
longer active entry points. They remain recoverable from Git history rather than
being duplicated in the current source tree.

The current source retains the FrictionAgent policy/training/diagnostic protocol,
minimal source-only P0 preparation, necessary Agent infrastructure and the
simulator dependency closure. Old CLI commands are not forwarded. The public
entry point is `dynamicsadapt` / `python -m orchestrator`.

Existing simulator wrappers still call `orchestrator.robodojo_image` and
`simulators.image_entrypoint`. Deployment `EMBODIEDEVO_*` environment keys,
provisioned vendor paths and installed image tags remain compatible. These are
runtime compatibility names, not alternate research workflows.

[`SOURCE_MANIFEST.json`](../SOURCE_MANIFEST.json) is preserved unchanged as a
historical export/provenance record. Its original source paths and checksums
belong to that export; it is not a manifest of the current DynamicsAdapt tree and
must not be rewritten to make historical hashes describe this refactor. Git
records the subsequent changes.

## Historical single-scene result

The existing local run `runs/202611041916_fa_run` reports:

| Frozen network | Target episodes | Strict successes | Lenient successes |
|---|---:|---:|---:|
| N000 (P0) | 30 | 0 | 0 |
| N002 (selected) | 30 | 30 | 30 |

The recorded Agent used two rounds, 100 PPO iterations, eight development target
episodes and eight diagnostic episodes. The historical specification describes
friction and mass probes followed by two reward revisions. Task performance and
support for the dynamics diagnosis must be assessed separately from the original
trajectories, predictions and job records.

These results predate the refactor. They use one fixed scene, 10 deterministic
and 20 stochastic evaluation episodes, simulated state/contact access, a source
imitation prior and no matched control. They do not establish robustness across
layouts, real-robot transfer or correctness of the refactored GPU execution.
The separately recorded oracle trainability run is researcher evidence and must
never enter an Agent workspace or the formal initialization.

Runtime results, weights, demonstrations, videos, credentials, environments,
images and machine profiles are not bundled in Git. A source checkout alone does
not reproduce the historical result; matching P0 artifacts and a separately
validated native runtime are required.

## Validation claims

Use `python -m unittest discover -s tests -v` for CPU software validation. Report
its actual outcome for the revision being reviewed. Real material/mass changes,
reset behavior, state/render parity and target recovery require separate GPU
checks. Docker image builds and model runs are also separate operations. Do not
present historical experiment outcomes or pre-refactor CPU counts as fresh
validation of this tree.

See [third-party notices](../THIRD_PARTY_NOTICES.md) for the unchanged historical
recipe sources and licensing records.
