# Installation

DynamicsAdapt uses three separate runtimes: a host Python environment for the
simulator wrapper, a host RL environment, and a thin Agent container. The GPU
simulator runs in a separately provisioned Docker image. Use Linux and Python
3.11+ and run commands from the repository root.

## Python environments

For a fresh source checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-host.txt -e .
python3 -m venv .venv-rl
.venv-rl/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv-rl/bin/python -m pip install -r requirements-host.txt -r requirements-rl.txt -e '.[dev]'
```

The RL network trains on the host CPU; Isaac uses the GPU inside its image. Keep
host pins out of the simulator environment. This project is distributed as a
source checkout with an editable install, not a standalone wheel containing task
assets or workspace templates. Reinstall editable metadata in each host Python
environment after upgrading the project from EmbodiedEvo to DynamicsAdapt.

The test suite is maintained locally and excluded from Git. A fresh checkout does
not include it. When available, run
`.venv-rl/bin/python -m unittest discover -s tests -v`.
CPU tests exercise the software contracts using fakes where simulation is needed.
They do not establish contact dynamics, reset/render parity or policy performance.

## Provisioned RoboDojo simulator

The supported path requires these local resources, supplied independently of Git:

- A repository-local `vendor/robodojo/` source/assets snapshot with the required
  RoboDojo, Isaac Lab and CuRobo dependencies. It must be a real local directory,
  not a link to a different development checkout.
- A working Docker simulator image, pinned by its full `sha256:` image ID, with
  the matching Python/native dependencies and offline assets/material support.
- Docker GPU access, an available NVIDIA GPU and compatible host driver.
- `.venv-robodojo/runtime.json`, configured from
  [`configs/runtime.example.json`](../configs/runtime.example.json), and an
  executable `.venv-robodojo/bin/python` wrapper.

The wrapper delegates to the host command
`python -m orchestrator.robodojo_image`; that launches
`simulators.image_entrypoint` in the pinned simulator image. Existing wrappers
retain these module paths. Configure `simulator_python` in the experiment to point
at this wrapper. Do not replace it with a Python interpreter from another checkout.

The runtime profile identifies `image`, image-internal `python`, `source_root`
and extension cache paths, plus a writable host cache. Use actual local values,
not the example placeholders. `EMBODIEDEVO_ROBODOJO_RUNTIME_PROFILE` can select an
external profile instead of `.venv-robodojo/runtime.json`. The image launcher mounts
this checkout and its vendor snapshot, not a historical source tree elsewhere.

This repository does not distribute or rebuild the historical simulator image,
assets or machine configuration. Provision and validate them separately before a
GPU experiment. Treat vendor files as read-only once provisioned; retain existing
`.embodiedevo-provisioned.json` and source-patch provenance records. The retained
runtime supports Docker, without an Apptainer or cluster-launch path.

## Thin Agent image

Build the separate CPU-only Agent image from the small Docker context:

```bash
bash scripts/build_agent_image.sh dynamicsadapt-agent:latest
```

The Dockerfile installs pinned Codex CLI and basic Python/image tools. It contains
no credentials, simulator sources, task assets or GPU runtime. Set both
`agent.image` and `sandbox.image` in the experiment configuration to a locally
available compatible image. An existing `embodiedevo-agent:latest` tag remains a
valid explicit configuration value; changing the project name does not require
renaming or rebuilding that installed image.

The model Agent gets only its workspace, a staged temporary CLI home and the
experiment socket. Reward programs run in a separate instance with no network and
only code/input/output mounts. Container builds and smoke runs are separate
resource-consuming operations, not part of CPU validation.

## Credentials and deployment settings

Follow [Codex API setup](CODEX_API.md). Credentials stay outside the checkout and
are staged into each temporary Agent home; the real user home is never mounted.
The deployment environment variable names remain compatible with existing setups:

| Variable | Purpose |
|---|---|
| `EMBODIEDEVO_CODEX_HOME` | External Codex provider configuration and file credentials |
| `EMBODIEDEVO_AGENT_HTTPS_PROXY` | Agent container proxy |
| `EMBODIEDEVO_AGENT_CPUSET` | Optional Agent CPU affinity |
| `EMBODIEDEVO_ROBODOJO_RUNTIME_PROFILE` | Local simulator runtime profile |
| `EMBODIEDEVO_SIMULATOR_TMPDIR` | Simulator temporary directory root |
| `EMBODIEDEVO_SIMULATOR_CPUSET` | Optional simulator CPU affinity |

Keep existing deployed values when upgrading. Do not commit runtime profiles,
credentials, virtual environments, images, vendor/assets or experiment outputs.
After setup, follow [running experiments](RUNNING.md) for CPU validation, a separate
scripted smoke, P0 preparation and the formal Agent experiment.
