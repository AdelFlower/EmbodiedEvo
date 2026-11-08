"""Host environment and GPU binding for simulator workers."""

from __future__ import annotations

import os


def sim_env(gpu: str | None, backend: str | None = None) -> dict[str, str]:
    """Bind the simulator to one GPU for both CUDA and EGL.

    Hide all other GPUs and select the first visible EGL device.
    """
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    # Prevent native BLAS/OpenMP pools from multiplying across the simulator,
    # API workers and the agent. Oversubscription makes EGL aborts much more
    # likely when several campaigns are active on one host.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    if gpu is not None and gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["MUJOCO_EGL_DEVICE_ID"] = "0"
    if backend == "robodojo":
        env.update(
            {
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "PYTHONNOUSERSITE": "1",
                "PIP_USER": "0",
                "PIP_NO_COMPILE": "1",
                "XDG_RUNTIME_DIR": f"/tmp/robodojo-runtime-{os.getuid()}",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            }
        )
        for path in (
            "/etc/vulkan/icd.d/nvidia_icd.json",
            "/usr/share/vulkan/icd.d/nvidia_icd.json",
        ):
            if os.path.isfile(path):
                env["VK_ICD_FILENAMES"] = path
                break
    return env
