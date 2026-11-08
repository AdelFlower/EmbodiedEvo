"""Repository paths and process lifecycle shared by host services."""

import os
from pathlib import Path
import signal
import subprocess

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT
# Preserve the existing simulator image's repository mount contract.
REPO = Path(os.environ.get("EMBODIEDEVO_REPO", CODE_ROOT)).resolve()


def stop_process(process):
    """Stop a job's process group without retrying its work."""
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
