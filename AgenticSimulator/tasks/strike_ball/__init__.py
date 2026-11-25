# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register the UR10 strike-through-hole task with Gymnasium."""

import os
from datetime import datetime
from pathlib import Path

import gymnasium as gym

TASK_ID = "StrikeBall-Hole-Direct-v0"

gym.register(
    id=TASK_ID,
    entry_point=f"{__name__}.env:StrikeBallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:StrikeBallEnvCfg",
    },
)


def enter_run_directory(path: str) -> None:
    """Keep relative simulator outputs inside the requested run directory."""
    run_dir = Path(path).expanduser().resolve()
    prefix = run_dir.name[:12]
    if len(prefix) != 12 or not prefix.isdigit():
        raise ValueError("--run-dir must start with YYYYMMDDHHMM")
    datetime.strptime(prefix, "%Y%m%d%H%M")
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
