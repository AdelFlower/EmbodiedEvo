# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Launch cube-lifting PPO training in a new, explicit output directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args, trainer_args = parser.parse_known_args(argv)
    if not re.match(r"^\d{12}(?:_|$)", args.run_dir.name):
        parser.error("--run-dir must start with YYYYMMDDHHMM followed by '_' or end of name")
    fixed_options = {"--task", "--rl_library", "--external_callback", "--object", "--embodiment"}
    if any(token.split("=", 1)[0] in fixed_options for token in trainer_args):
        parser.error("task, object, embodiment, RL library and registration callback are fixed")
    run_dir = args.run_dir.expanduser().resolve()
    # A failed attempt keeps its directory and cannot be overwritten by a new run.
    run_dir.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable, "-u", "-m", "isaaclab_arena.train",
        *trainer_args,
    ]
    print(f"Training log: {run_dir / 'training.log'}", flush=True)
    with (run_dir / "training.log").open("x") as log:
        result = subprocess.run(command, cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode == 0 and not (run_dir / "training_complete.json").is_file():
        print("Training ended without a completion record; see training.log", file=sys.stderr)
        return 1
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
