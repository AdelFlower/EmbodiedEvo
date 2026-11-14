# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Open an existing cube-lifting checkpoint in Isaac Sim without training."""

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
    parser.add_argument("--checkpoint", required=True, type=Path)
    args, playback_args = parser.parse_known_args(argv)
    if not re.match(r"^\d{12}(?:_|$)", args.run_dir.name):
        parser.error("--run-dir must start with YYYYMMDDHHMM followed by '_' or end of name")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    for name in ("env.yaml", "agent.yaml"):
        if not (checkpoint.parent / "params" / name).is_file():
            parser.error(f"checkpoint is missing its training configuration: params/{name}")
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    source_root = Path(__file__).resolve().parent
    env["PYTHONPATH"] = os.pathsep.join((str(source_root), str(source_root.parents[3]), env.get("PYTHONPATH", "")))
    command = [sys.executable, "-u", "-m", "isaaclab_arena.play", "--checkpoint", str(checkpoint), *playback_args]
    print(f"Playback log: {run_dir / 'play.log'}", flush=True)
    with (run_dir / "play.log").open("x") as log:
        result = subprocess.run(command, cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode == 0 and not (run_dir / "playback_complete.json").is_file():
        print("Playback ended without a completion record; see play.log", file=sys.stderr)
        return 1
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
