"""Run a reward program on every public episode in an input directory."""

import importlib.util
from pathlib import Path
import sys

import numpy as np

from frictionlib import load_trace


def main(argv=None):
    """Reject wrong shapes and nonfinite rewards before saving the batch."""
    program, incoming, outgoing = map(Path, sys.argv[1:] if argv is None else argv)
    spec = importlib.util.spec_from_file_location("reward_program", program)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outgoing.mkdir(parents=True, exist_ok=True)
    for path in sorted(incoming.glob("*.npz")):
        trace = load_trace(path)
        rewards = np.asarray(module.reward(trace), dtype=float)
        if rewards.shape != (len(trace["action"]),) or not np.isfinite(rewards).all():
            raise ValueError(f"{path.name}: reward must be a finite array of length steps")
        np.save(outgoing / (path.stem + ".npy"), rewards)


if __name__ == "__main__":
    main()
