"""Host tool: copy weights-only P0 and its frozen calibration into a DP bundle."""

import argparse
import json
from pathlib import Path
import re
import shutil

from .policy import BUNDLE_MANIFEST, load_directory


def make_bundle(policy, calibration, out, run_dir):
    run_dir, out = Path(run_dir), Path(out)
    if re.match(r"^\d{12}", run_dir.name) is None:
        raise ValueError("run_dir requires a YYYYMMDDHHMM prefix")
    run_dir.mkdir(parents=True)
    out.mkdir(parents=True)
    shutil.copyfile(policy, out / "policy.pt")
    shutil.copyfile(calibration, out / "calibration.json")
    (out / "policy_bundle.json").write_text(json.dumps(BUNDLE_MANIFEST, indent=2) + "\n")
    load_directory(out)
    (run_dir / "bundle.json").write_text(json.dumps({"bundle": str(out.resolve()), "training_state": None}) + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "calibration", "out", "run-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    make_bundle(args.policy, args.calibration, args.out, args.run_dir)


if __name__ == "__main__":
    main()
