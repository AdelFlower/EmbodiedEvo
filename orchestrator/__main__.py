"""DynamicsAdapt: diagnose dynamics and recover capability through RL."""

import argparse
import sys


def main(argv=None):
    """Dispatch the experiment and source-policy preparation commands."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "evaluate", "trainability", "summarize", "p0", "deploy"))
    if not argv or argv[0] in ("-h", "--help"):
        parser.parse_args(argv)
        return 0
    command = parser.parse_args(argv[:1]).command
    if command == "deploy":
        from orchestrator import deployable

        deployable.main(argv[1:])
    elif command == "p0":
        from orchestrator import p0

        p0.main(argv[1:])
    else:
        from orchestrator import experiment

        experiment.main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
