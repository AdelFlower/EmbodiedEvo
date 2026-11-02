"""Workspace client for the separate dynamics experiment control channel."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path


def request(payload):
    address = os.environ.get("EMBODIEDEVO_SOCKET")
    if not address:
        raise RuntimeError("EMBODIEDEVO_SOCKET is not set")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(address)
        with connection.makefile("rwb") as stream:
            stream.write(json.dumps(payload, ensure_ascii=True).encode() + b"\n")
            stream.flush()
            data = stream.readline(16 * 1024 * 1024 + 1)
            if not data or not data.endswith(b"\n"):
                raise ConnectionError("Connection failed; query status before any new request")
            return json.loads(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start", help="start a fresh simulator and Python trial")
    start.add_argument("--kind", required=True, choices=("source", "target", "diagnostic"))
    start.add_argument("--parameters", type=Path, help="JSON file with diagnostic overrides")
    start.add_argument("--hypothesis", default="")
    start.add_argument("--prediction", default="")
    end = sub.add_parser("end", help="close the current trial and keep its evidence")
    end.add_argument("--conclusion", default="")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    payload = {"op": "dynamics_" + args.command}
    try:
        if args.command == "start":
            payload.update(
                kind=args.kind,
                parameters=json.loads(args.parameters.read_text()) if args.parameters else {},
                hypothesis=args.hypothesis,
                prediction=args.prediction,
            )
        elif args.command == "end":
            payload["conclusion"] = args.conclusion
        result = request(payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
