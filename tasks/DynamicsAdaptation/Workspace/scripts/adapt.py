"""Request host-run adaptation experiments over the workspace socket."""

import argparse
import json
import os
import socket
from pathlib import Path


def request(payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(os.environ["EMBODIEDEVO_SOCKET"])
        with connection.makefile("rwb") as stream:
            stream.write(json.dumps(payload).encode() + b"\n")
            stream.flush()
            return json.loads(stream.readline())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="op", required=True)
    commands.add_parser("status")
    probe = commands.add_parser("probe")
    probe.add_argument("--network", required=True)
    probe.add_argument("--parameters", type=Path, required=True)
    probe.add_argument("--layouts", required=True)
    probe.add_argument("--hypothesis", required=True)
    probe.add_argument("--prediction", required=True)
    train = commands.add_parser("train")
    train.add_argument("--parameters", type=Path, required=True)
    train.add_argument("--actions", type=int, required=True)
    train.add_argument("--rationale", required=True)
    wait = commands.add_parser("wait")
    wait.add_argument("job")
    wait.add_argument("--seconds", type=int, default=600)
    submit = commands.add_parser("submit")
    submit.add_argument("diagnosis", type=Path)
    payload = vars(parser.parse_args(argv))
    for key in ("parameters", "diagnosis"):
        if key in payload:
            payload[key] = json.loads(payload[key].read_text())
    if "layouts" in payload:
        payload["layouts"] = [int(value) for value in payload["layouts"].split(",")]
    response = request(payload)
    print(json.dumps(response, indent=2))
    return 0 if response["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
