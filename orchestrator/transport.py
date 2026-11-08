"""Bounded JSON-line transport for private simulator sockets."""

import json

MAX_MESSAGE = 16 * 1024 * 1024


def receive(stream):
    """Read one complete object without retrying an ambiguous operation."""
    line = stream.readline(MAX_MESSAGE + 1)
    if not line or len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
        raise ConnectionError(
            "Incomplete response; execution may have happened. Do not retry automatically."
        )
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def send(stream, value):
    """Write one bounded JSON object."""
    data = json.dumps(value, ensure_ascii=True).encode() + b"\n"
    if len(data) > MAX_MESSAGE:
        raise ValueError("Message exceeds 16 MiB")
    stream.write(data)
    stream.flush()
