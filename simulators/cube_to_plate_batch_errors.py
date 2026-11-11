"""Consume Kit's configured log at control boundaries, without restart/retry behavior."""

import re


class PhysicsErrors:
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.broken = set()
        self.fatal = False
        self.partial = b''

    def poll(self):
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            chunk = stream.read()
            self.offset = stream.tell()
        lines = (self.partial+chunk).split(b'\n')
        self.partial = lines.pop()
        for line in lines:
            match = re.search(rb'Invalid PhysX transform detected for /World/envs/env_(\d+)/', line)
            if match:
                self.broken.add(int(match[1]))
            if (b'fail to launch kernel' in line or b'PhysX Internal CUDA error' in line
                    or b'PhysX has reported too many errors, simulation has been stopped' in line):
                self.fatal = True
        return self.broken, self.fatal
