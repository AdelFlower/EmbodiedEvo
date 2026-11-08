"""Shared simulator pools and scene construction for policy preparation and adaptation."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from queue import Queue

import numpy as np

from integrations.friction_rl.client import SimulatorClient
from integrations.friction_rl.control import sample_scene

SOURCE_PARAMETERS = {"friction_scale": 1.0, "mass_scale": 1.0,
                     "restitution": 0.0, "gripper_scale": 1.0}


class SimulatorPool:
    """Run a shared episode queue with one thread per persistent client."""

    def __init__(self, clients):
        self.clients = clients

    def map(self, function, jobs):
        """Apply a client function over the shared queue in job order."""
        queue = Queue()
        for index, job in enumerate(jobs):
            queue.put((index, job))
        for _ in self.clients:
            queue.put(None)
        records = [None] * len(jobs)

        def work(client):
            for index, job in iter(queue.get, None):
                records[index] = function(client, job)

        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            list(executor.map(work, self.clients))
        return records


@contextmanager
def start_pool(run_dir, cfg, physics, mode, gpu):
    """Start clients concurrently and close every client on any exit."""
    with ExitStack() as stack:
        clients = []
        for k in range(cfg["num_envs"]):
            client = SimulatorClient(
                run_dir / f"{run_dir.name[:12]}_sim_w{k}",
                parameters=physics, gpu=gpu, mode=mode,
                simulator_python=cfg["simulator_python"],
                startup_timeout=cfg["startup_timeout"], step_timeout=cfg["step_timeout"],
                worker_module="integrations.friction_rl.worker")
            stack.callback(client.close)
            clients.append(client)
        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            list(executor.map(lambda client: client.start(), clients))
        yield SimulatorPool(clients)


def build_scene(seed, cfg):
    """Sample a layout and apply the configured cube-only XY shift."""
    scene = sample_scene(seed, cfg["layout_radius"])
    position = scene["Rigid"]["cube"][0]["default_pos"]
    position[:2] = (np.asarray(position[:2]) + cfg["cube_shift"]).tolist()
    return scene
