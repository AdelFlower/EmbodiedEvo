"""Parallel socket clients and ordered episode evidence for fixability."""

import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from threading import Event, Lock

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from integrations.dynamics_rl.data import EpisodeData
from orchestrator.dynamics_rl import simulator
from orchestrator.files import write_json


@contextmanager
def simulator_pool(run_dir, config, gpu, parameters, mode):
    """Start all clients concurrently and close every client on pool exit."""
    clients = [simulator(Path(run_dir), f"{mode}_w{k}", config, gpu, mode, parameters=parameters)
               for k in range(config["num_envs"])]
    with ExitStack() as stack:
        for client in clients:
            stack.callback(client.close)
        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            list(executor.map(lambda client: client.start(), clients))
        yield clients


class ThreadedVecEnv(DummyVecEnv):
    """Reuse SB3 VecEnv bookkeeping while dispatching blocking RPCs concurrently."""

    def __init__(self, envs):
        super().__init__([lambda env=env: env for env in envs])
        self.executor = ThreadPoolExecutor(max_workers=len(envs))

    def _step(self, index):
        obs, reward, terminated, truncated, info = self.envs[index].step(self.actions[index])
        done = terminated or truncated
        info["TimeLimit.truncated"] = truncated and not terminated
        if done:
            info["terminal_observation"] = obs
            obs, self.reset_infos[index] = self.envs[index].reset()
        self._save_obs(index, obs)
        self.buf_rews[index], self.buf_dones[index], self.buf_infos[index] = reward, done, info

    def step_wait(self):
        list(self.executor.map(self._step, range(self.num_envs)))
        return self._obs_from_buf(), self.buf_rews.copy(), self.buf_dones.copy(), deepcopy(self.buf_infos)

    def close(self):
        self.executor.shutdown(wait=True)
        super().close()


def episode_records(layouts):
    return [{"layout": item["index"], "success": False, "complete": False,
             "actions": 0, "error": None, "episode_dir": None, "not_started": True}
            for item in layouts]


def run_parallel(clients, layouts, records, episode):
    """Stop dequeuing after an episode error and join in-flight episodes before raising."""
    queue, lock, stopped = deque(enumerate(layouts)), Lock(), Event()

    def worker(client):
        while True:
            with lock:
                if stopped.is_set() or not queue:
                    return
                index, item = queue.popleft()
            record = records[index]
            folder = client.run_dir / item.get("name", f"E{item['index']:03d}")
            record.update(episode_dir=str(folder), not_started=False)
            started = time.time()
            try:
                episode(client, item, folder, record)
            except BaseException as exc:
                stopped.set()
                record.update(error=str(exc), type=type(exc).__name__, traceback=traceback.format_exc())
                raise
            finally:
                record["elapsed_seconds"] = time.time() - started
                write_json(folder / "episode-summary.json", record)

    with ThreadPoolExecutor(max_workers=len(clients)) as executor:
        futures = [executor.submit(worker, client) for client in clients]
    for future in futures:
        future.result()
    return records


def run_episodes(clients, layouts, policy, *, deterministic):
    """Evaluate ordered layouts with clipped deterministic or sampled actor actions."""
    records, prediction_lock = episode_records(layouts), Lock()

    def episode(client, item, folder, record):
        data, trace = EpisodeData(), []
        try:
            response = client.reset(item["scene"], folder)
            trace.append(response["state"]["sample"]["cube_position"])
            while not response["terminated"] and not response["truncated"]:
                with prediction_lock:
                    action = policy.predict(np.asarray(response["obs"], np.float32), deterministic=deterministic)[0]
                following = client.step(action)
                data.append(response, action, following)
                record["actions"] = len(data.rows)
                response = following
                trace.append(response["state"]["sample"]["cube_position"])
            record.update(complete=True, success=bool(response["info"]["success"]),
                          terminated=response["terminated"], truncated=response["truncated"],
                          videos=response["info"].get("videos", {}))
        finally:
            folder.mkdir(parents=True, exist_ok=True)
            arrays = data.arrays()
            record["reward"] = float(arrays["rewards"].sum())
            data.save(folder / "transitions.npz")
            np.save(folder / "cube-trajectory.npy", np.asarray(trace))

    return run_parallel(clients, layouts, records, episode)
