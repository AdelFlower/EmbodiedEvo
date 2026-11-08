"""Export public friction evidence and evaluate reward programs in isolated batches."""

from concurrent.futures import ThreadPoolExecutor, wait
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from orchestrator.files import write_json

SCRIPTS = Path(__file__).resolve().parents[2] / "tasks/DynamicsAdapt/Workspace/scripts"
TRACE_KEYS = ("commanded_ee_pose observed_ee_pose action closed grip_force_command right_grip_force "
              "gripper_opening right_finger_contacts contacts cube_position cube_quaternion plate_position "
              "initial_observed_ee_pose initial_cube_position initial_cube_quaternion "
              "initial_right_finger_contacts stop_reason").split()
RESULT_KEYS = ("strict_success held_to_release stop_reason steps max_clear slip force_mean_held "
               "force_min_held release_dist cmd_peak_speed cmd_peak_accel").split()
TRAIN_COLUMNS = ("iteration warmup strict_successes lenient_successes mean_return mean_max_clear "
                 "mean_slip mean_force_mean_held approx_kl policy_loss value_loss").split()


def public_trace(trace):
    """Select exactly the public trajectory columns."""
    return {key: trace[key] for key in TRACE_KEYS}


def export_episode(source, destination):
    """Export whitelisted arrays, outcomes and hard-linked raw videos."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (source / "trace.npz").is_file():
        with np.load(source / "trace.npz", allow_pickle=False) as trace:
            np.savez(destination / "trajectory.npz", **public_trace(trace))
    if (source / "episode.json").is_file():
        record = json.loads((source / "episode.json").read_text())
        write_json(destination / "result.json", {**{key: record[key] for key in RESULT_KEYS},
                                                "success_lenient": record["success"]})
    for camera in ("head", "left_wrist", "right_wrist"):
        video = source / f"raw_cam_{camera}.mp4"
        if video.is_file():
            os.link(video, destination / f"cam_{camera}.mp4")


def export_rollout(result, destination):
    """Export each planned episode that produced evidence."""
    for row in result["episodes"]:
        if row["episode_dir"] is not None:
            export_episode(row["episode_dir"], Path(destination) / row["name"])


def export_training(network, destination, result):
    """Export iteration metrics and the final batch with the rewards actually used."""
    network, destination = Path(network), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "iterations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRAIN_COLUMNS)
        writer.writeheader()
        for line in (network / "iterations.jsonl").read_text().splitlines():
            row = json.loads(line)
            writer.writerow({key: row[key] for key in TRAIN_COLUMNS})
    for k, source in enumerate(result["last_iteration"]):
        folder = destination / "last_iteration" / f"E{k}"
        export_episode(source, folder)
        shutil.copyfile(Path(source) / "rewards.npy", folder / "rewards.npy")


def sandbox_argv(code, incoming, outgoing, cfg):
    """Build the single-batch Docker command with only three data mounts."""
    return ["docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "64", "--memory",
            f"{cfg['memory_mb']}m", "--cpus", "1", "--user", "1000:1000",
            "--mount", f"type=bind,src={Path(code).resolve()},dst=/code,readonly",
            "--mount", f"type=bind,src={Path(incoming).resolve()},dst=/in,readonly",
            "--mount", f"type=bind,src={Path(outgoing).resolve()},dst=/out",
            cfg["image"], "python3", "/code/reward_runner.py", "/code/program.py", "/in", "/out"]


def check_output(outgoing, lengths):
    """Reject missing, incorrectly shaped, nonfinite or excessive reward arrays."""
    rewards = {}
    for name, steps in lengths.items():
        path = Path(outgoing) / f"{name}.npy"
        if not path.is_file():
            return {"ok": False, "error": f"Missing rewards for {name}"}
        value = np.load(path, allow_pickle=False)
        if value.shape != (steps,) or not np.isfinite(value).all() or np.any(np.abs(value) > 1e4):
            return {"ok": False, "error": f"Invalid rewards for {name}: shape, finiteness or magnitude"}
        rewards[name] = value
    return {"ok": True, "rewards": rewards}


def run_sandbox(program_text, traces, directory, cfg):
    """Snapshot code, run one isolated batch and validate every returned array."""
    directory = Path(directory)
    code, incoming, outgoing = [directory / name for name in ("code", "in", "out")]
    for folder in (code, incoming, outgoing):
        folder.mkdir(parents=True, exist_ok=False)
        folder.chmod(0o755)
    outgoing.chmod(0o777)
    (code / "program.py").write_text(program_text)
    for name in ("frictionlib.py", "reward_runner.py"):
        shutil.copyfile(SCRIPTS / name, code / name)
    for name, trace in traces.items():
        np.savez(incoming / f"{name}.npz", **public_trace(trace))
    with subprocess.Popen(sandbox_argv(code, incoming, outgoing, cfg), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as process, ThreadPoolExecutor(1) as executor:
        future = executor.submit(process.communicate)
        done, _ = wait([future], timeout=cfg["timeout"])
        if not done:
            process.kill()
        stdout, stderr = future.result()
    (directory / "stdout.log").write_text(stdout)
    (directory / "stderr.log").write_text(stderr)
    if not done or process.returncode:
        return {"ok": False, "error": ("Reward sandbox timed out: " if not done else "Reward program failed: ")
                + stderr[-4000:]}
    result = check_output(outgoing, {name: len(trace["action"]) for name, trace in traces.items()})
    if not result["ok"]:
        result["error"] += "\n" + stderr[-4000:]
    return result
