"""Export only the public adaptation observation whitelist."""

import csv
import json
import os
from pathlib import Path

import numpy as np

from orchestrator.files import write_json


def episode_result(record):
    """Keep outcomes while withholding private exception messages and paths."""
    return {"layout": record["layout"], "success": record["success"],
            "complete": record["complete"], "terminated": record["terminated"] if record["complete"] else False,
            "truncated": record["truncated"] if record["complete"] else False, "actions": record["actions"],
            "error": None if record["complete"] else "Episode incomplete"}


def export_episode(host_episode_dir, agent_dir):
    """Export state rows, normalized/native actions, outcomes and raw camera videos."""
    source, destination = Path(host_episode_dir), Path(agent_dir)
    destination.mkdir(parents=True, exist_ok=True)
    record = json.loads((source / "result.json").read_text())
    write_json(destination / "result.json", episode_result(record))
    transitions = source / "transitions.jsonl"
    rows = ([json.loads(line) for line in transitions.read_text().splitlines()]
            if record["complete"] or transitions.exists() else [])
    if rows:
        fields = {name: [] for name in (
            "step", "right_ee_pose", "right_arm_joint_position", "right_arm_joint_velocity",
            "right_gripper_joint", "right_gripper_opening", "right_finger_center",
            "cube_position", "cube_orientation", "cube_linear_velocity", "cube_angular_velocity",
            "plate_position", "plate_orientation",
        )}
        for row in rows:
            state = row["state"]
            sample, public = state["sample"], state["public_state"]
            values = {
                "step": sample["step"], "right_ee_pose": public["right_ee_pose"],
                "right_arm_joint_position": public["right_arm_joint_state"],
                "right_arm_joint_velocity": state["right_arm_joint_velocity"],
                "right_gripper_joint": public["right_ee_joint_state"],
                "right_gripper_opening": sample["gripper_opening"]["right"],
                "right_finger_center": sample["gripper_position"]["right"],
                **{key: sample[key] for key in ("cube_position", "cube_orientation",
                   "cube_linear_velocity", "cube_angular_velocity")},
                **{key: state[key] for key in ("plate_position", "plate_orientation")},
            }
            for key, value in values.items():
                fields[key].append(value)
        fields = {key: np.asarray(value) for key, value in fields.items()}
        fields["action"] = np.asarray([row["action"] for row in rows[1:]]).reshape(-1, 7)
        # Native commands are dictionaries; D concatenates both poses, then both grippers.
        native_keys = ("left_ee_pose", "right_ee_pose", "left_ee_joint_state", "right_ee_joint_state")
        fields["applied_action"] = np.asarray([
            [value for key in native_keys for value in row["applied_action"][key]]
            for row in rows[1:]
        ]).reshape(len(rows) - 1, -1) if len(rows) > 1 else np.empty((0, 16))
        np.savez_compressed(destination / "trajectory.npz", **fields)
    for camera in ("head", "left_wrist", "right_wrist"):
        video = source / f"raw_cam_{camera}.mp4"
        if record["complete"] or video.exists():
            os.link(video, destination / f"cam_{camera}.mp4")


def export_rollout(result, agent_dir):
    """Retain every planned outcome, including episodes that never started."""
    for record in result["episodes"]:
        destination = Path(agent_dir) / f"L{record['layout']}"
        if "episode_dir" in record:
            export_episode(record["episode_dir"], destination)
        write_json(destination / "result.json", episode_result(record))


def export_training(job_dir, agent_dir):
    """Export training totals and only the allowed episode and loss columns."""
    job_dir, agent_dir = Path(job_dir), Path(agent_dir)
    spec = json.loads((job_dir / "spec.json").read_text())
    result = json.loads((job_dir / "result.json").read_text())
    network = Path(spec["output"])
    write_json(agent_dir / "training.json", {
        "round": spec["round"], "network": network.name,
        "parent": Path(spec["parent"]).parent.name, "parameters": spec["parameters"],
        **{key: result[key] for key in ("round_actions", "online_episodes", "online_successes",
                                      "elapsed_seconds", "status")},
    })
    for source, name, columns in (
        ("online-episodes.jsonl", "online-episodes.csv", ("episode", "success", "complete", "actions")),
        ("rl-metrics.jsonl", "losses.csv", ("online_steps", "value_only", "policy_loss", "value_loss",
                                           "exact_kl", "step_fraction", "drift_from_parent")),
    ):
        with (agent_dir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for line in (network / source).read_text().splitlines():
                row = json.loads(line)
                writer.writerow({key: row[key] for key in columns if key in row})
