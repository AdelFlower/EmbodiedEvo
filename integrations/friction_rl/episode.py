"""Episode traces, early stops and friction-transfer outcome rewards."""

from pathlib import Path

import numpy as np

from integrations.friction_rl.primitive import (
    CUBE_HALF, PHASES, TABLE_Z, PrimitivePolicy, grasp_point,
)
from orchestrator.files import write_json
from simulators.dynamics_scoring import Criteria, cube_vertices

PLATE_REST_RADIUS = 0.045
REST_STEPS = Criteria().stable_steps


def clearance(position, quaternion):
    """Measure the lowest cube corner above the table."""
    return float(cube_vertices(position, quaternion, 2 * CUBE_HALF)[:, 2].min() - TABLE_Z)


def rest_row(phase, contacts, fingers, opening, position, quaternion, plate_position):
    """Recognize a released cube supported inside the plate without robot contact."""
    criteria = Criteria()
    if (phase < PHASES.index("release") or not contacts[1] or contacts[0] or contacts[2]
            or any(fingers) or opening < .8):
        return False
    lowest = cube_vertices(position, quaternion, criteria.cube_size)[:, 2].min()
    return bool(criteria.plate_bottom - .002 <= lowest <= criteria.plate_top + .003
                and np.linalg.norm(np.asarray(position)[:2] - np.asarray(plate_position)[:2])
                <= PLATE_REST_RADIUS)


def motion_peaks(poses):
    """Measure peak grasp-point speed and acceleration from 25 Hz pose samples."""
    points = np.array([grasp_point(pose) for pose in poses]).reshape(-1, 3)
    velocity = np.diff(points, axis=0) / .04
    acceleration = np.diff(velocity, axis=0) / .04
    return (float(np.linalg.norm(velocity, axis=1).max(initial=0.)),
            float(np.linalg.norm(acceleration, axis=1).max(initial=0.)))


def hand_offsets(poses, cubes):
    """Express cube offsets from the pad centres in the hand frames."""
    poses = np.asarray(poses)
    offsets = np.asarray(cubes) - np.array([grasp_point(p) for p in poses])
    inverse_xyz = -poses[:, 4:]
    return offsets + 2 * np.cross(
        inverse_xyz, np.cross(inverse_xyz, offsets) + poses[:, 3:4] * offsets)


def outcome(trace):
    """Compute task features and both rewards solely from recorded evidence."""
    phases = trace["phase"]
    fingers = trace["right_finger_contacts"].astype(bool)
    contacts = trace["contacts"].astype(bool)
    cube = trace["cube_position"]
    both, neither = fingers.all(axis=1), ~fingers.any(axis=1)
    release = np.flatnonzero(phases == PHASES.index("release"))
    release_step = int(release[0]) if len(release) else len(phases)
    clear = np.array([clearance(p, q) for p, q in zip(cube, trace["cube_quaternion"])])
    airborne = both & ~contacts[:, 0] & ~contacts[:, 1]
    before_release = np.arange(len(phases)) < release_step
    held_clear = clear[airborne & before_release]
    max_clear = float(held_clear.max()) if len(held_clear) else 0.0
    held_to_release = bool(len(release) and release_step >= 1 and both[release_step - 1])
    consecutive = rest_steps = 0
    for index, values in enumerate(zip(
            phases, contacts, fingers, trace["gripper_opening"], cube, trace["cube_quaternion"])):
        resting = index >= release_step and rest_row(*values, trace["plate_position"])
        consecutive = consecutive + 1 if resting else 0
        rest_steps = max(rest_steps, consecutive)
    detached = np.flatnonzero(neither & (np.arange(len(phases)) >= release_step))
    distance_step = int(detached[0]) if len(detached) else len(phases) - 1
    if trace["stop_reason"] == "dropped":
        distance_step = len(phases) - 1
    release_dist = float(np.linalg.norm(cube[distance_step, :2] - trace["plate_position"][:2]))
    carry = phases == PHASES.index("carry")
    moving = carry | (phases == PHASES.index("settle")) | (phases == PHASES.index("lower"))
    motion = moving | (phases == PHASES.index("lift"))
    cmd_speed, cmd_accel = motion_peaks(trace["commanded_ee_pose"][motion])
    obs_speed, _ = motion_peaks(trace["observed_ee_pose"][motion])
    lift = np.flatnonzero(phases == PHASES.index("lift"))
    slip = 0.0
    if len(lift) and moving.any():
        offsets = hand_offsets(trace["observed_ee_pose"], cube)
        slip = float(np.linalg.norm(offsets[moving] - offsets[lift[-1]], axis=1).max())
    grip_force = float(trace["right_grip_force"][carry].mean(axis=1).min()) if carry.any() else 0.0
    track_err = (float(np.linalg.norm(
        trace["commanded_ee_pose"][carry, :3] - trace["observed_ee_pose"][carry, :3],
        axis=1).max()) if carry.any() else 0.0)
    record = {
        "stop_reason": trace["stop_reason"], "grasped": bool(both.any()),
        "max_clear": max_clear, "lifted5": max_clear >= .05,
        "held_to_release": held_to_release, "release_dist": release_dist,
        "placed": bool(trace["scoring"][:, 1].any()), "slip": slip,
        "grip_force_min_carry": grip_force, "track_err_carry": track_err,
        "steps": len(phases), "rest_steps": rest_steps, "in_plate": rest_steps >= REST_STEPS,
        "native_success": bool(trace["native_success"]),
        "cmd_peak_speed": cmd_speed, "cmd_peak_accel": cmd_accel, "obs_peak_speed": obs_speed,
        "settle_before_release": int(np.sum((phases == PHASES.index("settle")) & before_release)),
    }
    success = record["in_plate"]
    record["success"] = bool(success)
    record["success_strict"] = bool(success and record["lifted5"] and held_to_release and record["placed"])
    common = (record["grasped"] + 2 * min(max_clear / .05, 1)
              + 2 * held_to_release * (1 - np.tanh(release_dist / .05)) + record["placed"])
    record["reward_v1"] = float(common + success * (6 - .005 * len(phases)))
    record["reward_v2"] = float(common + 6 * success - 2 * min(slip / .02, 1))
    return record


def run_episode(client, params, scene, episode_dir, grip_force=0.0):
    """Execute one primitive, preserving every action until the first stop."""
    episode_dir = Path(episode_dir)
    state = client.reset(scene, episode_dir)["state"]
    drive, plate = state["gripper_drive"], np.asarray(state["plate_position"])
    policy = PrimitivePolicy(params)
    columns = {key: [] for key in (
        "phase", "commanded_ee_pose", "observed_ee_pose", "gripper_command",
        "gripper_opening", "right_finger_contacts", "contacts", "cube_position",
        "cube_quaternion", "right_grip_force", "scoring", "grip_force_command")}
    rest_count = 0
    native_success = False
    while True:
        action, phase = policy.act(state)
        force = grip_force if phase in ("close", "lift", "carry", "settle", "lower") else 0.0
        response = client.request({"op": "step", "action": action.tolist(), "grip_force": force})
        state = response["state"]
        sample, scoring = state["sample"], state["scoring"]
        fingers = sample["finger_contacts"]["right"]
        phase_index = PHASES.index(phase)
        contacts = [sample["table_contact"], sample["plate_contact"], sample["robot_contact"]]
        row = (
            phase_index, policy.commanded_pose.copy(),
            state["public_state"]["right_ee_pose"], policy.grip,
            sample["gripper_opening"]["right"], fingers,
            contacts,
            sample["cube_position"], sample["cube_orientation"], state["right_grip_force"],
            [scoring["transport_completed"], scoring["released_over_plate"], scoring["stable_steps"]],
            state["grip_force_command"],
        )
        for values, value in zip(columns.values(), row):
            values.append(value)
        native_success |= bool(response["info"]["success"])
        resting = rest_row(phase_index, contacts, fingers, sample["gripper_opening"]["right"],
                           sample["cube_position"], sample["cube_orientation"], plate)
        rest_count = rest_count + 1 if resting else 0
        if (rest_count >= REST_STEPS
                or (response["terminated"] and response["info"]["success"])):
            reason = "success"
        elif response["terminated"] or response["truncated"]:
            reason = "native_end"
        elif phase == "lift" and policy.phase_end and (
                not all(fingers) or clearance(sample["cube_position"], sample["cube_orientation"]) < .02):
            reason = "no_lift"
        elif phase in ("carry", "settle", "lower") and sample["table_contact"]:
            reason = "dropped"
        elif policy.finished:
            reason = "not_placed"
        else:
            continue
        break
    trace = {key: np.asarray(value) for key, value in columns.items()}
    trace.update(plate_position=plate, stop_reason=reason, native_success=native_success,
                 gripper_drive=np.asarray([drive["stiffness"], drive["max_force"]]))
    record = {**outcome(trace), "physical": dict(params), "scene": scene,
              "episode_dir": str(episode_dir)}
    np.savez(episode_dir / "trace.npz", **trace)
    write_json(episode_dir / "episode.json", record)
    return record, trace
