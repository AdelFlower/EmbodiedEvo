"""NumPy observation, action decoding, demonstrations and network episodes."""

from pathlib import Path

import numpy as np

from integrations.dynamics_rl.features import ACTION_HIGH, ACTION_LOW, encode_action, unit_quaternion
from integrations.friction_rl.episode import clearance, hand_offsets, motion_peaks, rest_row, REST_STEPS
from integrations.friction_rl.primitive import ee_position, grasp_point, hand_q, slerp, PrimitivePolicy, PHASES
from orchestrator.files import write_json

OBS_DIM = 33
ACT_DIM = 5
HORIZON = 300
NO_LIFT_STEP = 160
STEP = 0.01
PAD_FLOOR = 0.778
F_MID = 25
F_HALF = 20
APPROACH_STEPS = 40


def observe(state, t, pad_cmd):
    """Encode state and commanded pad position with the fixed observation scales."""
    sample = state["sample"]
    pad_obs = grasp_point(state["public_state"]["right_ee_pose"])
    cube, q = sample["cube_position"], sample["cube_orientation"]
    angles = 2 * np.pi * np.array([1, 2, 4, 8, 16]) * t / HORIZON
    return np.r_[
        t / HORIZON, np.column_stack((np.sin(angles), np.cos(angles))).ravel(),
        (np.asarray(pad_cmd) - [.15, -.15, .85]) / [.2, .2, .1],
        (cube - pad_obs) / .05,
        (state["plate_position"] - pad_obs) / .3, clearance(cube, q) / .05,
        unit_quaternion(q), np.asarray(sample["cube_linear_velocity"]) / .2,
        sample["gripper_opening"]["right"], sample["finger_contacts"]["right"],
        sample["table_contact"], sample["plate_contact"],
    ].astype(np.float32)


class Decoder:
    """Integrate bounded pad displacements with a fixed top-down orientation schedule."""

    def __init__(self, state):
        pose = state["public_state"]["right_ee_pose"]
        self.pad_cmd = grasp_point(pose)
        self.q0 = unit_quaternion(pose[3:])
        self.commanded_pose = np.asarray(pose, dtype=float)

    def step(self, t, a):
        """Return the native action, squeeze force and gripper state."""
        a = np.clip(a, -1, 1)
        tau = min((t + 1) / APPROACH_STEPS, 1)
        q = slerp(self.q0, hand_q(0, 0), 10*tau**3 - 15*tau**4 + 6*tau**5)
        p = self.pad_cmd + STEP * a[:3]
        p[2] = max(p[2], PAD_FLOOR)
        ee = np.clip(ee_position(p, q), ACTION_LOW, ACTION_HIGH)
        self.commanded_pose = np.r_[ee, q]
        self.pad_cmd = grasp_point(self.commanded_pose)
        closed = bool(a[3] > 0)
        return (encode_action(self.commanded_pose, 0.0 if closed else 1.0),
                float(F_MID + F_HALF * a[4]) if closed else 0.0, closed)


def held_to_release(trace):
    """Require uninterrupted two-finger contact from lift-off to the open command."""
    both = np.asarray(trace["right_finger_contacts"], dtype=bool).all(axis=1)
    clear = np.array([clearance(p, q) for p, q in zip(trace["cube_position"], trace["cube_quaternion"])])
    lifts = np.flatnonzero(both & ~np.asarray(trace["contacts"], dtype=bool)[:, :2].any(axis=1) & (clear > .002))
    if not len(lifts):
        return False
    lift = lifts[0]
    closed = np.asarray(trace["closed"], dtype=bool)
    opens = np.flatnonzero(closed[:-1] & ~closed[1:]) + 1
    opens = opens[opens > lift]
    return bool(len(opens) and both[lift:opens[0]].all())


def run_net_episode(client, scene, episode_dir, controller):
    """Execute and record one terminal network episode and its learning batch."""
    episode_dir = Path(episode_dir)
    state = client.reset(scene, episode_dir)["state"]
    initial = {"initial_observed_ee_pose": np.array(state["public_state"]["right_ee_pose"], copy=True),
               "initial_cube_position": np.array(state["sample"]["cube_position"], copy=True),
               "initial_cube_quaternion": np.array(state["sample"]["cube_orientation"], copy=True),
               "initial_right_finger_contacts": np.array(state["sample"]["finger_contacts"]["right"], copy=True)}
    decoder = Decoder(state)
    plate = np.asarray(state["plate_position"])
    distance0 = np.linalg.norm(np.asarray(state["sample"]["cube_position"])[:2] - plate[:2])
    previous_offset = hand_offsets([state["public_state"]["right_ee_pose"]],
                                   [state["sample"]["cube_position"]])[0]
    previous_both = all(state["sample"]["finger_contacts"]["right"])
    m = c = 0.0
    released = was_closed = native_success = False
    rest_count = 0
    columns = {key: [] for key in (
        "commanded_ee_pose", "observed_ee_pose", "action", "closed", "grip_force_command",
        "right_grip_force", "gripper_opening", "right_finger_contacts", "contacts",
        "cube_position", "cube_quaternion", "reward")}
    batch = {key: [] for key in ("obs", "stored", "reward", "done")}
    for t in range(HORIZON):
        obs = observe(state, t, decoder.pad_cmd)
        stored, executed = controller(t, obs, state, decoder)
        executed = np.clip(executed, -1, 1)
        action, force, closed = decoder.step(t, executed)
        released |= was_closed and not closed
        was_closed |= closed
        response = client.request({"op": "step", "action": action.tolist(), "grip_force": force})
        state = response["state"]
        sample = state["sample"]
        pose = state["public_state"]["right_ee_pose"]
        cube, q = sample["cube_position"], sample["cube_orientation"]
        fingers = sample["finger_contacts"]["right"]
        contacts = [sample["table_contact"], sample["plate_contact"], sample["robot_contact"]]
        both = all(fingers)
        m_prev, c_prev = m, c
        if both and not any(contacts[:2]):
            m = max(m, clearance(cube, q))
        c = float(m >= .02 and not contacts[0]) * (
            1 - min(np.linalg.norm(np.asarray(cube)[:2] - plate[:2]) / distance0, 1))
        offset = hand_offsets([pose], [cube])[0]
        slip_mm = 1000 * np.linalg.norm(offset - previous_offset) if both and previous_both else 0.0
        previous_offset, previous_both = offset, both
        resting = rest_row(PHASES.index("release") if released else 0, contacts, fingers,
                           sample["gripper_opening"]["right"], cube, q, plate)
        rest_count = rest_count + 1 if resting else 0
        native_success |= bool(response["info"]["success"])
        if rest_count >= REST_STEPS or (response["terminated"] and response["info"]["success"]):
            reason = "success"
        elif response["terminated"] or response["truncated"]:
            reason = "native_end"
        elif contacts[0] and m >= .005:
            reason = "dropped"
        elif t + 1 >= NO_LIFT_STEP and m < .005:
            reason = "no_lift"
        elif t + 1 == HORIZON:
            reason = "timeout"
        else:
            reason = None
        reward = (4 * (min(m, .05) - min(m_prev, .05)) / .05 + 4 * (c - c_prev)
                  - .25 * slip_mm + 10 * (reason == "success")
                  - 3 * (reason in ("dropped", "no_lift")))
        row = (decoder.commanded_pose, pose, executed, closed, state["grip_force_command"],
               state["right_grip_force"], sample["gripper_opening"]["right"], fingers, contacts,
               cube, q, reward)
        for values, value in zip(columns.values(), row):
            values.append(np.array(value, copy=True))
        for key, value in zip(batch, (obs, stored, reward, reason is not None)):
            batch[key].append(np.array(value, copy=True))
        if reason is not None:
            break
    trace = {key: np.asarray(values) for key, values in columns.items()}
    trace.update(plate_position=plate, stop_reason=reason, **initial)
    batch = {key: np.asarray(values) for key, values in batch.items()}
    both = trace["right_finger_contacts"].all(axis=1)
    held = both & ~trace["contacts"][:, :2].any(axis=1)
    closed_rows = trace["closed"]
    force_rows = closed_rows & both
    offsets = hand_offsets(trace["observed_ee_pose"], trace["cube_position"])
    slip = 0.0
    if held.any() and closed_rows.any():
        first, last = np.flatnonzero(held)[0], np.flatnonzero(closed_rows)[-1]
        slip = float(np.linalg.norm(offsets[first:last + 1] - offsets[first], axis=1).max(initial=0.))
    speed, accel = motion_peaks(trace["commanded_ee_pose"][closed_rows])
    forces = trace["grip_force_command"][force_rows]
    record = {
        "success": reason == "success", "stop_reason": reason, "steps": len(batch["done"]),
        "return": float(batch["reward"].sum()), "max_clear": float(m), "lifted5": bool(m >= .05),
        "slip": slip, "force_mean_held": float(forces.mean()) if len(forces) else 0.0,
        "force_min_held": float(forces.min()) if len(forces) else 0.0,
        "grip_measured_held": float(trace["right_grip_force"][force_rows].mean()) if len(forces) else 0.0,
        "release_dist": float(np.linalg.norm(trace["cube_position"][-1, :2] - plate[:2])),
        "cmd_peak_speed": speed, "cmd_peak_accel": accel, "native_success": native_success,
    }
    record["held_to_release"] = held_to_release(trace)
    record["strict_success"] = record["success"] and record["held_to_release"]
    np.savez(episode_dir / "trace.npz", **trace)
    write_json(episode_dir / "episode.json", record)
    return record, trace, batch


class ScriptController:
    """Label corrective script displacements while executing bounded AR(1) command noise."""

    def __init__(self, params, force, sigma, rho, rng):
        self.script = PrimitivePolicy(params)
        self.force, self.sigma, self.rho, self.rng = force, np.asarray(sigma), rho, rng
        self.noise = np.zeros(3)

    def __call__(self, t, obs, state, decoder):
        """Return the clean corrective label and the noisy command."""
        _, phase = self.script.act(state)
        target = grasp_point(self.script.commanded_pose)
        closed = phase in ("close", "lift", "carry", "settle", "lower")
        self.noise = np.clip(self.rho * self.noise + np.sqrt(1 - self.rho**2)
                             * self.sigma * self.rng.standard_normal(3), -2*self.sigma, 2*self.sigma)
        label = np.r_[np.clip((target - decoder.pad_cmd) / STEP, -1, 1),
                      1 if closed else -1, (self.force - F_MID) / F_HALF]
        executed = label.copy()
        executed[:3] = np.clip((target + self.noise - decoder.pad_cmd) / STEP, -1, 1)
        return label, executed
