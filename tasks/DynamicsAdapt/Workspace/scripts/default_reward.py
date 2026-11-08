"""Compute the supplied lift, carry, slip and terminal reward."""

import numpy as np

from frictionlib import clearance, hand_offsets


def reward(trace):
    """Reconstruct the host default reward from post-step states and the initial state."""
    plate = trace["plate_position"]
    distance0 = np.linalg.norm(trace["initial_cube_position"][:2] - plate[:2])
    previous_offset = hand_offsets([trace["initial_observed_ee_pose"]],
                                   [trace["initial_cube_position"]])[0]
    previous_both = all(trace["initial_right_finger_contacts"])
    offsets = hand_offsets(trace["observed_ee_pose"], trace["cube_position"])
    rewards = np.empty(len(trace["action"]))
    m = c = 0.0
    for i, (cube, q, fingers, contacts, offset) in enumerate(zip(
            trace["cube_position"], trace["cube_quaternion"], trace["right_finger_contacts"],
            trace["contacts"], offsets)):
        both = all(fingers)
        m_prev, c_prev = m, c
        if both and not any(contacts[:2]):
            m = max(m, clearance(cube, q))
        c = float(m >= .02 and not contacts[0]) * (
            1 - min(np.linalg.norm(cube[:2] - plate[:2]) / distance0, 1))
        slip_mm = 1000 * np.linalg.norm(offset - previous_offset) if both and previous_both else 0.0
        previous_offset, previous_both = offset, both
        reason = trace["stop_reason"] if i == len(rewards) - 1 else None
        rewards[i] = (4 * (min(m, .05) - min(m_prev, .05)) / .05 + 4 * (c - c_prev)
                      - .25 * slip_mm + 10 * (reason == "success")
                      - 3 * (reason in ("dropped", "no_lift")))
    return rewards
