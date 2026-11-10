"""Host-only oracle reward: default terms, success bonus only when held to release, penalty for losing the cube."""

import numpy as np

from integrations.friction_rl.episode import clearance, hand_offsets


def reward(trace):
    """Default lift/carry/slip terms; -5 when the cube leaves the closed fingers; +10 only for strict success."""
    plate = trace["plate_position"]
    distance0 = np.linalg.norm(trace["initial_cube_position"][:2] - plate[:2])
    previous_offset = hand_offsets([trace["initial_observed_ee_pose"]], [trace["initial_cube_position"]])[0]
    previous_both = bool(np.all(trace["initial_right_finger_contacts"]))
    offsets = hand_offsets(trace["observed_ee_pose"], trace["cube_position"])
    closed = np.asarray(trace["closed"], dtype=bool)
    rewards = np.zeros(len(closed))
    m = c = 0.0
    lifted = opened = lost = False
    for i in range(len(closed)):
        both = bool(np.all(trace["right_finger_contacts"][i]))
        contacts = trace["contacts"][i]
        m_prev, c_prev = m, c
        if both and not contacts[0] and not contacts[1]:
            clear = clearance(trace["cube_position"][i], trace["cube_quaternion"][i])
            m = max(m, clear)
            lifted |= clear > .002
        c = float(m >= .02 and not contacts[0]) * (
            1 - min(np.linalg.norm(trace["cube_position"][i][:2] - plate[:2]) / distance0, 1))
        slip_mm = 1000 * np.linalg.norm(offsets[i] - previous_offset) if both and previous_both else 0.0
        previous_offset, previous_both = offsets[i], both
        opened |= i > 0 and closed[i - 1] and not closed[i]
        rewards[i] = 4 * (min(m, .05) - min(m_prev, .05)) / .05 + 4 * (c - c_prev) - .25 * slip_mm
        if lifted and not opened and not lost and closed[i] and not both:
            lost = True
            rewards[i] -= 5
    if trace["stop_reason"] == "success":
        rewards[-1] += -3 if lost else 10
    elif trace["stop_reason"] in ("dropped", "no_lift"):
        rewards[-1] -= 3
    return rewards
