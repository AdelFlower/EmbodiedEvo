"""DP reward on parallel T+1 public/source_truth rows; SI units, no host imports."""

import numpy as np


def reward(trace):
    public, truth = trace["public"], trace["source_truth"]
    plate = np.asarray(truth[0]["plate_position"])
    distance0 = np.linalg.norm(np.asarray(truth[0]["cube_position"])[:2] - plate[:2])
    closed = [row["executed_action"][3] > 0 for row in public[1:]]
    rewards = np.zeros(len(closed))
    m = c = 0.0
    lifted = opened = lost = False
    for i, row in enumerate(truth[1:]):
        both = bool(np.all(row["right_finger_contacts"]))
        m_prev, c_prev = m, c
        if both and not row["table_contact"] and not row["plate_contact"]:
            clear = row["lift_height"]
            m = max(m, clear)
            lifted |= clear > .002
        c = float(m >= .02 and not row["table_contact"]) * (
            1 - min(np.linalg.norm(np.asarray(row["cube_position"])[:2] - plate[:2]) / distance0, 1))
        opened |= i > 0 and closed[i - 1] and not closed[i]
        rewards[i] = 4 * (min(m, .05) - min(m_prev, .05)) / .05 + 4 * (c - c_prev) - .25 * 1000 * row["slip"]
        if lifted and not opened and not lost and closed[i] and not both:
            lost = True
            rewards[i] -= 5
    reason = truth[-1]["stop_reason"]
    if reason == "success":
        rewards[-1] += -3 if lost else 10
    elif reason in ("dropped", "no_lift"):
        rewards[-1] -= 3
    return rewards
