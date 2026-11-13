"""Backward roll-in curriculum. No expert phase is supplied to the policy."""

from collections import deque

# Phase to hand off in; fraction=0 hands off before its first expert action.
STAGES = (("release", 8), ("lower", 7), ("transport", 6), ("lift", 5),
          ("grasp", 4), ("descend", 3), ("approach", 2), ("clearance", 1), ("full", 0))


def fractions(config, level):
    return config.get("handoff_fractions", [0.0]) if 0 < level < len(STAGES) - 1 else [0.0]


class Curriculum:
    def __init__(self, config, level=0):
        self.config, self.level = config, level
        self.progress_index = config.get("start_progress_index", 0)
        self.window = deque(maxlen=config["window"])
        self.last_evaluation = 0

    def observe(self, rows):
        for row in rows:
            if (row["curriculum_level"] == self.level
                    and row.get("curriculum_progress_index", 0) == self.progress_index):
                self.window.append(bool(row["success"]))

    def ready(self, update):
        enough = len(self.window) >= self.config["window"]
        # Exploration can spoil an otherwise competent deterministic policy.
        # Only frozen validation decides promotion, not stochastic training rate.
        return enough and update - self.last_evaluation >= self.config["evaluation_interval"]

    def state(self):
        return dict(level=self.level, name=STAGES[self.level][0],
                    progress_index=self.progress_index,
                    handoff_fraction=fractions(self.config, self.level)[self.progress_index],
                    window_episodes=len(self.window),
                    window_success_rate=sum(self.window) / len(self.window) if self.window else None)

    def promote(self):
        if self.progress_index + 1 < len(fractions(self.config, self.level)):
            self.progress_index += 1
        else:
            self.level = min(self.level + 1, len(STAGES) - 1)
            self.progress_index = 0
        self.window.clear()
