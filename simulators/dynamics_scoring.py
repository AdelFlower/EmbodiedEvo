"""Host-private task scoring; this module has no simulator dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class Criteria:
    cube_size: float = 0.025
    table_height: float = 0.765
    lift_clearance: float = 0.03
    transport_distance: float = 0.05
    grasp_steps: int = 5
    plate_inner_radius: float = 0.04
    plate_bottom: float = 0.765
    plate_top: float = 0.8006
    release_opening: float = 0.8
    hand_clearance: float = 0.05
    release_height: float = 0.05
    stable_steps: int = 13


def describe_criteria(c):
    """The public statement of PlacementScorer; keep it in step with update()."""

    def cm(value):
        return f"{value * 100:g} cm"

    return f"""Success is judged at every control step from the cube's pose and the simulator's \
contact reports. The table top is at z = {c.table_height:g} m and the plate's rim at \
z = {c.plate_top:g} m.

1. **Grasp and carry.** Both fingers of one gripper touch the cube while its lowest corner \
is at least {cm(c.lift_clearance)} above the table top and it touches neither the table nor \
the plate. This airborne two-finger grasp must last at least {c.grasp_steps} consecutive \
control steps and move the cube at least {cm(c.transport_distance)} horizontally from where it \
began. If any of these conditions lapses, the count and the starting point begin again.
2. **Release over the plate.** At the first step after the carry when neither finger of the \
carrying gripper touches the cube, every corner of the cube must lie horizontally within \
{cm(c.plate_inner_radius)} of the plate's centre, the cube must not touch the table, and its \
lowest corner must be no more than {cm(c.release_height)} above the rim. A cube that is \
tilted or off-centre when the fingers lose contact can fail this check even if it then \
settles in the middle of the plate. If the check fails, the carry no longer counts, and only \
a new grasp and carry can count again. Losing finger contact anywhere else, for example by \
slipping during the carry, is treated the same way.
3. **Rest.** The cube must then rest on the plate for {c.stable_steps} consecutive control \
steps: touching the plate but not the table, every corner still within \
{cm(c.plate_inner_radius)} of the centre, its lowest corner no higher than 3 mm above the \
rim, no robot link touching it, the carrying gripper at least {c.release_opening:.0%} open, \
and that gripper's fingers at least {cm(c.hand_clearance)} from the cube's centre.

Touching the table after a carry cancels that carry. The cube's speed is not checked."""


def cube_vertices(position, orientation, size=0.025):
    """World-space corners, using native scalar-first unit quaternions."""
    w, x, y, z = np.asarray(orientation, dtype=float)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    corners = np.array(list(product((-size / 2, size / 2), repeat=3)))
    return corners @ rotation.T + np.asarray(position)


class PlacementScorer:
    """Require a continuous bilateral grasp and airborne carry before placement."""

    def __init__(self, plate_position, criteria=None):
        self.criteria = criteria or Criteria()
        self.plate_position = np.asarray(plate_position, dtype=float)
        self.grasp_arm = None
        self.grasp_start = None
        self.grasp_count = 0
        self.transported = False
        self.transport_arm = None
        self.released_over_plate = False
        self.stable_count = 0
        self.success = False
        self.last_step = -1

    def update(self, sample):
        # Re-reading a host snapshot must never create time or scoring progress.
        step = sample["step"]
        if step == self.last_step or self.success:
            return self.success
        if step < self.last_step:
            raise ValueError("Create a fresh scorer after reset")
        self.last_step = step
        c = self.criteria
        position = np.asarray(sample["cube_position"])
        vertices = cube_vertices(position, sample["cube_orientation"], c.cube_size)
        inside = bool(
            np.all(
                np.linalg.norm(vertices[:, :2] - self.plate_position[:2], axis=1)
                <= c.plate_inner_radius
            )
        )
        both_fingers = [arm for arm, contacts in sample["finger_contacts"].items() if all(contacts)]
        clear = (
            vertices[:, 2].min() >= c.table_height + c.lift_clearance
            and not sample["table_contact"]
            and not sample["plate_contact"]
        )
        arm = self.grasp_arm if self.grasp_arm in both_fingers else next(iter(both_fingers), None)
        if arm is None:
            self.grasp_arm = self.grasp_start = None
            self.grasp_count = 0
        else:
            if self.grasp_arm != arm:
                self.grasp_arm = arm
                self.grasp_start = None
                self.grasp_count = 0
            if clear:
                if self.grasp_start is None:
                    self.grasp_start = position.copy()
                self.grasp_count += 1
            else:
                self.grasp_start = None
                self.grasp_count = 0
            moved = (
                self.grasp_start is not None
                and np.linalg.norm(position[:2] - self.grasp_start[:2]) >= c.transport_distance
            )
            if clear and moved and self.grasp_count >= c.grasp_steps:
                self.transported = True
                self.transport_arm = arm

        if (
            self.transported
            and not self.released_over_plate
            and not any(sample["finger_contacts"][self.transport_arm])
        ):
            self.released_over_plate = bool(
                inside
                and not sample["table_contact"]
                and c.plate_bottom - 0.002 <= vertices[:, 2].min() <= c.plate_top + c.release_height
            )
            if not self.released_over_plate:
                self.transported, self.transport_arm = False, None
        if sample["table_contact"]:
            self.transported, self.released_over_plate, self.transport_arm = False, False, None
        supported_height = c.plate_bottom - 0.002 <= vertices[:, 2].min() <= c.plate_top + 0.003
        # robot_contact covers every robot link, not only fingers: a palm, wrist or
        # the other arm still touching the cube is continued support, not rest.
        released = (
            self.transport_arm is not None
            and not sample["robot_contact"]
            and not any(any(value) for value in sample["finger_contacts"].values())
            and sample["gripper_opening"][self.transport_arm] >= c.release_opening
            and np.linalg.norm(
                np.asarray(sample["gripper_position"][self.transport_arm]) - position
            )
            >= c.hand_clearance
        )
        # Score v4 judges landing by support, position and release only. Measured cube
        # speeds are recorded but not checked: in run 202610292357 a cube resting on the
        # plate, unmoved for 100 control steps, measured a steady 0.11 rad/s peak.
        stable = (
            self.released_over_plate
            and inside
            and supported_height
            and released
            and sample["plate_contact"]
            and not sample["table_contact"]
        )
        self.stable_count = self.stable_count + 1 if stable else 0
        self.success = self.stable_count >= c.stable_steps
        return self.success

    def snapshot(self):
        return {
            "criteria": asdict(self.criteria),
            "grasp_steps": self.grasp_count,
            "transport_completed": self.transported,
            "transport_arm": self.transport_arm,
            "released_over_plate": self.released_over_plate,
            "stable_steps": self.stable_count,
            "success": self.success,
        }
