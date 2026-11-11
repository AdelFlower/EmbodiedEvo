"""Host-fixed task contract, independent of the source/target world model."""

from integrations.friction_rl.control import ACTION_HIGH, ACTION_LOW
from integrations.friction_rl.episode import PLATE_REST_RADIUS, REST_STEPS
from integrations.friction_rl.network import (
    ACT_DIM, APPROACH_STEPS, Decoder, F_HALF, F_MID, HORIZON, NO_LIFT_STEP, PAD_FLOOR, STEP,
)
from simulators.dynamics_scoring import Criteria, describe_criteria


def describe():
    return {
        "task": "cube_to_plate", "contract": "cube_to_plate_v1",
        "control_hz": 25, "horizon": HORIZON, "gripper_scale": 1.0,
        "action": {
            "dim": ACT_DIM, "decoder": "cube_to_plate_v1", "clip": [-1, 1],
            "dimensions": ["pad_dx", "pad_dy", "pad_dz", "close_if_positive", "squeeze_force"],
            "pad_step_m": STEP, "pad_floor_m": PAD_FLOOR,
            "ee_low_m": ACTION_LOW.tolist(), "ee_high_m": ACTION_HIGH.tolist(),
            "orientation": f"fixed top-down, quintic approach over {APPROACH_STEPS} steps",
            "force": {"closed_N": f"{F_MID} + {F_HALF} * action[4]", "open_N": 0},
        },
        "native_scoring": describe_criteria(Criteria()),
        "lenient": (
            f"DP success: after an open command following a close, {REST_STEPS} consecutive steps "
            f"with cube centre within {PLATE_REST_RADIUS} m horizontally of plate centre, "
            "lowest corner between plate bottom - 0.002 m and rim + 0.003 m, plate contact, "
            "no table/robot/finger contact and opening >= 0.8; or terminated native success."),
        "strict": (
            "DP success and uninterrupted bilateral finger contact from the first two-finger "
            "lift with neither table nor plate contact and clearance > 0.002 m through "
            "the first subsequent close-to-open command transition."),
        "termination_order": [
            "success: DP rest or terminated native success", "native_end: native termination/truncation",
            "dropped: table contact after maximum held clearance >= 0.005 m",
            f"no_lift: step >= {NO_LIFT_STEP} and maximum held clearance < 0.005 m",
            f"timeout: step == {HORIZON}",
        ],
        "actor_schema": "dp_actor_v1", "critic_schema": "dp_critic_v1",
        "evidence_schema": "cube_to_plate_evidence_v1",
    }
