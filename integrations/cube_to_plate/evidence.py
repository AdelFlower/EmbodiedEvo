"""Positive field allowlists: runtime inputs and host records never pass through."""

from copy import deepcopy

PUBLIC_FIELDS = (
    "time_s", "joint_position", "joint_velocity", "gripper_opening", "executed_action",
    "commanded_ee_pose", "grip_force_command", "ee_pose", "tactile_normal_force", "video_frame_index",
)
SOURCE_TRUTH_FIELDS = (
    "cube_position", "cube_quaternion", "cube_linear_velocity", "cube_angular_velocity",
    "plate_position", "plate_quaternion", "table_contact", "plate_contact", "robot_contact",
    "right_finger_contacts", "stop_reason", "slip", "lift_height", "release_distance",
)
OUTCOME_FIELDS = ("strict", "lenient", "execution_status")


def public_step(step):
    """One SI/world-frame row; t=0 is reset, then one row per executed action."""
    return {key: deepcopy(step[key]) for key in PUBLIC_FIELDS}


def source_truth_step(step):
    """Source-only truth; never call this builder for target exports."""
    return {key: deepcopy(step[key]) for key in SOURCE_TRUTH_FIELDS}


def public_outcome(outcome, execution_status):
    if execution_status not in ("complete", "execution_error"):
        raise ValueError("public outcome requires complete or execution_error")
    return {"strict": bool(outcome["strict"]) if execution_status == "complete" else False,
            "lenient": bool(outcome["lenient"]) if execution_status == "complete" else False,
            "execution_status": execution_status}
