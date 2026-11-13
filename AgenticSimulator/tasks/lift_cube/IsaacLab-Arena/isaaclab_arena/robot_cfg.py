# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.assets import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg
from isaaclab.managers import ActionTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from isaaclab_arena.mdp import gripper_pos, ee_frame_pos, ee_frame_quat
from isaaclab_arena.stand import RobotPrimSpec, StandPrimSpec, compose_on_stand_usd

_FRANKA_ROBOT_PRIM = RobotPrimSpec(
    robot_usd_path=FRANKA_PANDA_CFG.spawn.usd_path,
    root_prim_path="/panda",
    robot_base_prim_name="panda_link0",
    stand_prim_name="stand_instanceable",
)


_FRANKA_STAND_PRIM = StandPrimSpec(
    stand_usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
    ref_prim_path="/Stand",
    payload_child_name="Stand",
    footprint_translate_xyz=(-0.05, 0.0, 0.0),
    stand_default_footprint_xy_m=(0.3888, 0.3888),
    stand_default_height=0.8755,
)


@configclass
class FrankaJointPosActionsCfg:
    """Joint-position action specification matching IsaacLab's FrankaCubeLiftEnvCfg."""

    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        # Actions are displacements from Isaac Lab's default Franka pose, which is the zero point
        # trained policies were fitted against. Stated rather than left to ``use_default_offset``,
        # which reads the spawn state and so would move the zero point whenever that pose changes.
        use_default_offset=False,
        offset={
            name: value
            for name, value in FRANKA_PANDA_CFG.init_state.joint_pos.items()
            if name.startswith("panda_joint")
        },
    )

    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class FrankaSceneCfg:
    """Additions to the scene configuration coming from the Franka embodiment."""

    robot: ArticulationCfg | None = None

    # The end-effector frame marker
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.1034],
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ],
    )

    def __post_init__(self):
        # Add a marker to the end-effector frame
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


@configclass
class FrankaObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)
        gripper_pos = ObsTerm(func=gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


_FRANKA_READY_POSE = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.785,
    "panda_joint3": -0.1107,
    "panda_joint4": -1.1775,
    "panda_joint5": 0.0,
    "panda_joint6": 0.785,
    "panda_joint7": 0.785,
    "panda_finger_joint.*": 0.0400,
}


@configclass
class FrankaEventCfg:
    """Configuration for Franka."""

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class FrankaRewardsCfg:
    """Reward specifications for the MDP."""

    action_rate = RewardTermCfg(func=mdp_isaac_lab.action_rate_l2, weight=-0.0001)
    joint_vel = RewardTermCfg(
        func=mdp_isaac_lab.joint_vel_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")}
    )


def _franka_robot_cfg_on_stand(robot_cfg: ArticulationCfg) -> ArticulationCfg:
    """Copy ``robot_cfg`` onto ``{ENV_REGEX_NS}/Robot`` with the composed on-stand USD."""
    cfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Arena reaches for objects on a table, so it spawns at its own ready pose rather than at the
    # pose Isaac Lab ships, which folds the elbow back.
    cfg.init_state = cfg.init_state.replace(joint_pos=_FRANKA_READY_POSE)
    cfg.spawn.usd_path = compose_on_stand_usd(
        _FRANKA_ROBOT_PRIM,
        _FRANKA_STAND_PRIM,
        stand_height_m=_FRANKA_STAND_PRIM.stand_default_height,
        output_basename="franka_panda_on_stand",
    )
    return cfg
