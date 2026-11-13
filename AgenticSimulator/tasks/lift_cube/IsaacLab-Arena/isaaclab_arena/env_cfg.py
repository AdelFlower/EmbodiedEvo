# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Native Isaac Lab configuration for the single Franka cube-lifting task."""

from __future__ import annotations

import isaaclab.envs.mdp as lab_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_arena import mdp
from isaaclab_arena.robot_cfg import (
    FrankaEventCfg, FrankaJointPosActionsCfg, FrankaObservationsCfg,
    FrankaRewardsCfg, FrankaSceneCfg, _franka_robot_cfg_on_stand, FRANKA_PANDA_CFG,
)


@configclass
class LiftCubeSceneCfg(InteractiveSceneCfg):
    robot = None
    ee_frame = None
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
    )
    dex_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/dex_cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8), activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.055), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    ground_plane = AssetBaseCfg(
        prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(FrankaObservationsCfg.PolicyCfg):
        target_object_position = ObservationTermCfg(func=mdp.generated_object_command)
        object_position = ObservationTermCfg(
            func=mdp.object_position_in_frame,
            params={"root_frame_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("dex_cube")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy = PolicyCfg()


@configclass
class CommandsCfg:
    object_pose = lab_mdp.UniformPoseCommandCfg(
        asset_name="robot", body_name="panda_hand", resampling_time_range=(5.0, 5.0), debug_vis=True,
        ranges=lab_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.255, 0.455),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


@configclass
class RewardsCfg(FrankaRewardsCfg):
    reaching_object = RewardTermCfg(
        func=mdp.object_ee_distance, weight=1.0,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("dex_cube"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    lifting_object = RewardTermCfg(
        func=mdp.object_is_lifted, weight=15.0,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("dex_cube")},
    )
    object_goal_tracking = RewardTermCfg(
        func=mdp.object_goal_distance, weight=16.0,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose",
                "robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("dex_cube")},
    )
    object_goal_tracking_fine_grained = RewardTermCfg(
        func=mdp.object_goal_distance, weight=5.0,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose",
                "robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("dex_cube")},
    )


@configclass
class EventsCfg(FrankaEventCfg):
    reset_cube = EventTermCfg(func=mdp.reset_cube, mode="reset")


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=lab_mdp.time_out, time_out=True)
    object_dropped = TerminationTermCfg(
        func=lab_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("dex_cube")},
    )
    success = TerminationTermCfg(
        func=mdp.lift_object_rl_success,
        params={"object_cfg": SceneEntityCfg("dex_cube"), "robot_cfg": SceneEntityCfg("robot"),
                "command_name": "object_pose", "position_tolerance": 0.05, "rl_training": True},
    )


@configclass
class LiftCubeEnvCfg(ManagerBasedRLEnvCfg):
    scene = LiftCubeSceneCfg(num_envs=1, env_spacing=30.0, replicate_physics=False)
    observations = ObservationsCfg()
    actions = FrankaJointPosActionsCfg()
    events = EventsCfg()
    rewards = RewardsCfg()
    commands = CommandsCfg()
    terminations = TerminationsCfg()
    curriculum = None
    sim = SimulationCfg(dt=1 / 200, render_interval=2)
    decimation = 4
    episode_length_s = 5.0
    viewer = ViewerCfg(eye=(-1.0, -1.5, 1.555), lookat=(0.5, 0.0, 0.055), origin_type="env")

    def __post_init__(self):
        self.sim.physx = PhysxCfg()
        self.scene.robot = _franka_robot_cfg_on_stand(FRANKA_PANDA_CFG.copy())
        self.scene.ee_frame = FrankaSceneCfg().ee_frame
