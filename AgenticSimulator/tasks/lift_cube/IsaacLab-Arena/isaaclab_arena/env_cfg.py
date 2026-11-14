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
from isaaclab.utils.configclass import configclass

from isaaclab_arena import mdp
from isaaclab_arena.robot_cfg import (
    FrankaEventCfg, FrankaJointPosActionsCfg, FrankaObservationsCfg,
    FrankaRewardsCfg, FrankaSceneCfg, plate_cube_robot_cfg,
)

from AgenticSimulator.scenes.plate_cube import (
    TABLE_SIZE, TABLE_POSITION, TABLE_HEIGHT, ROBOT_POSITION,
    load_scene_config, object_spawn_config, object_pose_bounds,
)

SCENE_CONFIG = load_scene_config()
SCENE_CONFIG["objects"] = [item for item in SCENE_CONFIG["objects"] if item["name"] == "Cube"]
OBJECTS = {item["name"]: item for item in SCENE_CONFIG["objects"]}
CUBE_BOUNDS, CUBE_ORIENTATION = object_pose_bounds(OBJECTS["Cube"])
CUBE_POSITION = tuple((low + high) / 2 for low, high in CUBE_BOUNDS)
MINIMUM_HEIGHT = TABLE_HEIGHT + 0.04



@configclass
class LiftCubeSceneCfg(InteractiveSceneCfg):
    robot = None
    ee_frame = None
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE, collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.40, 0.25)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POSITION),
    )
    # Keep the scene key for existing checkpoints; the asset is now the local Cube.
    dex_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube", spawn=object_spawn_config(OBJECTS["Cube"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POSITION, rot=CUBE_ORIENTATION),
    )
    ground_plane = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


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
            pos_x=(CUBE_POSITION[0] - ROBOT_POSITION[0] - 0.1, CUBE_POSITION[0] - ROBOT_POSITION[0] + 0.1),
            pos_y=(CUBE_POSITION[1] - ROBOT_POSITION[1] - 0.25, CUBE_POSITION[1] - ROBOT_POSITION[1] + 0.25),
            pos_z=(CUBE_POSITION[2] - ROBOT_POSITION[2] + 0.2, CUBE_POSITION[2] - ROBOT_POSITION[2] + 0.4),
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
        params={"minimal_height": MINIMUM_HEIGHT, "object_cfg": SceneEntityCfg("dex_cube")},
    )
    object_goal_tracking = RewardTermCfg(
        func=mdp.object_goal_distance, weight=16.0,
        params={"std": 0.3, "minimal_height": MINIMUM_HEIGHT, "command_name": "object_pose",
                "robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("dex_cube")},
    )
    object_goal_tracking_fine_grained = RewardTermCfg(
        func=mdp.object_goal_distance, weight=5.0,
        params={"std": 0.05, "minimal_height": MINIMUM_HEIGHT, "command_name": "object_pose",
                "robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("dex_cube")},
    )


@configclass
class EventsCfg(FrankaEventCfg):
    reset_cube = EventTermCfg(
        func=mdp.reset_cube, mode="reset",
        params={"position_range": CUBE_BOUNDS, "yaw_range_deg": OBJECTS["Cube"].get("yaw_range_deg")},
    )



@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=lab_mdp.time_out, time_out=True)
    object_dropped = TerminationTermCfg(
        func=lab_mdp.root_height_below_minimum,
        params={"minimum_height": TABLE_HEIGHT - 0.05, "asset_cfg": SceneEntityCfg("dex_cube")},
    )
    success = TerminationTermCfg(
        func=mdp.lift_object_rl_success,
        params={"object_cfg": SceneEntityCfg("dex_cube"), "robot_cfg": SceneEntityCfg("robot"),
                "command_name": "object_pose", "position_tolerance": 0.05, "rl_training": True},
    )


@configclass
class LiftCubeEnvCfg(ManagerBasedRLEnvCfg):
    scene_config = SCENE_CONFIG
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
    viewer = ViewerCfg(eye=tuple(SCENE_CONFIG["camera"]["eye"]), lookat=tuple(SCENE_CONFIG["camera"]["target"]), origin_type="env")

    def __post_init__(self):
        self.sim.physx = PhysxCfg()
        self.scene.robot = plate_cube_robot_cfg()
        self.scene.ee_frame = FrankaSceneCfg().ee_frame
