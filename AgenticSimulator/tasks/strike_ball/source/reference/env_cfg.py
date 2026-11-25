# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based configuration for striking the stationary ball through the hole."""

from pathlib import Path

import isaaclab.envs.mdp as lab_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg, TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import ArticulationRootPropertiesCfg, CollisionPropertiesCfg, RigidBodyPropertiesCfg, UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from .scene import (
    CAMERA_EYE, CAMERA_TARGET, CONTACT_OFFSET, HOLE_CENTER, SAMPLE_X, SAMPLE_Y, SAMPLE_Z,
    StrikeBall, spawn_ball, spawn_robot, spawn_target_board,
)

ASSET_DIR = Path(__file__).resolve().parent / "assets"


@configclass
class StrikeSceneCfg(InteractiveSceneCfg):
    # robot
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            func=spawn_robot,
            usd_path=str(ASSET_DIR / "UR10_instanceable_pong.usd"),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=ArticulationRootPropertiesCfg(fix_root_link=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            rot=(0.70711, 0.0, 0.0, -0.70711),
            joint_pos={
                "shoulder_pan_joint": -0.29,
                "shoulder_lift_joint": -1.212,
                "elbow_joint": 1.712,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": -0.33,
                "wrist_3_joint": 1.39,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )

    ball = RigidObjectCfg(
        class_type=StrikeBall,
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=UsdFileCfg(
            func=spawn_ball,
            usd_path=str(ASSET_DIR / "Ping_pong_ball.usd"),
            activate_contact_sensors=True,
            collision_props=CollisionPropertiesCfg(contact_offset=CONTACT_OFFSET, rest_offset=0.0),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
        ),
    )

    platform = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.6), collision_props=CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.38, 0.42)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., 0.3)),
    )
    board = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetBoard",
        spawn=sim_utils.RigidObjectSpawnerCfg(
            func=spawn_target_board,
            collision_props=CollisionPropertiesCfg(contact_offset=CONTACT_OFFSET, rest_offset=0.),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=HOLE_CENTER),
    )
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg(), collision_group=-1)
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.))


@configclass
class ActionsCfg:
    arm = lab_mdp.JointPositionActionCfg(
        class_type=mdp.LimitedJointPositionAction, asset_name="robot", joint_names=[".*"],
        scale=0.5, use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # Order is part of the 35-value checkpoint interface.
        joint_position = ObservationTermCfg(func=lab_mdp.joint_pos_rel)
        joint_velocity = ObservationTermCfg(func=lab_mdp.joint_vel)
        ball_from_paddle = ObservationTermCfg(func=mdp.ball_from_paddle)
        paddle_normal = ObservationTermCfg(func=mdp.paddle_normal)
        paddle_velocity = ObservationTermCfg(func=mdp.paddle_velocity)
        ball_velocity = ObservationTermCfg(func=lab_mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("ball")})
        hole_from_ball = ObservationTermCfg(func=mdp.hole_from_ball)
        last_action = ObservationTermCfg(func=lab_mdp.last_action)
        has_hit = ObservationTermCfg(func=mdp.has_hit)
        remaining = ObservationTermCfg(func=mdp.remaining_time)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy = PolicyCfg()


@configclass
class EventsCfg:
    reset_robot = EventTermCfg(
        func=lab_mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (0., 0.), "velocity_range": (0., 0.)},
    )
    place_ball = EventTermCfg(
        func=lab_mdp.reset_root_state_uniform, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {"x": SAMPLE_X, "y": (SAMPLE_Y, SAMPLE_Y), "z": SAMPLE_Z},
            "velocity_range": {},
        },
    )
    # Capture the sampled position only after place_ball has written it.
    reset_shot = EventTermCfg(func=mdp.reset_shot, mode="reset")


@configclass
class TerminationsCfg:
    # The deadline ends this finite-horizon task, including the last physics substep.
    deadline = TerminationTermCfg(func=lab_mdp.time_out, time_out=False)


@configclass
class StrikeBallEnvCfg(ManagerBasedRLEnvCfg):
    scene = StrikeSceneCfg(num_envs=4096, env_spacing=5., replicate_physics=True)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    # Warmup scores paddle motion; shot PPO scores completed strikes.
    rewards = {}
    events = EventsCfg()
    terminations = TerminationsCfg()
    is_finite_horizon = True
    decimation = 4
    episode_length_s = 3.
    sim = sim_utils.SimulationCfg(dt=1 / 200, render_interval=decimation)
    viewer = ViewerCfg(
        eye=CAMERA_EYE, lookat=CAMERA_TARGET, resolution=(1920, 1080), origin_type="env", env_index=0,
    )
