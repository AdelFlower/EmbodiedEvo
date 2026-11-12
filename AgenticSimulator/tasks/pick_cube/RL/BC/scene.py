"""复用批量基准的场景配置；不导入或启动基准入口。"""

import math
from pathlib import Path

import numpy as np
import torch


def build_scene(config, num_envs, env_spacing):
    # 必须在 AppLauncher 启动后调用。
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

    objects = {item["name"]: item for item in config["objects"]}
    assets = Path(__file__).resolve().parents[4] / "assets"

    # 每个物体加载后，显式补充质量和物理材质。
    # InteractiveScene 随后会复制整个源环境。
    def make_object_spawn(item):
        def spawn(prim_path, cfg, translation=None, orientation=None):
            prim = sim_utils.spawn_from_usd(
                prim_path,
                cfg,
                translation=translation,
                orientation=orientation,
            )
            path = str(prim.GetPath())

            sim_utils.define_mass_properties(
                path,
                sim_utils.MassPropertiesCfg(
                    mass=float(item["physics"]["mass"]),
                    density=0.0,
                ),
            )

            material_path = f"{path}/BenchmarkPhysicsMaterial"
            material = sim_utils.RigidBodyMaterialCfg(
                static_friction=item["material"]["static_friction"],
                dynamic_friction=item["material"]["dynamic_friction"],
                restitution=item["material"]["restitution"],
                friction_combine_mode="average",
                restitution_combine_mode="average",
            )
            material.func(material_path, material)
            sim_utils.bind_physics_material(path, material_path)
            return prim

        return spawn

    scene_cfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=True,
        filter_collisions=True,
    )

    # 全局共享的地面和灯光。
    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/Ground",
        collision_group=-1,
        spawn=sim_utils.GroundPlaneCfg(),
    )
    scene_cfg.light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )

    # 每个环境自己的固定桌面。
    scene_cfg.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.40, 0.25)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.725)
        ),
    )

    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    robot_cfg.spawn.usd_path = str(
        assets / "robots" / "franka" / "Franka.usd"
    )
    robot_cfg.spawn.articulation_props.fix_root_link = True
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.init_state.pos = (-0.45, 0.0, 0.75)
    robot_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    scene_cfg.robot = robot_cfg

    for name, item in objects.items():
        asset_path = assets / item["asset"]
        if not asset_path.is_file():
            raise FileNotFoundError(asset_path)

        physics = item["physics"]

        # 这是源环境的生成位置，初始化时会对各环境重新采样。
        if "position_range" in item:
            position = tuple(
                sum(item["position_range"][axis]) / 2
                for axis in ("x", "y", "z")
            )
        else:
            position = tuple(item["position"])

        object_cfg = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.UsdFileCfg(
                func=make_object_spawn(item),
                usd_path=str(asset_path.resolve()),
                scale=tuple(item["scale"]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=not physics["gravity"],
                    linear_damping=physics["linear_damping"],
                    angular_damping=physics["angular_damping"],
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=position,
                rot=tuple(
                    item.get("orientation", [1.0, 0.0, 0.0, 0.0])
                ),
            ),
        )
        setattr(scene_cfg, name.lower(), object_cfg)

    scene = InteractiveScene(scene_cfg)
    return scene


def sample_objects(config, episode_ids, seed):
    """每条 episode 独立派生 RNG，采样不依赖并行环境数量。"""
    samples = {item["name"].lower(): [] for item in config["objects"]}
    for episode_id in episode_ids:
        rng = np.random.default_rng(np.random.SeedSequence([seed, episode_id]))
        for item in config["objects"]:
            if "position_range" in item:
                position = [rng.uniform(*item["position_range"][axis]) for axis in ("x", "y", "z")]
            else:
                position = list(item["position"])
            if "yaw_range_deg" in item:
                yaw = math.radians(rng.uniform(*item["yaw_range_deg"]))
                quaternion = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            else:
                quaternion = list(item.get("orientation", [1.0, 0.0, 0.0, 0.0]))
            samples[item["name"].lower()].append(position + quaternion)
    return {name: np.asarray(rows, dtype=np.float32) for name, rows in samples.items()}


def reset_scene(scene, samples, finger_ids, finger_gains):
    """整批重置；必须先恢复增益，再创建新控制器保存其默认值。"""
    robot = scene["robot"]
    stiffness, damping = finger_gains
    robot.write_joint_stiffness_to_sim(stiffness, joint_ids=finger_ids)
    robot.write_joint_damping_to_sim(damping, joint_ids=finger_ids)
    robot.actuators["panda_hand"].stiffness.copy_(stiffness)
    robot.actuators["panda_hand"].damping.copy_(damping)
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone(),
    )
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    robot.set_joint_velocity_target(torch.zeros_like(robot.data.default_joint_vel))
    robot.set_joint_effort_target(torch.zeros_like(robot.data.default_joint_pos))
    for name, poses in samples.items():
        obj = scene[name]
        state = obj.data.default_root_state.clone()
        state[:, :7] = torch.as_tensor(poses, device=state.device)
        state[:, :3] += scene.env_origins
        state[:, 7:] = 0
        obj.write_root_pose_to_sim(state[:, :7])
        obj.write_root_velocity_to_sim(state[:, 7:])
    scene.reset()
