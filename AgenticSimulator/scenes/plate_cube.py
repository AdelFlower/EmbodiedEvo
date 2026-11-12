# import argparse
# from pathlib import Path
# import yaml
# import math
# import random

# from isaaclab.app import AppLauncher

# parser = argparse.ArgumentParser()
# parser.add_argument("--run-dir", required=True)
# parser.add_argument(
#     "--scene-config",
#     default=str(
#         Path(__file__).resolve().parent / "configs" / "plate_cube.yaml"
#     ),
# )
# AppLauncher.add_app_launcher_args(parser)
# args = parser.parse_args()
# with Path(args.scene_config).open(encoding="utf-8") as f:
#     scene_config = yaml.safe_load(f)

# # 必须先启动 Isaac Sim，再导入仿真模块。
# launcher = AppLauncher(args)
# app = launcher.app

# import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation
# from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# ASSETS = Path(__file__).resolve().parents[1] / "assets"
from pathlib import Path
import math
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from AgenticSimulator.scene_api import SceneBundle

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def create_scene(config):
    # 地面
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)

    # 灯光
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/Light", light)

    # 桌面：长 1.2 m、宽 0.8 m、厚 0.05 m。
    # 中心高度 0.725 m，因此上表面高度为 0.75 m。
    # 只设置碰撞，不设置动态刚体，使桌面固定。
    table = sim_utils.CuboidCfg(
        size=(1.2, 0.8, 0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.40, 0.25)
        ),
    )
    table.func(
        "/World/Table",
        table,
        translation=(0.0, 0.0, 0.725),
    )

    for item in config["objects"]:
        prim_path = f"/World/{item['name']}"
        asset_path = ASSETS / item["asset"]

        if not asset_path.is_file():
            raise FileNotFoundError(asset_path)

        physics = item["physics"]

        cfg = sim_utils.UsdFileCfg(
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
        )

        if "position_range" in item:
            ranges = item["position_range"]
            position = tuple(
                random.uniform(*ranges[axis])
                for axis in ("x", "y", "z")
            )
        else:
            position = tuple(item["position"])

        if "yaw_range_deg" in item:
            yaw = math.radians(
                random.uniform(*item["yaw_range_deg"])
            )

            # 绕 Z 轴旋转，四元数顺序为 (w, x, y, z)。
            orientation = (
                math.cos(yaw / 2),
                0.0,
                0.0,
                math.sin(yaw / 2),
            )
        else:
            orientation = tuple(item["orientation"])


        cfg.func(
            prim_path,
            cfg,
            translation=position,
            orientation=orientation,
        )

        # 显式添加质量配置，确保资产原本没有 MassAPI 时也能生效。
        sim_utils.define_mass_properties(
            prim_path,
            sim_utils.MassPropertiesCfg(
                mass=physics["mass"],
                density=0.0,
            ),
        )

        # 创建物理材质，并绑定到物体的碰撞形状。
        material_path = f"{prim_path}/CustomPhysicsMaterial"

        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=item["material"]["static_friction"],
            dynamic_friction=item["material"]["dynamic_friction"],
            restitution=item["material"]["restitution"],
            friction_combine_mode="average",
            restitution_combine_mode="average",
        )

        material_cfg.func(material_path, material_cfg)
        sim_utils.bind_physics_material(prim_path, material_path)


def create_robot():
    cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()

    cfg.prim_path = "/World/Robot"
    cfg.spawn.usd_path = str(
        ASSETS / "robots" / "franka" / "Franka.usd"
    )

    # 固定机器人底座，开启各连杆的重力。
    cfg.spawn.articulation_props.fix_root_link = True
    cfg.spawn.rigid_props.disable_gravity = False

    # 底座位于桌面左侧，机械臂朝 +X 方向。
    # 当前桌面高度为 0.75 m。
    cfg.init_state.pos = (-0.45, 0.0, 0.75)
    cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)

    return Articulation(cfg)

def build_scene(config):
    create_scene(config)
    robot = create_robot()

    cube = RigidObject(
        RigidObjectCfg(prim_path="/World/Cube", spawn=None)
    )
    plate = RigidObject(
        RigidObjectCfg(prim_path="/World/Plate", spawn=None)
    )

    hold_target = None

    def initialize():
        nonlocal hold_target

        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos.clone(),
            robot.data.default_joint_vel.clone(),
        )

        robot.reset()
        cube.reset()
        plate.reset()

        hold_target = robot.data.default_joint_pos.clone()

    def preview_step(dt):
        robot.set_joint_position_target(hold_target)

    return SceneBundle(
        entities={
            "robot": robot,
            "cube": cube,
            "plate": plate,
        },
        initialize=initialize,
        preview_step=preview_step,
    )