from pathlib import Path
import math
import random

import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from AgenticSimulator.scene_api import SceneBundle

ASSETS = Path(__file__).resolve().parents[1] / "assets"

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "cube.yaml"
TABLE_SIZE = (1.2, 0.8, 0.05)
TABLE_POSITION = (0.0, 0.0, 0.725)
TABLE_HEIGHT = TABLE_POSITION[2] + TABLE_SIZE[2] / 2
ROBOT_POSITION = (-0.45, 0.0, TABLE_HEIGHT)
ROBOT_ORIENTATION = (1.0, 0.0, 0.0, 0.0)


def load_scene_config():
    """Read the configuration shared by the standalone scene and PPO task."""
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@sim_utils.clone
def spawn_configured_object(prim_path, cfg, translation=None, orientation=None):
    """Spawn the local USD with the configured mass and collision material."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation)
    sim_utils.define_mass_properties(
        prim_path, sim_utils.MassPropertiesCfg(mass=cfg.custom_mass, density=0.0),
    )
    material_path = f"{prim_path}/CustomPhysicsMaterial"
    cfg.custom_material.func(material_path, cfg.custom_material)
    sim_utils.bind_physics_material(prim_path, material_path)
    return prim


@configclass
class ConfiguredObjectCfg(sim_utils.UsdFileCfg):
    func = spawn_configured_object
    custom_mass: float = 1.0
    custom_material: sim_utils.RigidBodyMaterialCfg | None = None


def object_spawn_config(item):
    """Use the same local asset and physics for standalone and parallel scenes."""
    asset_path = ASSETS / item["asset"]
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)
    physics = item["physics"]
    material = item["material"]
    return ConfiguredObjectCfg(
        usd_path=str(asset_path.resolve()), scale=tuple(item["scale"]),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True, kinematic_enabled=False,
            disable_gravity=not physics["gravity"],
            linear_damping=physics["linear_damping"], angular_damping=physics["angular_damping"],
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        custom_mass=physics["mass"],
        custom_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=material["static_friction"], dynamic_friction=material["dynamic_friction"],
            restitution=material["restitution"], friction_combine_mode="average", restitution_combine_mode="average",
        ),
    )


def object_pose_bounds(item):
    """Return configured position bounds and nominal WXYZ orientation."""
    if "position_range" in item:
        bounds = [tuple(item["position_range"][axis]) for axis in ("x", "y", "z")]
    else:
        bounds = [(value, value) for value in item["position"]]
    return bounds, tuple(item.get("orientation", (1.0, 0.0, 0.0, 0.0)))


def robot_config(prim_path="/World/Robot"):
    """Return the cube scene's local Franka high-PD configuration."""
    cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
    cfg.prim_path = prim_path
    cfg.spawn.usd_path = str(ASSETS / "robots" / "franka" / "Franka.usd")
    if not Path(cfg.spawn.usd_path).is_file():
        raise FileNotFoundError(cfg.spawn.usd_path)
    cfg.spawn.articulation_props.fix_root_link = True
    cfg.spawn.rigid_props.disable_gravity = False
    cfg.init_state.pos = ROBOT_POSITION
    cfg.init_state.rot = ROBOT_ORIENTATION
    return cfg


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
        size=TABLE_SIZE,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.40, 0.25)
        ),
    )
    table.func(
        "/World/Table",
        table,
        translation=TABLE_POSITION,
    )

    for item in config["objects"]:
        prim_path = f"/World/{item['name']}"
        cfg = object_spawn_config(item)

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


def create_robot():
    return Articulation(robot_config())


def build_scene(config):
    create_scene(config)
    robot = create_robot()

    cube = RigidObject(
        RigidObjectCfg(prim_path="/World/Cube", spawn=None)
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

        hold_target = robot.data.default_joint_pos.clone()

    def preview_step(dt):
        robot.set_joint_position_target(hold_target)

    return SceneBundle(
        entities={
            "robot": robot,
            "cube": cube,
        },
        initialize=initialize,
        preview_step=preview_step,
    )
