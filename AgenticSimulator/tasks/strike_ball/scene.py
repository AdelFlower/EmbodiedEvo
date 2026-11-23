"""Shared geometry, materials and scoring for the UR10 strike scene (metres)."""

import math

import isaaclab.sim as sim_utils
import omni.usd
import torch
from isaaclab.assets import RigidObject
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

HOLE_CENTER = (0.15, -1.50, 0.93)
HOLE_RADIUS = 0.024
BOARD_THICKNESS = 0.01
BOARD_SEGMENTS = 64
FLIGHT_TIME = 0.6
BALL_RADIUS = 0.020
CONTACT_OFFSET = 0.00025
SAMPLE_Y = -0.90
PADDLE_OFFSET = (0.0, 0.265, 0.0)
CAMERA_EYE = (1.80, -2.45, 1.88)
CAMERA_TARGET = (0.15, -1.20, 1.46)
SAMPLE_X = (-0.05, 0.35)
SAMPLE_Z = (HOLE_CENTER[2] + BOARD_THICKNESS / 2, 1.11)


@sim_utils.clone
def spawn_target_board(path, cfg, translation, orientation):
    """用整板网格显示外观，用环绕通孔的实心分块计算碰撞。"""
    stage = omni.usd.get_context().get_stage()
    board_width = 0.20
    segments = BOARD_SEGMENTS
    # 让每段孔壁到中心的最短距离等于设定的孔半径。
    inner_vertex_radius = HOLE_RADIUS / math.cos(math.pi / segments)
    top = BOARD_THICKNESS / 2
    bottom = -top
    points = []
    indices = []
    for i in range(segments):
        angle = math.tau * i / segments
        dx, dy = math.cos(angle), math.sin(angle)
        outer_radius = (board_width / 2) / max(abs(dx), abs(dy))
        inner_x, inner_y = inner_vertex_radius * dx, inner_vertex_radius * dy
        outer_x, outer_y = outer_radius * dx, outer_radius * dy
        points.extend([
            (inner_x, inner_y, top), (outer_x, outer_y, top),
            (outer_x, outer_y, bottom), (inner_x, inner_y, bottom),
        ])

        # 连接相邻截面，依次形成上表面、外壁、下表面和孔壁。
        current = 4 * i
        following = 4 * ((i + 1) % segments)
        for start, end in ((0, 1), (1, 2), (2, 3), (3, 0)):
            indices.extend((current + start, current + end, following + end, following + start))

    board = UsdGeom.Mesh.Define(stage, path)
    board.AddTranslateOp().Set(translation)
    board.AddOrientOp().Set(Gf.Quatf(orientation[0], Gf.Vec3f(*orientation[1:])))
    board.CreatePointsAttr(points)
    board.CreateFaceVertexCountsAttr([4] * (4 * segments))
    board.CreateFaceVertexIndicesAttr(indices)
    board.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    board.CreateDisplayColorAttr([(0.26, 0.42, 0.50)])

    # 相邻截面各提供四个顶点，围成一块从孔壁延伸到板边的凸体。
    wedge_faces = [
        0, 1, 5, 4,  1, 2, 6, 5,  # 上表面和外壁
        2, 3, 7, 6,  3, 0, 4, 7,  # 下表面和孔壁
        3, 2, 1, 0,  4, 5, 6, 7,  # 两端截面
    ]
    for i in range(segments):
        current = 4 * i
        following = 4 * ((i + 1) % segments)
        wedge = UsdGeom.Mesh.Define(stage, board.GetPath().AppendChild(f"Wedge_{i:02d}"))
        wedge.CreatePointsAttr(points[current:current + 4] + points[following:following + 4])
        wedge.CreateFaceVertexCountsAttr([4] * 6)
        wedge.CreateFaceVertexIndicesAttr(wedge_faces)
        wedge.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        wedge.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(wedge.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(wedge.GetPrim()).CreateApproximationAttr("convexHull")
    material = sim_utils.RigidBodyMaterialCfg(restitution=0.9, restitution_combine_mode="max")
    material_path = f"{path}/PhysicsMaterial"
    material.func(material_path, material)
    sim_utils.bind_physics_material(path, material_path)
    sim_utils.modify_collision_properties(path, cfg.collision_props)
    return board.GetPrim()


def spawn_robot(path, cfg, translation, orientation):
    prim = sim_utils.spawn_from_usd(path, cfg, translation, orientation)
    surface = UsdPhysics.MaterialAPI.Get(prim.GetStage(), prim.GetPath().AppendPath("Looks/OmniPBR"))
    surface.CreateStaticFrictionAttr().Set(0.8)
    surface.CreateDynamicFrictionAttr().Set(0.8)
    return prim


def spawn_ball(path, cfg, translation, orientation):
    prim = sim_utils.spawn_from_usd(path, cfg, translation, orientation)
    body = prim.GetChild("Ball")
    if cfg.rigid_props.kinematic_enabled:
        PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateEnableCCDAttr().Set(False)
    material, _ = UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial("physics")
    surface = UsdPhysics.MaterialAPI(material.GetPrim())
    surface.CreateStaticFrictionAttr().Set(0.4)
    surface.CreateDynamicFrictionAttr().Set(0.4)
    return prim


class StrikeBall(RigidObject):
    """A spin-locked ball with a post-physics callback for training contacts."""

    after_physics_step = None

    def _initialize_impl(self):
        super()._initialize_impl()
        if not self.cfg.spawn.rigid_props.kinematic_enabled:
            # PhysX treats zero principal inertia as infinite resistance to rotation.
            inertia = self.data.default_inertia.zero_()
            self.root_physx_view.set_inertias(inertia, torch.arange(self.num_instances, device=inertia.device))

    def update(self, dt):
        super().update(dt)
        # InteractiveScene refreshes the articulation before rigid objects.
        if dt > 0 and self.after_physics_step is not None:
            self.after_physics_step()


def make_contact_view(robot, root_pattern):
    from omni.physics.tensors import create_simulation_view

    view = create_simulation_view("torch")
    wrist = robot.body_names.index("wrist_3_link")
    wrist_path = robot.root_physx_view.link_paths[0][wrist]
    robot_suffix = wrist_path[wrist_path.index("/Robot/"):]
    contacts = view.create_rigid_contact_view(
        f"{root_pattern}/Ball/Ball", filter_patterns=[root_pattern + robot_suffix],
        max_contact_data_count=16 * robot.num_instances,
    )
    if contacts.sensor_count != robot.num_instances or contacts.filter_count != 1:
        raise RuntimeError("Expected one ball/wrist contact pair per environment")
    return view, contacts


def make_board_contact_view(view, root_pattern, num_envs):
    # Each static convex collider is a separate PhysX actor.
    wedges = [f"{root_pattern}/TargetBoard/Wedge_{i:02d}" for i in range(BOARD_SEGMENTS)]
    contacts = view.create_rigid_contact_view(f"{root_pattern}/Ball/Ball", filter_patterns=wedges)
    if contacts.sensor_count != num_envs or contacts.filter_count != BOARD_SEGMENTS:
        raise RuntimeError("Expected each ball to track all colliders of its own target board")
    return contacts


def paddle_contacts(contacts, robot, wrist):
    """Reduce the ragged contact points on the GPU, one boolean per ball."""
    from isaaclab.utils.math import quat_rotate_inverse

    forces, points, _, _, counts, offsets = contacts.get_contact_data(dt=1.0)
    counts = counts.flatten().long()
    env_ids = torch.arange(counts.numel(), device=counts.device).repeat_interleave(counts)
    first = counts.cumsum(0) - counts
    slots = torch.arange(env_ids.numel(), device=counts.device) + (offsets.flatten() - first)[env_ids]
    local = quat_rotate_inverse(
        robot.data.body_link_quat_w[env_ids, wrist],
        points[slots] - robot.data.body_link_pos_w[env_ids, wrist],
    )
    # The wrist cylinder ends at local y=0.0921 m, before the paddle.
    valid = (local[:, 1] > 0.093) & (forces.flatten()[slots].abs() > 0)
    touching = torch.zeros_like(counts, dtype=torch.bool)
    touching[env_ids[valid]] = True
    return touching


def target_velocity(start, hole, gravity, dt):
    """Reference launch velocity for the semi-implicit PhysX gravity step."""
    return (hole - start) / FLIGHT_TIME - start.new_tensor(gravity) * ((FLIGHT_TIME + dt) / 2)


def check_passage(previous, position, hole, pending):
    """Interpolate the descending mid-plane crossing; require full clearance below."""
    dz = previous[..., 2] - position[..., 2]
    down = (previous[..., 2] >= hole[..., 2]) & (position[..., 2] < hole[..., 2])
    fraction = ((previous[..., 2] - hole[..., 2]) / dz.clamp_min(1e-12)).clamp(0.0, 1.0)
    crossing = torch.lerp(previous[..., :2], position[..., :2], fraction.unsqueeze(-1))
    error = (crossing - hole[..., :2]).norm(dim=-1)
    pending = torch.where(down, error <= HOLE_RADIUS - BALL_RADIUS, pending)
    near_hole = (position[..., :2] - hole[..., :2]).norm(dim=-1) <= HOLE_RADIUS
    pending = pending & near_hole & (position[..., 2] <= hole[..., 2])
    passed = pending & (position[..., 2] + BALL_RADIUS < hole[..., 2] - BOARD_THICKNESS / 2)
    return down, error, pending, passed
