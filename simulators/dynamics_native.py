"""Isaac-only implementation. Imported after SimulationApp is running."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import numpy as np

from simulators.dynamics_motion import PoseMotion
from simulators.dynamics_scoring import Criteria, PlacementScorer


def beneath(path, root):
    return path == root or path.startswith(root + "/")


class ContactMonitor:
    """PhysX reports only; observing/scoring never advances the simulation."""

    def __init__(self, stage):
        from omni.physx import get_physx_simulation_interface

        self.stage = stage
        self.cube_path = None
        self.pairs = {}
        self.error = None
        self.report_count = 0
        self.material_paths = set()
        # Materials PhysX applied to the cube's own side of each contact.
        self.cube_material_paths = set()
        self.subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            self._report
        )

    def bind_cube(self, obj):
        from pxr import PhysxSchema

        self.cube_path = str(obj.prim.GetPath())
        self.pairs.clear()
        self.material_paths.clear()
        self.cube_material_paths.clear()
        self.error = None
        self.report_count = 0
        api = PhysxSchema.PhysxContactReportAPI.Apply(obj.prim)
        api.CreateThresholdAttr().Set(0.0)

    def _report(self, headers, data):
        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysicsSchemaTools

        try:
            if self.cube_path is None:
                return
            for header in headers:
                actors = [
                    str(PhysicsSchemaTools.intToSdfPath(value))
                    for value in (header.actor0, header.actor1)
                ]
                colliders = [
                    str(PhysicsSchemaTools.intToSdfPath(value))
                    for value in (header.collider0, header.collider1)
                ]
                cube_side = next(
                    (
                        i
                        for i in (0, 1)
                        if beneath(actors[i], self.cube_path)
                        or beneath(colliders[i], self.cube_path)
                    ),
                    None,
                )
                if cube_side is None:
                    continue
                self.report_count += 1
                key = tuple(sorted(colliders))
                if header.type == ContactEventType.CONTACT_LOST:
                    self.pairs.pop(key, None)
                    continue
                offset, count = header.contact_data_offset, header.num_contact_data
                points = []
                for index in range(offset, offset + count):
                    contact = data[index]
                    materials = [
                        str(PhysicsSchemaTools.intToSdfPath(value))
                        for value in (contact.material0, contact.material1)
                    ]
                    self.material_paths.update(path for path in materials if path)
                    self.cube_material_paths.add(materials[cube_side])
                    points.append(
                        {
                            "position": list(contact.position),
                            "normal": list(contact.normal),
                            "impulse": list(contact.impulse),
                            "separation": float(contact.separation),
                            "materials": materials,
                        }
                    )
                self.pairs[key] = {
                    "other_actor": actors[1 - cube_side],
                    "other_collider": colliders[1 - cube_side],
                    "colliders": colliders,
                    "points": points,
                }
        except Exception as exc:
            # Kit callbacks may swallow exceptions; surface the original failure at the next action.
            self.error = exc

    def contacts(self):
        if self.error is not None:
            raise RuntimeError("Dynamics contact reporting failed") from self.error
        return list(self.pairs.values())

    def require_initial_contacts(self):
        self.contacts()
        if self.report_count == 0:
            raise RuntimeError(
                "No cube contacts were reported during initial settling; scoring is unavailable"
            )

    def touches(self, root):
        return any(
            beneath(pair["other_actor"], root) or beneath(pair["other_collider"], root)
            for pair in self.contacts()
        )

    def close(self):
        self.subscription = None


def material_record(stage, path):
    from pxr import PhysxSchema, UsdPhysics

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return {"path": path, "resolved": False}
    material = UsdPhysics.MaterialAPI(prim)
    extended = PhysxSchema.PhysxMaterialAPI(prim)
    return {
        "path": path,
        "resolved": True,
        "static_friction": material.GetStaticFrictionAttr().Get(),
        "dynamic_friction": material.GetDynamicFrictionAttr().Get(),
        "restitution": material.GetRestitutionAttr().Get(),
        "friction_combine_mode": extended.GetFrictionCombineModeAttr().Get(),
        "restitution_combine_mode": extended.GetRestitutionCombineModeAttr().Get(),
    }


# Nominal cube friction. With ``multiply`` against the scene-default fingers and table (0.5),
# the source contacts get 0.5/0.5: the friction that was actually in effect when run
# 202610291735 learned and qualified its P0, before the cube material reached PhysX.
NOMINAL_STATIC_FRICTION = 1.0
NOMINAL_DYNAMIC_FRICTION = 1.0


def cube_material_values(settings):
    return {
        "static_friction": NOMINAL_STATIC_FRICTION * settings["friction_scale"],
        "dynamic_friction": NOMINAL_DYNAMIC_FRICTION * settings["friction_scale"],
        "restitution": settings["restitution"],
    }


def physx_shape_materials(path):
    """Static friction, dynamic friction and restitution of each live PhysX shape."""
    from isaacsim.core.simulation_manager import SimulationManager

    view = SimulationManager.get_physics_sim_view().create_rigid_body_view(path)
    values = view.get_material_properties()
    values = values.numpy() if hasattr(values, "numpy") else values
    return np.asarray(values, dtype=float).reshape(-1, 3).tolist()


def make_task_class(settings):
    from env.environment.task_env import TaskEnv
    from env.reward_manager.func_parser import Func_Parser
    from env.reward_manager.reward_manager import RewardManager
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

    criteria = Criteria(**settings.get("criteria", {}))

    class PlacementParser(Func_Parser):
        def dynamics_finished(self, args):
            return float(self.env.placement_scorer.success)

    class PlacementRewards(RewardManager):
        def __init__(self, count):
            super().__init__(count)
            self.func_parser = PlacementParser(count)

        def init_state(self):
            super().init_state()
            self.env.prepare_scoring()

        def step(self, env_idx_list=None):
            self.env.motion_tracker.finish_interval()
            sample = self.env.dynamics_sample()
            self.env.placement_scorer.update(sample)
            self.env.dynamics_trace.append(copy.deepcopy(sample))
            super().step(env_idx_list)

    class RedCubeToPlate(TaskEnv):
        def __init__(self, config, app, **kwargs):
            super().__init__(config, app, **kwargs)
            self.step_lim = settings["action_limit"]
            self.reward_manager = PlacementRewards(self.num_envs)
            self.contact_monitor = ContactMonitor(self.stage)
            self.dynamics_trace = []
            self.motion_tracker = None
            instantiate = self.scene_manager.instantiate_object_wrapper

            def create_object(obj_type, prim_path, usd_path, primitive_type, env_id,
                              default_pos, default_ori, scale, inst_cfg):
                cube = inst_cfg.get("label") == "red_cube"
                if cube:
                    material = self.author_cube_material(prim_path)
                obj = instantiate(obj_type, prim_path, usd_path, primitive_type, env_id,
                                  default_pos, default_ori, scale, inst_cfg)
                label = obj.instance_config.get("label")
                if cube:
                    # Override the native 0.5 kg clamp with the requested experimental mass.
                    obj.mass = 0.35 * settings["mass_scale"]
                    obj.set_mass(obj.mass)
                    # Native _setup_physics rebinds its own material after PhysX has
                    # created the shape; restore the USD record of the material in use.
                    self.bind_physics_material(obj.prim, material)
                    self.contact_monitor.bind_cube(obj)
                elif label == "plate":
                    UsdPhysics.RigidBodyAPI(obj.prim).CreateKinematicEnabledAttr().Set(True)
                return obj

            self.scene_manager.instantiate_object_wrapper = create_object

        @staticmethod
        def bind_physics_material(prim, material):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )

        def author_cube_material(self, body_path):
            """Author the cube's material and binding before native construction.

            Native RigidObject.__init__ constructs SingleRigidPrim, which flushes pending
            USD changes into the running PhysX scene, before _setup_physics binds the
            object's material. PhysX then creates the collision shapes with the scene
            default material and does not re-resolve a later binding on the body prim.
            """
            if self.stage.GetPrimAtPath(body_path).IsValid():
                raise RuntimeError(
                    f"Cube prim {body_path} already exists; PhysX may already own its shapes"
                )
            body = self.stage.DefinePrim(body_path, "Xform")
            material = UsdShade.Material.Define(self.stage, body_path + "/dynamics_material")
            values = cube_material_values(settings)
            api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr().Set(values["static_friction"])
            api.CreateDynamicFrictionAttr().Set(values["dynamic_friction"])
            api.CreateRestitutionAttr().Set(values["restitution"])
            extended = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
            extended.CreateFrictionCombineModeAttr().Set("multiply")
            self.bind_physics_material(body, material)
            self.cube_material_path = str(material.GetPath())
            return material

        def sim_step(self, render=True):
            result = super().sim_step(render=render)
            if self.motion_tracker is not None:
                position, orientation = self.cube.get_local_pose()
                self.motion_tracker.record(position, orientation, self.sim_cfg.sim.dt)
            return result

        def _post_setup_scene(self, sim):
            from carb.settings import get_settings

            # SimulationContext disables contact processing during construction.
            # Enable it after that default and before the task's settling steps.
            get_settings().set_bool("/physics/disableContactProcessing", False)
            super()._post_setup_scene(sim)
            self.reward_manager.initialize(self)

        def reset(self, seed=None, options=None):
            # Reset teleports/replaces bodies; seed motion only after settling.
            self.motion_tracker = None
            self.contact_monitor.pairs.clear()
            self.dynamics_trace = []
            super().reset(seed=seed, options=options)
            self.reward_manager.reset()

        def gen_instruction(self, env_idx):
            return [
                "Pick up the red cube with the gripper, lift it clear of the table, "
                "carry it to the plate, then release it inside the plate and withdraw "
                "the gripper so the cube rests stably."
            ]

        def run_reward(self):
            self.reward_manager.check([("dynamics_finished", {})])

        def scene_object(self, label):
            layout = self.scene_manager.layout_manager
            name = layout.get_instance_name(env_idx=0, label=label)
            return layout.get_scene_object(env_idx=0, inst_name=name)

        def prepare_scoring(self):
            if self.sim_cfg.decimation != 1:
                raise ValueError("Dynamics motion scoring requires decimation=1")
            self.cube = self.scene_object("red_cube")
            self.plate = self.scene_object("plate")
            self.plate_path = str(self.plate.prim.GetPath())
            self.table_path = str(self.scene_manager._tables[0].prim.GetPath())
            self.finger_paths = {}
            self.finger_links = {}
            self.robot_paths = []
            for index, robot in enumerate(self.robot_manager.robot_list):
                tree = ET.parse(robot.urdf_path)
                links = [
                    tree.find(f".//joint[@name='{name}']/child").attrib["link"]
                    for name in robot.gripper_joints_name
                ]
                self.robot_paths.append(f"/World/envs/env_0/robot{index}")
                root = self.stage.GetPrimAtPath(self.robot_paths[-1])
                paths = []
                for link in links:
                    matches = [
                        str(prim.GetPath())
                        for prim in Usd.PrimRange(root)
                        if prim.GetName() == link and prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    ]
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"Cannot identify the X5 finger rigid body {link}: {matches}"
                        )
                    paths.append(matches[0])
                arm = self.robot_manager.process_name(robot.arm_name)
                self.finger_paths[arm] = paths
                self.finger_links[arm] = links
            plate_position, _ = self.plate.get_local_pose()
            self.placement_scorer = PlacementScorer(plate_position, criteria)
            self.contact_monitor.require_initial_contacts()
            self.verify_cube_material()
            self.motion_tracker = PoseMotion(*self.cube.get_local_pose())

        def verify_cube_material(self):
            """Compare PhysX's live shapes and contacts with the authored cube material."""
            expected = cube_material_values(settings)
            target = [expected[key] for key in ("static_friction", "dynamic_friction",
                                                "restitution")]
            shapes = physx_shape_materials(str(self.cube.prim.GetPath()))
            if not shapes or not np.allclose(shapes, [target] * len(shapes), atol=1e-6):
                raise RuntimeError(
                    f"PhysX cube shapes use {shapes}, not the authored cube material {target}"
                )
            reported = self.contact_monitor.cube_material_paths
            if reported != {self.cube_material_path}:
                raise RuntimeError(
                    f"PhysX cube contacts report {sorted(reported)}, "
                    f"not {self.cube_material_path}"
                )

        def dynamics_sample(self):
            position, orientation = self.cube.get_local_pose()
            motion = self.motion_tracker.snapshot()
            opening, gripper_position = {}, {}
            for robot in self.robot_manager.robot_list:
                arm = self.robot_manager.process_name(robot.arm_name)
                value = self.robot_manager.get_end_effector_real_val(robot, env_idx_list=[0])[0]
                lo, hi = robot.gripper_scale
                opening[arm] = float((np.mean(value) - lo) / (hi - lo))
                poses = [
                    self.robot_manager.get_link_pose(robot, link, env_idx_list=[0])[0]
                    for link in self.finger_links[arm]
                ]
                gripper_position[arm] = np.mean([pose[:3] for pose in poses], axis=0).tolist()
            return {
                "step": int(self.take_action_cnt[0]),
                "cube_position": np.asarray(position).tolist(),
                "cube_orientation": np.asarray(orientation).tolist(),
                "cube_linear_velocity": motion["linear_velocity"],
                "cube_angular_velocity": motion["angular_velocity"],
                "motion_physics_steps": motion["physics_steps"],
                "cube_solver_linear_velocity": np.asarray(self.cube.get_linear_velocity()).tolist(),
                "cube_solver_angular_velocity": (
                    np.asarray(self.cube.get_angular_velocity()).tolist()
                ),
                "finger_contacts": {
                    arm: [self.contact_monitor.touches(path) for path in paths]
                    for arm, paths in self.finger_paths.items()
                },
                "robot_contact": any(
                    self.contact_monitor.touches(path) for path in self.robot_paths
                ),
                "plate_contact": self.contact_monitor.touches(self.plate_path),
                "table_contact": self.contact_monitor.touches(self.table_path),
                "gripper_opening": opening,
                "gripper_position": gripper_position,
                "contacts": copy.deepcopy(self.contact_monitor.contacts()),
            }

        def dynamics_snapshot(self):
            physics_scene = PhysxSchema.PhysxSceneAPI(
                self.stage.GetPrimAtPath(self.sim_cfg.sim.physics_prim_path)
            )
            cube_materials = {}
            for prim in Usd.PrimRange(self.cube.prim):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                    if not material:
                        raise RuntimeError(
                            f"Cube collider has no physics material: {prim.GetPath()}"
                        )
                    path = str(material.GetPath())
                    cube_materials[str(prim.GetPath())] = material_record(self.stage, path)
            if not cube_materials:
                raise RuntimeError("The red cube has no collision shapes")
            plate_position, plate_orientation = self.plate.get_local_pose()
            robot_state = {}
            for robot, articulation in zip(
                self.robot_manager.robot_list, self.robot_manager.robot_key, strict=True
            ):
                arm = self.robot_manager.process_name(robot.arm_name)
                robot_state[arm] = {
                    "joint_position": articulation.data.joint_pos[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                    "joint_velocity": articulation.data.joint_vel[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                    "root_pose": articulation.data.root_state_w[0, :7]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist(),
                }
            return {
                "sample": self.dynamics_sample(),
                "scoring": self.placement_scorer.snapshot(),
                "velocity_measurement": "physics_step_pose_delta_peak_v1",
                "physics_scene": {
                    "solver_type": physics_scene.GetSolverTypeAttr().Get(),
                    "enable_external_forces_every_iteration": (
                        physics_scene.GetEnableExternalForcesEveryIterationAttr().Get()
                    ),
                    "min_velocity_iteration_count": (
                        physics_scene.GetMinVelocityIterationCountAttr().Get()
                    ),
                    "physics_dt": self.sim_cfg.sim.dt,
                },
                "cube_mass": float(self.cube.get_mass()),
                "cube_collision_materials": cube_materials,
                "cube_physx_shape_materials": physx_shape_materials(str(self.cube.prim.GetPath())),
                "cube_contact_materials": sorted(self.contact_monitor.cube_material_paths),
                "plate_position": np.asarray(plate_position).tolist(),
                "plate_orientation": np.asarray(plate_orientation).tolist(),
                "plate_kinematic": bool(
                    UsdPhysics.RigidBodyAPI(self.plate.prim).GetKinematicEnabledAttr().Get()
                ),
                "contact_materials": [
                    material_record(self.stage, path)
                    for path in sorted(self.contact_monitor.material_paths)
                ],
                "robot_state": robot_state,
            }

        def close(self):
            self.contact_monitor.close()
            super().close()

    return RedCubeToPlate
