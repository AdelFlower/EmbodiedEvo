"""Isaac Lab construction only; imported by the runtime after AppLauncher."""

import json
from pathlib import Path

# Host diagnostic only (backend_gates --physics-device); formal runs use GPU PhysX.
PHYSICS_DEVICE = 'cuda:0'


def launch_app(run_dir):
    import argparse
    import os
    import sys
    import sysconfig

    from simulators.robodojo import ROOT
    from simulators.robodojo_experience import prepare_experience

    sys.path[:0] = [str(ROOT), str(ROOT / 'XPolicyLab')]
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.headless, args.enable_cameras, args.device = True, False, PHYSICS_DEVICE
    args.kit_args = os.environ.get('EMBODIEDEVO_ROBODOJO_KIT_ARGS', args.kit_args)
    args.kit_args += ' --/log/file=' + str((Path(run_dir)/'kit.log').resolve())
    source = ROOT / 'third_party/IsaacLab/apps/isaaclab.python.kit'
    experience = prepare_experience(source, Path(run_dir) / 'batch.kit',
        Path(sysconfig.get_path('purelib')) / 'isaacsim/exts/isaacsim.asset.importer.urdf/config/extension.toml')
    if experience != source:
        args.experience = str(experience)
    return AppLauncher(args).app


def build_scene(num_envs, target):
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext, PhysxCfg
    from omegaconf import OmegaConf
    from pxr import Usd, UsdPhysics, UsdShade, PhysxSchema, Gf
    from env.environment.base_env import DEFAULT_PHYSX_CONFIG
    from env.robot_manager.robot_manager import RobotManager
    from utils.load_file import load_yaml
    from simulators.robodojo import ROOT
    from integrations.friction_rl.control import SCENE_PATH

    nominal = json.loads(SCENE_PATH.read_text())
    robot_cfg = OmegaConf.create(load_yaml(str(ROOT / 'env_cfg/robot/dual_x5.yml')))
    for cfg in robot_cfg.robots:
        cfg.need_planner = False  # only the controller gate constructs the reference solver
    manager = RobotManager(num_envs=num_envs, env_spacing=7, config=robot_cfg, dt=.004, device=PHYSICS_DEVICE)
    sim = SimulationContext(SimulationCfg(dt=.004, device=PHYSICS_DEVICE, render_interval=10,
        use_fabric=False, physx=PhysxCfg(**DEFAULT_PHYSX_CONFIG, enable_external_forces_every_iteration=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.5, dynamic_friction=.5,
                                                       restitution=0., friction_combine_mode='average',
                                                       restitution_combine_mode='average')))
    cfg = manager._get_SceneCfg(num_envs, 7, replicate_physics=True)
    for i in range(2):
        robot = getattr(cfg, f'robot{i}')
        robot.spawn.activate_contact_sensors = True
    table = nominal['Table']

    @sim_utils.clone
    def table_spawn(path, spawn_cfg, translation=None, orientation=None):
        # The native table is a tessellated Mesh cube, not an analytic collider.
        import omni.kit.commands
        from omni.physx.scripts import physicsUtils
        omni.kit.commands.execute('CreateMeshPrimCommand', prim_type='Cube', prim_path=path,
                                  u_patches=10, v_patches=10, w_patches=10)
        prim = sim.stage.GetPrimAtPath(path)
        physicsUtils.setup_transform_as_scale_orient_translate(prim)
        physicsUtils.set_or_add_scale_op(prim, scale=Gf.Vec3f(*spawn_cfg.size))
        physicsUtils.set_or_add_translate_op(prim, translate=Gf.Vec3f(*translation))
        physicsUtils.set_or_add_orient_op(prim, orient=Gf.Quatf(orientation[0], Gf.Vec3f(*orientation[1:])))
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set('convexDecomposition')
        return prim

    cfg.table = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Table',
        spawn=sim_utils.CuboidCfg(func=table_spawn, size=tuple(table['scale'])),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(table['default_pos']), rot=tuple(table['default_ori'])))
    spec = ({'cube_mass_kg': .35, 'cube_material': {'static_friction': 1., 'dynamic_friction': 1., 'restitution': 0.}}
            if target is None else target._spec)

    masses = {'cube': spec['cube_mass_kg'], 'plate': .5}

    def define_mass(path, name):
        # spawn_from_usd only *modifies* MassAPI, which these assets lack; without this the
        # density default (1000 kg/m^3) silently applies (cube 0.0156 kg instead of 0.35 kg).
        sim_utils.define_mass_properties(path, sim_utils.MassPropertiesCfg(mass=masses[name]))

    @sim_utils.clone
    def plate_spawn(path, spawn_cfg, translation=None, orientation=None):
        prim = sim_utils.spawn_from_usd(path, spawn_cfg, translation, orientation)
        define_mass(path, 'plate')
        return prim

    @sim_utils.clone
    def cube_spawn(path, spawn_cfg, translation=None, orientation=None):
        prim = sim_utils.spawn_from_usd(path, spawn_cfg, translation, orientation)
        define_mass(path, 'cube')
        material_path = str(prim.GetPath())+'/dynamics_material'
        sim_utils.spawn_rigid_body_material(material_path, sim_utils.RigidBodyMaterialCfg(
            **spec['cube_material'], friction_combine_mode='multiply', restitution_combine_mode='average'))
        material = UsdShade.Material.Get(sim.stage, material_path)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
        return prim

    for name, index in (('cube', 4), ('plate', 2)):
        row = nominal['Rigid'][name][0]
        spawn = sim_utils.UsdFileCfg(
            usd_path=str(ROOT / f'Assets/Object/RoboDojo/Rigid/{name}/{index:05d}/object.usdz'),
            scale=tuple(row['scale']), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=name == 'plate'))
        spawn.func = cube_spawn if name == 'cube' else plate_spawn
        # Plate retains its imported shape bindings, just as native construction
        # creates PhysX shapes before applying the later per-body scene material.
        setattr(cfg, name, RigidObjectCfg(prim_path='{ENV_REGEX_NS}/'+name.capitalize(), spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(row['default_pos']), rot=tuple(row['default_ori']))))
    scene = InteractiveScene(cfg)

    def body_paths(root):
        return [str(p.GetPath()) for p in Usd.PrimRange(sim.stage.GetPrimAtPath(root)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]

    paths = {}
    for name in ('cube', 'plate'):
        bodies = body_paths('/World/envs/env_0/'+name.capitalize())
        if len(bodies) != 1:
            raise ValueError(f'{name} asset must have exactly one rigid body')
        paths[name] = bodies[0]
    finger_paths, arm_paths = [], []
    for i in range(2):
        robot_bodies = body_paths(f'/World/envs/env_0/robot{i}')
        arm_paths += [p for p in robot_bodies if p.rsplit('/', 1)[-1] not in ('link7', 'link8')]
        for link in ('link7', 'link8'):
            matches = [p for p in robot_bodies if p.rsplit('/', 1)[-1] == link]
            if len(matches) != 1:
                raise ValueError(f'robot{i} must have exactly one rigid body for {link}')
            finger_paths.append(matches[0])
    paths['fingers'] = finger_paths
    paths['robot_bodies'] = arm_paths
    # PhysX filter patterns must match exactly one body per env, so every robot body is listed.
    filters = ['/World/envs/env_0/Table', paths['plate'], *arm_paths, *finger_paths]
    def expression(p):
        return p.replace('/World/envs/env_0', scene.env_regex_ns)
    scene.sensors['cube_contacts'] = ContactSensor(ContactSensorCfg(
        prim_path=expression(paths['cube']), update_period=0.,
        filter_prim_paths_expr=[expression(p) for p in filters], max_contact_data_count_per_prim=128))
    for i, path in enumerate(finger_paths):
        scene.sensors[f'pad{i}'] = ContactSensor(ContactSensorCfg(prim_path=expression(path), update_period=0.))
    # Contact processing is disabled by SimulationContext construction.
    import carb
    carb.settings.get_settings().set_bool('/physics/disableContactProcessing', False)
    sim.reset()
    for name in ('cube', 'plate'):
        effective = scene[name].root_physx_view.get_masses()
        if not torch.allclose(effective, torch.full_like(effective, masses[name]), rtol=1e-5):
            raise ValueError(f'effective {name} mass differs from the authored mass')
    if scene.sensors['cube_contacts'].contact_physx_view.filter_count != len(filters):
        raise ValueError('cube contact view must resolve every partner body')
    # Use link-frame states (not inertial/COM frames) throughout the runtime.
    arms = [scene[f'robot{i}'] for i in range(2)]
    ids = []
    for art in arms:
        arm_ids, names = art.find_joints([f'joint{i}' for i in range(1, 7)], preserve_order=True)
        grip_ids, _ = art.find_joints(['joint7', 'joint8'], preserve_order=True)
        body_ids, _ = art.find_bodies(['link6', 'link7', 'link8'], preserve_order=True)
        if names != [f'joint{i}' for i in range(1, 7)]:
            raise ValueError('X5 arm joints do not match the declared controller order')
        ids.append((arm_ids, grip_ids, body_ids))
    return sim, scene, manager, ids, paths


def source_build_report(sim, scene, ids, paths):
    """Inspect effective live data once; never called for a private target."""
    from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema
    from simulators.dynamics_native import material_record
    import torch
    import isaaclab
    import curobo
    from isaaclab.utils import get_isaac_sim_version
    report = {'verification': 'runtime inspection; parity requires separate gates', 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'isaaclab': isaaclab.__version__, 'isaacsim': str(get_isaac_sim_version()),
              'curobo': curobo.__version__, 'backend': 'isaaclab_gpu_physx', 'modules': [],
              'arm_dofs': 6, 'finger_dofs': 2, 'control_substeps': 10, 'dt': sim.get_physics_dt(),
              'device': str(sim.device), 'paths': paths, 'bodies': {}, 'robots': {}}
    for name in ('cube', 'plate'):
        body = scene[name]
        report['bodies'][name] = {'masses': body.root_physx_view.get_masses().cpu().tolist(),
            'shape_materials': body.root_physx_view.get_material_properties().cpu().tolist(),
            'coms': body.root_physx_view.get_coms().cpu().tolist(),
            'inertias': body.root_physx_view.get_inertias().cpu().tolist()}
    for i, (arm_ids, grip_ids, body_ids) in enumerate(ids):
        art = scene[f'robot{i}']
        report['robots'][str(i)] = {'joints': art.joint_names, 'bodies': art.body_names,
            'default_joint_position': art.data.default_joint_pos[0].cpu().tolist(),
            'joint_limits': art.data.joint_pos_limits[0].cpu().tolist(),
            'stiffness': art.data.joint_stiffness[0].cpu().tolist(),
            'damping': art.data.joint_damping[0].cpu().tolist(),
            'effort_limit': art.data.joint_effort_limits[0].cpu().tolist(),
            'shape_materials': art.root_physx_view.get_material_properties()[0].cpu().tolist(),
            'masses': art.root_physx_view.get_masses()[0].cpu().tolist(),
            'coms': art.root_physx_view.get_coms()[0].cpu().tolist(),
            'inertias': art.root_physx_view.get_inertias()[0].cpu().tolist()}
    bindings = {}
    from pxr import UsdShade
    for prim in Usd.PrimRange(sim.stage.GetPrimAtPath('/World/envs/env_0')):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            bindings[str(prim.GetPath())] = material_record(sim.stage, str(material.GetPath())) if material else {'scene_default': True}
    report['collision_bindings'] = bindings
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    report['usd_authored_world_bounds'] = {}
    for name in ('Cube', 'Plate', 'Table'):
        extent = box.ComputeWorldBound(sim.stage.GetPrimAtPath('/World/envs/env_0/'+name)).ComputeAlignedRange()
        report['usd_authored_world_bounds'][name] = [list(extent.GetMin()), list(extent.GetMax())]
    physics = PhysxSchema.PhysxSceneAPI(sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path))
    report['effective_solver'] = {'solver_type': physics.GetSolverTypeAttr().Get(),
        'external_forces_every_iteration': physics.GetEnableExternalForcesEveryIterationAttr().Get(),
        'min_position_iterations': physics.GetMinPositionIterationCountAttr().Get(),
        'max_position_iterations': physics.GetMaxPositionIterationCountAttr().Get(),
        'min_velocity_iterations': physics.GetMinVelocityIterationCountAttr().Get(),
        'max_velocity_iterations': physics.GetMaxVelocityIterationCountAttr().Get()}
    return report
