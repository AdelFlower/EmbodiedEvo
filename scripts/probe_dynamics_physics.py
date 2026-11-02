"""Model-free GPU check of the cube material PhysX actually uses.

The host invocation launches one fresh simulator per condition (source, target) and
writes host-private records under --run-dir. Each simulator:

1. resets the experiment scene and records the live PhysX shape materials of the
   cube and plate, plus the materials named by contact reports;
2. gives the resting cube a fixed horizontal velocity and records how far it slides
   (deceleration on the table is set by dynamic friction);
3. performs two Playground resets and repeats both measurements after each;
4. creates two extra cubes far from the task, one in native RigidObject order and one
   whose material is bound before construction, and records their live materials.

No agent, model or policy runs, and the experiment scene is not used afterwards.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from orchestrator.config import REPO

SLIDE_SPEED = 0.3
SLIDE_STEPS = 200
CONDITIONS = ("source", "target")


def contact_record(monitor):
    sides = []
    for pair in monitor.contacts():
        cube_side = 0 if pair["colliders"][0].startswith(monitor.cube_path) else 1
        sides.append({
            "other_collider": pair["other_collider"],
            "cube_materials": sorted({p["materials"][cube_side] for p in pair["points"]}),
            "other_materials": sorted({p["materials"][1 - cube_side] for p in pair["points"]}),
        })
    return sides


def live_state(env):
    from simulators.dynamics_native import material_record, physx_shape_materials

    native = env.native
    cube_path = str(native.cube.prim.GetPath())
    return {
        "cube_path": cube_path,
        "authored_cube_material": material_record(native.stage, native.cube_material_path),
        "cube_physx_shape_materials": physx_shape_materials(cube_path),
        "cube_contact_materials": sorted(native.contact_monitor.cube_material_paths),
        "plate_physx_shape_materials": physx_shape_materials(native.plate_path),
        "contacts": contact_record(native.contact_monitor),
        "contact_materials": [
            material_record(native.stage, path)
            for path in sorted(native.contact_monitor.material_paths)
        ],
    }


def slide(env, gravity=9.81):
    """Slide the resting cube along +x on the table and return the measured response."""
    import numpy as np
    import torch

    cube = env.native.cube
    monitor = env.native.contact_monitor
    start = np.asarray(cube.get_local_pose()[0], dtype=float)
    cube.set_linear_velocity(torch.tensor([SLIDE_SPEED, 0.0, 0.0]))
    dt = env.native.sim_cfg.sim.dt
    trace = []
    for step in range(1, SLIDE_STEPS + 1):
        env.native.sim_step(render=False)
        position, orientation = (np.asarray(v, dtype=float) for v in cube.get_local_pose())
        velocity = np.asarray(cube.get_linear_velocity(), dtype=float).reshape(-1)
        angular = np.asarray(cube.get_angular_velocity(), dtype=float).reshape(-1)
        # Angle between the cube's local z axis and world z (w, x, y, z quaternion).
        w, x, y, _ = orientation / np.linalg.norm(orientation)
        tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))
        trace.append({
            "step": step, "x": float(position[0] - start[0]), "y": float(position[1] - start[1]),
            "z": float(position[2] - start[2]), "vx": float(velocity[0]),
            "angular": angular.tolist(), "tilt_deg": tilt,
            "pairs": [{"other": pair["other_collider"], "points": len(pair["points"])}
                      for pair in monitor.contacts()],
        })
    sliding = [s for s in trace if 0.2 * SLIDE_SPEED < s["vx"] < 0.8 * SLIDE_SPEED]
    deceleration = None
    if len(sliding) >= 2:
        times = np.array([s["step"] * dt for s in sliding])
        speeds = np.array([s["vx"] for s in sliding])
        deceleration = float(-np.polyfit(times, speeds, 1)[0])
    final = trace[-1]
    return {
        "initial_speed": SLIDE_SPEED,
        "physics_dt": dt,
        "distance": final["x"],
        "final_vx": final["vx"],
        "max_abs_z_change": max(abs(s["z"]) for s in trace),
        "max_tilt_deg": max(s["tilt_deg"] for s in trace),
        "other_colliders": sorted({p["other"] for s in trace for p in s["pairs"]}),
        "deceleration": deceleration,
        "effective_dynamic_friction": None if deceleration is None else deceleration / gravity,
        "trace": trace,
    }


def ordering_comparison(env):
    """Same asset, two construction orders, in the running scene."""
    from env.scene_manager.objects.rigid import RigidObject
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    from simulators.dynamics_native import physx_shape_materials
    from simulators.robodojo import robodojo_cwd

    native = env.native
    stage = native.stage
    asset = native.cube.usd_path
    records = {}

    def material(path, value):
        shade = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(shade.GetPrim())
        api.CreateStaticFrictionAttr().Set(value)
        api.CreateDynamicFrictionAttr().Set(value)
        api.CreateRestitutionAttr().Set(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(shade.GetPrim()).CreateFrictionCombineModeAttr().Set(
            "multiply"
        )
        return shade

    def construct(path, position, native_value):
        return RigidObject(
            prim_path=path, usd_path=asset,
            inst_config={"label": path.rsplit("/", 1)[-1], "physics": {
                "type": "rigid", "mass": 0.35, "static_friction": native_value,
                "dynamic_friction": native_value, "restitution": 0.0,
            }},
            env_origin=[0.0, 0.0, 0.0], default_pos=position, default_ori=(1.0, 0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        )

    with robodojo_cwd():
        # Native order: the vendor material (0.11) is bound after construction.
        first = construct("/World/probe/native/probe_0_1", (0.55, 0.3, 0.78), 0.11)
        # Bound first: our material (0.12) before construction; vendor's (0.13) after.
        path = "/World/probe/prebound/probe_0_2"
        body = stage.DefinePrim(path, "Xform")
        UsdShade.MaterialBindingAPI.Apply(body).Bind(
            material(path + "/dynamics_material", 0.12),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics",
        )
        second = construct(path, (0.55, 0.1, 0.78), 0.13)
        for obj in (first, second):
            obj.initialize()
        for _ in range(30):
            native.sim_step(render=False)
    for name, obj, legend in (
        ("native_order", first, {"vendor_bound_after_construction": 0.11}),
        ("bound_before_construction", second, {"bound_before_construction": 0.12,
                                               "vendor_bound_after_construction": 0.13}),
    ):
        records[name] = {
            "path": str(obj.prim.GetPath()), "authored": legend,
            "physx_shape_materials": physx_shape_materials(str(obj.prim.GetPath())),
        }
    records["scene_default"] = 0.5
    return records


def probe(condition, run):
    from orchestrator.dynamics_repair import environment_settings, load_config
    from orchestrator.files import write_json
    from simulators.dynamics_task import build_environment

    config = load_config(REPO / "tasks/DynamicsRepair/experiment.example.json")
    settings = environment_settings(config, config["target_friction_scale"])[condition]
    record = {"condition": condition, "settings": {
        key: settings[key] for key in ("friction_scale", "mass_scale", "restitution")
    }, "stages": {}}
    env = None
    try:
        env = build_environment(settings)
        env.execution_phase = "playground"
        env.reset(seed=0)
        record["stages"]["initial"] = live_state(env)
        record["stages"]["initial"]["slide"] = slide(env)
        for stage in ("after_reset", "after_second_reset"):
            env.reset_from_program()
            record["stages"][stage] = live_state(env)
            record["stages"][stage]["slide"] = slide(env)
        record["ordering"] = ordering_comparison(env)
        record["status"] = "complete"
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        raise
    finally:
        write_json(run / "probe.json", record)
        if env is not None:
            env.close()
    return record


def launch(run, gpu, conditions=CONDITIONS):
    """Host side: one fresh simulator per condition, then a comparison."""
    from orchestrator.files import run_lock, write_json
    from orchestrator.resources import gpu_lease

    interpreter = REPO / ".venv-robodojo/bin/python"
    summary = {"purpose": "cube_material_physx_check", "model_calls": 0, "conditions": {}}
    with run_lock(run), gpu_lease(gpu):
        # Keep the exact instrument beside its records.
        (run / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        for condition in conditions:
            directory = run / condition
            directory.mkdir()
            with (directory / "stdout.log").open("wb") as out, \
                    (directory / "stderr.log").open("wb") as err:
                code = subprocess.run(
                    [str(interpreter), "-m", "scripts.probe_dynamics_physics", "--inside",
                     "--condition", condition, "--run-dir", str(directory), "--gpu", gpu],
                    cwd=REPO, stdin=subprocess.DEVNULL, stdout=out, stderr=err, check=False,
                ).returncode
            record = directory / "probe.json"
            result = json.loads(record.read_text()) if record.is_file() else None
            summary["conditions"][condition] = {
                "returncode": code,
                "status": result["status"] if result else "no_record",
                **({"error": result["error"]} if result and "error" in result else {}),
                **({
                    stage: {
                        "cube_physx_shape_materials": values["cube_physx_shape_materials"],
                        "cube_contact_materials": values["cube_contact_materials"],
                        "plate_physx_shape_materials": values["plate_physx_shape_materials"],
                        "other_contact_materials": sorted({
                            m for pair in values["contacts"] for m in pair["other_materials"]
                        }),
                        "slide_distance": values["slide"]["distance"],
                        "effective_dynamic_friction":
                            values["slide"]["effective_dynamic_friction"],
                    }
                    for stage, values in (result or {}).get("stages", {}).items()
                    if "slide" in values
                }),
                **({"ordering": result["ordering"]} if result and "ordering" in result else {}),
            }
        write_json(run / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    args = parser.parse_args(argv)
    if args.inside:
        probe(args.condition, args.run_dir.resolve())
        return 0
    run = args.run_dir.absolute()
    if not re.match(r"^\d{12}_", run.name):
        raise ValueError("--run-dir must start with YYYYMMDDHHMM_")
    datetime.strptime(run.name[:12], "%Y%m%d%H%M")
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Use a fresh timestamp-prefixed run directory")
    summary = launch(run, args.gpu, args.conditions)
    print(json.dumps(summary, indent=2))
    ok = all(entry["status"] == "complete" for entry in summary["conditions"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
