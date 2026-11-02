"""Model-free GPU check that the resting cube stays put before any robot contact.

The host invocation launches one fresh simulator per condition and writes host-private
records under --run-dir. Each simulator:

1. resets the experiment scene and records the cube at every physics step of the native
   reset, tagged by reset phase (reload, robot reset, saved poses, stability check);
2. records the initial head-camera frame the agent would receive;
3. holds both arms at their current joint positions for --hold-steps control steps and
   records the cube after every step.

--variant applies one candidate solver configuration to the cube or the PhysX scene;
"none" runs the adapter unchanged. From dynamics-repair-v4 the adapter itself applies
external forces every TGS iteration, the "external_forces" candidate. No agent, model or
policy runs.
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

AUTHORED_XY = (0.2, -0.13)
VARIANTS = {
    "none": {},
    "cube_iterations": {"cube": {"SolverPositionIterationCount": 64,
                                 "SolverVelocityIterationCount": 4}},
    "scene_iterations": {"physx": {"min_position_iteration_count": 32}},
    "external_forces": {"physx": {"enable_external_forces_every_iteration": True}},
    "stabilization": {"physx": {"enable_stabilization": True}},
}
PHASES = (
    ("scene_manager", "reload_scene"),
    ("robot_manager", "reset"),
    ("scene_manager", "apply_saved_poses"),
    ("layout_manager", "check_layout_stability"),
    ("robot_manager", "set_robot_init_state"),
)


def cube_record(native, phase):
    import numpy as np

    try:
        cube = native.scene_object("red_cube")
        position, orientation = cube.get_local_pose()
        record = {
            "phase": phase,
            "position": np.asarray(position, dtype=float).tolist(),
            "orientation": np.asarray(orientation, dtype=float).tolist(),
            "linear_velocity": np.asarray(cube.get_linear_velocity(), dtype=float).tolist(),
            "angular_velocity": np.asarray(cube.get_angular_velocity(), dtype=float).tolist(),
        }
    except Exception as exc:  # the cube may not exist yet during reload
        return {"phase": phase, "error": f"{type(exc).__name__}: {exc}"}
    record["contacts"] = [
        {"other": pair["other_collider"], "points": len(pair["points"])}
        for pair in native.contact_monitor.pairs.values()
    ]
    return record


def instrument(spec, trace, phase):
    """Wrap the task class so every physics step and reset phase is recorded."""
    from simulators import dynamics_native

    original = dynamics_native.make_task_class

    def make_task_class(settings):
        base = original(settings)

        class Probed(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                owners = {"scene_manager": self.scene_manager,
                          "layout_manager": self.scene_manager.layout_manager,
                          "robot_manager": self.robot_manager}
                for owner_name, method in PHASES:
                    owner = owners[owner_name]
                    wrapped = getattr(owner, method)

                    def tagged(*a, _wrapped=wrapped, _name=f"{owner_name}.{method}", **k):
                        phase["name"] = _name
                        try:
                            return _wrapped(*a, **k)
                        finally:
                            phase["name"] = f"after {_name}"

                    setattr(owner, method, tagged)

            def author_cube_material(self, body_path):
                material = super().author_cube_material(body_path)
                if spec.get("cube"):
                    from pxr import PhysxSchema

                    api = PhysxSchema.PhysxRigidBodyAPI.Apply(self.stage.GetPrimAtPath(body_path))
                    for name, value in spec["cube"].items():
                        getattr(api, f"Create{name}Attr")().Set(value)
                return material

            def sim_step(self, render=True):
                result = super().sim_step(render=render)
                trace.append(cube_record(self, phase["name"]))
                return result

        return Probed

    dynamics_native.make_task_class = make_task_class


def scene_physx(env, spec):
    if not spec.get("physx"):
        return
    configure = env.native.setup_sim_cfg

    def with_variant(config):
        configure(config)
        for name, value in spec["physx"].items():
            setattr(env.native.sim_cfg.sim.physx, name, value)

    env.native.setup_sim_cfg = with_variant


def summarize(trace, initial, hold):
    import numpy as np

    rows = [r for r in trace if "position" in r and abs(r["position"][0]) < 10]
    phases = {}
    for row in rows:
        entry = phases.setdefault(row["phase"], {"steps": 0, "first": row["position"]})
        entry["steps"] += 1
        entry["last"] = row["position"]
        entry["max_angular_speed"] = max(entry.get("max_angular_speed", 0.0),
                                         float(np.linalg.norm(row["angular_velocity"])))
    start = np.asarray(initial["cube_position"][:2])
    positions = np.asarray([s["cube_position"][:2] for s in hold])
    return {
        "physics_steps": len(trace),
        "phases": phases,
        "initial_offset_mm": float(np.linalg.norm(start - AUTHORED_XY) * 1000),
        "initial_solver_angular_speed": float(
            np.linalg.norm(initial["cube_solver_angular_velocity"])
        ),
        "hold_steps": len(hold),
        "hold_drift_mm": float(np.linalg.norm(positions[-1] - start) * 1000) if len(hold) else None,
        "hold_max_step_mm": float(np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) * 1000)
        if len(hold) > 1 else None,
        "hold_max_pose_angular_speed": max(
            (float(np.linalg.norm(s["cube_angular_velocity"])) for s in hold), default=None),
        "hold_robot_contact": any(s["robot_contact"] for s in hold),
    }


def probe(settings, variant, hold_steps, run):
    import numpy as np

    from orchestrator.files import write_json
    from simulators.dynamics_task import build_environment
    from simulators.primitives import rgb_images

    spec = VARIANTS[variant]
    trace, phase = [], {"name": "construction"}
    record = {"settings": {key: settings[key] for key in
                           ("friction_scale", "mass_scale", "restitution")},
              "variant": variant, "spec": spec}
    instrument(spec, trace, phase)
    env = None
    try:
        env = build_environment(settings)
        env.execution_phase = "playground"
        scene_physx(env, spec)
        env.reset(seed=0)
        initial = env.initial_snapshot
        frames = rgb_images(env.public_observation())
        np.savez_compressed(run / "initial_frames.npz", **frames)
        for _ in range(hold_steps):
            state = env.public_observation()["state"]
            env.control_step({
                key: np.asarray(value, dtype=float).reshape(-1)
                for key, value in state.items()
                if key.endswith("arm_joint_state") or key.endswith("ee_joint_state")
            })
        hold = env.native.dynamics_trace
        record.update(
            status="complete",
            physics_scene=env.native.dynamics_snapshot()["physics_scene"],
            summary=summarize(trace, initial["sample"], hold),
            initial_sample=initial["sample"],
            reset_trace=trace,
            hold_trace=[{k: s[k] for k in ("step", "cube_position", "cube_orientation",
                                           "cube_angular_velocity", "cube_solver_angular_velocity",
                                           "robot_contact", "table_contact")} for s in hold],
        )
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(), reset_trace=trace)
        raise
    finally:
        write_json(run / "probe.json", record)
        if env is not None:
            env.close()
    return record


def conditions(friction_scales, mass_scales, restitutions):
    rows = [{"friction_scale": f} for f in friction_scales]
    rows += [{"mass_scale": m} for m in mass_scales]
    rows += [{"restitution": r} for r in restitutions]
    return rows


def launch(run, gpu, rows, variant, hold_steps):
    from orchestrator.dynamics_repair import environment_settings, load_config
    from orchestrator.files import run_lock, write_json
    from orchestrator.resources import gpu_lease

    config = load_config(REPO / "tasks/DynamicsRepair/experiment.example.json")
    interpreter = REPO / ".venv-robodojo/bin/python"
    summary = {"purpose": "resting_cube_settling_check", "model_calls": 0,
               "variant": variant, "spec": VARIANTS[variant], "conditions": {}}
    with run_lock(run), gpu_lease(gpu):
        (run / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        for row in rows:
            settings = {**environment_settings(config)["source"], **row}
            name = "_".join(f"{key.split('_')[0]}{value}" for key, value in row.items())
            directory = run / name
            directory.mkdir()
            with (directory / "stdout.log").open("wb") as out, \
                    (directory / "stderr.log").open("wb") as err:
                code = subprocess.run(
                    [str(interpreter), "-m", "scripts.probe_dynamics_settling", "--inside",
                     "--settings", json.dumps(settings), "--variant", variant,
                     "--hold-steps", str(hold_steps), "--run-dir", str(directory),
                     "--gpu", gpu],
                    cwd=REPO, stdin=subprocess.DEVNULL, stdout=out, stderr=err, check=False,
                ).returncode
            path = directory / "probe.json"
            result = json.loads(path.read_text()) if path.is_file() else None
            summary["conditions"][name] = {
                "returncode": code,
                "status": result["status"] if result else "no_record",
                **({"error": result["error"]} if result and "error" in result else {}),
                **({"summary": result["summary"]} if result and "summary" in result else {}),
            }
            write_json(run / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="none")
    parser.add_argument("--hold-steps", type=int, default=150)
    parser.add_argument("--friction-scales", type=float, nargs="*",
                        default=[1.0, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2])
    parser.add_argument("--mass-scales", type=float, nargs="*", default=[])
    parser.add_argument("--restitutions", type=float, nargs="*", default=[])
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--settings", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.inside:
        probe(json.loads(args.settings), args.variant, args.hold_steps, args.run_dir.resolve())
        return 0
    run = args.run_dir.absolute()
    if not re.match(r"^\d{12}_", run.name):
        raise ValueError("--run-dir must start with YYYYMMDDHHMM_")
    datetime.strptime(run.name[:12], "%Y%m%d%H%M")
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Use a fresh timestamp-prefixed run directory")
    run.mkdir(parents=True, exist_ok=True)
    rows = conditions(args.friction_scales, args.mass_scales, args.restitutions)
    summary = launch(run, args.gpu, rows, args.variant, args.hold_steps)
    print(json.dumps({name: entry.get("summary", entry) for name, entry in
                      summary["conditions"].items()}, indent=1, default=str)[:20000])
    return 0 if all(e["status"] == "complete" for e in summary["conditions"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
