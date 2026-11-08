"""Custom fixed-scene dynamics experiment, separate from native RoboDojo tasks."""

from __future__ import annotations

import copy
import json
import math
from unittest.mock import patch

from integrations.friction_rl.control import SCENE_PATH
from simulators.dynamics_native import cube_material_values
from simulators.robodojo import RoboDojoRuntime

TASK_NAME = "red_cube_to_plate"


def experiment_settings(settings):
    values = {
        "task_name": TASK_NAME,
        "seed": 0,
        "action_limit": 500,
        "friction_scale": 1.0,
        "mass_scale": 1.0,
        "restitution": 0.0,
        **settings,
    }
    if values["task_name"] != TASK_NAME or values["seed"] != 0 or values["action_limit"] != 500:
        raise ValueError("DynamicsAdapt uses red_cube_to_plate, seed=0, action_limit=500")
    for key in ("friction_scale", "mass_scale", "restitution"):
        values[key] = float(values[key])
        if not math.isfinite(values[key]):
            raise ValueError(f"{key} must be finite")
    if (
        values["friction_scale"] < 0
        or values["mass_scale"] <= 0
        or not 0 <= values["restitution"] <= 1
    ):
        raise ValueError("Require nonnegative friction, positive mass, restitution in [0, 1]")
    return values


def scene_layout(settings):
    """A new value copy for every episode; all perturbations are host-selected."""
    layout = (
        copy.deepcopy(settings["scene"])
        if "scene" in settings
        else json.loads(SCENE_PATH.read_text())
    )
    physics = layout["Rigid"]["cube"][0]["physics"]
    physics.update(mass=0.35 * settings["mass_scale"], **cube_material_values(settings))
    return layout


class DynamicsTask(RoboDojoRuntime):
    def __init__(self, settings):
        self.settings = experiment_settings(settings)
        self.initial_snapshot = None
        super().__init__(TASK_NAME, seed=0)

    def _scene_identity(self, task_name, seed):
        return 0, 0, SCENE_PATH

    def _load_task_config(self):
        return {}

    def _create_eval_env(self, eval_module, cfg):
        # Isaac imports must occur after the parent has launched SimulationApp.
        from env.seed_manager.seed_manager import SeedManager
        from task.RoboDojo import task_registry

        from simulators.dynamics_native import make_task_class

        layout = scene_layout(self.settings)

        class FixedSeedManager(SeedManager):
            def init_eval(self, **kwargs):
                self.seed_list = [0]
                self.seed_info = {0: {"scene_layout": str(SCENE_PATH)}}
                self.st_idx, self.ed_idx, self.idx = 0, 1, 0
                self.type = "eval"
                self._current_batch_seeds = None

            def get_seed_scene_info(self, seed):
                if seed != 0:
                    raise ValueError("DynamicsAdapt owns one fixed scene")
                return copy.deepcopy(layout)

        task_class = make_task_class(self.settings)
        with (
            patch.object(task_registry, "load_task_class", return_value=(TASK_NAME, task_class)),
            patch.object(eval_module, "SeedManager", FixedSeedManager),
        ):
            return super()._create_eval_env(eval_module, cfg)

    def _create_native(self):
        super()._create_native()
        configure = self.native.setup_sim_cfg

        def configure_dynamics(config):
            configure(config)
            # dynamics-repair-v4. With gravity applied once per TGS step, the resting cube
            # falls into a two-step bounce: 10 um in height, up to 5 rad/s about a
            # horizontal axis. At lowered friction each bounce walks it, 20 mm at 0.3,
            # before anything touches it. Applying external forces in every position
            # iteration leaves it still at every friction scale.
            self.native.sim_cfg.sim.physx.enable_external_forces_every_iteration = True

        self.native.setup_sim_cfg = configure_dynamics

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self.initial_snapshot = self.native.dynamics_snapshot()
        if not self.initial_snapshot["physics_scene"]["enable_external_forces_every_iteration"]:
            raise RuntimeError("The PhysX scene does not apply external forces every iteration")
        return result

    def host_snapshot(self):
        if self.initial_snapshot is None:
            raise RuntimeError("Reset the simulator before recording its initial state")
        return {
            "task_name": TASK_NAME,
            "settings": dict(self.settings),
            "initial_state": copy.deepcopy(self.initial_snapshot),
            "current_state": self.native.dynamics_snapshot(),
            "trajectory": copy.deepcopy(self.native.dynamics_trace),
        }
