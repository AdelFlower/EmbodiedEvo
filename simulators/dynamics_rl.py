"""Host-only state observations and episode resets for the independent RL experiment.

The coding-agent primitives and the original DynamicsRepair task remain unchanged.
Isaac imports are deliberately deferred to DynamicsTask's existing launch path.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from simulators.dynamics_task import DynamicsTask, experiment_settings, scene_layout
from simulators.primitives import rgb_images
from simulators.robodojo import robodojo_cwd


def json_value(value):
    """Convert native arrays/tensors without importing Torch before Isaac starts."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if callable(getattr(value, "detach", None)) and callable(getattr(value, "cpu", None)):
        return json_value(value.detach().cpu().tolist())
    return value


def install_state_observations(manager):
    """Skip RGB acquisition while preserving native robot observations and physics."""
    original = manager.get_obs

    def get_state_observation(env_idx_list=None):
        camera, capture = manager.camera_manager, manager.capture_manager
        manager.camera_manager = manager.capture_manager = None
        try:
            return original(env_idx_list=env_idx_list)
        finally:
            manager.camera_manager, manager.capture_manager = camera, capture

    manager.get_obs = get_state_observation
    manager.render_for_capture = lambda: None


def reset_contact_records(monitor):
    """Retained rigid bodies must not inherit an earlier reset's contact evidence."""
    monitor.pairs.clear()
    monitor.material_paths.clear()
    monitor.cube_material_paths.clear()
    monitor.error = None
    monitor.report_count = 0


def source_episode_scene(scene, *, parameters):
    """Validate XY translations while preserving the host's fixed dynamics."""
    nominal = scene_layout(parameters)
    candidate = scene_layout({"scene": scene, **parameters})
    comparison = copy.deepcopy(candidate)
    for category in ("cube", "plate"):
        position = np.asarray(candidate["Rigid"][category][0]["default_pos"], dtype=float)
        original = np.asarray(nominal["Rigid"][category][0]["default_pos"], dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("RL layout positions must be finite three-vectors")
        if np.any(np.abs(position[:2] - original[:2]) > 0.03 + 1e-9):
            raise ValueError("RL layout XY translations must stay within +/- 3 cm")
        comparison["Rigid"][category][0]["default_pos"][:2] = original[:2].tolist()
    if comparison != nominal:
        raise ValueError("RL layouts may change only the cube and plate XY positions")
    return candidate


def install_kinematic_plate_reset(manager):
    """Keep native pose resets, without invalid velocity writes to the fixed plate.

    DynamicsTask's object factory makes the plate kinematic before returning it.
    Native RigidObject.apply_saved_pose()/relocate_offscreen() subsequently call
    _apply_default_velocities(); its explicit None guards are the supported way
    to omit these writes. Repeated invalid writes otherwise accumulate until
    PhysX stops the simulation after 1,000 errors. Dynamic cube resets retain
    their original zero-velocity defaults.
    """
    instantiate = manager.instantiate_object_wrapper

    def create_object(*args, **kwargs):
        obj = instantiate(*args, **kwargs)
        if obj.instance_config.get("label") == "plate":
            obj._default_linear_velocity = None
            obj._default_angular_velocity = None
        return obj

    manager.instantiate_object_wrapper = create_object


class DynamicsRLTask(DynamicsTask):
    """Source training or frozen-policy transfer; the host fixes the dynamics."""

    reuse_reset_rigid_bodies = True

    def __init__(self, *, mode="state", settings=None):
        if mode not in {"state", "render"}:
            raise ValueError("RL mode must be state or render")
        settings = experiment_settings(settings or {})
        self.mode = mode
        self.episode_scene = scene_layout(settings)
        self.episode_dir = None
        self.episode_started = False
        self.video_writers = {}
        self.video_paths = {}
        self.episode_steps = 0
        super().__init__(settings)

    def _create_eval_env(self, eval_module, cfg):
        native = super()._create_eval_env(eval_module, cfg)

        def current_layout(seed):
            if seed != 0:
                raise ValueError("RL layouts use native seed zero and a separate layout RNG")
            return copy.deepcopy(self.episode_scene)

        native.seed_manager.get_seed_scene_info = current_layout
        return native

    def _create_native(self):
        super()._create_native()
        # Install before the first reset constructs any scene object. This is
        # deliberately scoped to this RL adapter, not the vendor or base task.
        install_kinematic_plate_reset(self.native.scene_manager)
        if self.mode == "state":
            install_state_observations(self.native.obs_manager)
        # Stream each public observation exactly once into the durable episode
        # directory. Native temporary writers delete partial recordings on reset.
        self.native._stream_vision = lambda env_idx, frame: None

    def _update_retained_poses(self):
        records = {
            record["label"]: record
            for objects in self.episode_scene["Rigid"].values()
            for record in objects
        }
        for objects in self.native.scene_manager._rigid_and_dynamic_objects:
            for obj in objects.values():
                label = obj.instance_config.get("label")
                if label not in records:
                    continue
                record = records[label]
                obj.default_pos = tuple(record["default_pos"])
                obj.default_ori = tuple(record["default_ori"])
                # apply_saved_pose clears velocities again after native reload;
                # clear them now as reload itself advances physics as well.
                obj.apply_saved_pose()

    def reset_episode(self, scene, episode_dir):
        self.close_videos()
        self.episode_scene = source_episode_scene(
            scene, parameters={key: self.settings[key] for key in
                               ("friction_scale", "mass_scale", "restitution")},
        )
        self.settings["scene"] = copy.deepcopy(self.episode_scene)
        self.episode_dir = Path(episode_dir)
        self.video_paths = {}
        self._frames = {}
        self._restoring_scene = self.episode_started
        try:
            with robodojo_cwd():
                if self.episode_started:
                    self._update_retained_poses()
                reset_contact_records(self.native.contact_monitor)
                result = super().reset(seed=0)
                if abs(float(self.native.cube.get_mass()) - 0.35 * self.settings["mass_scale"]) > 1e-4:
                    raise RuntimeError("Live cube mass differs from the configured mass")
            self.episode_started = True
            self.episode_steps = int(self.native.take_action_cnt[0])
            self.left_pose = np.array(self._observation["state"]["left_ee_pose"], copy=True)
            self.left_grip = np.array(
                self._observation["state"]["left_ee_joint_state"], copy=True
            )
            return result
        finally:
            self._restoring_scene = False

    def control_step(self, action):
        result = super().control_step(action)
        self.episode_steps = int(self.native.take_action_cnt[0])
        return result

    def observe(self, obs):
        # Keep only the current observation. CodeTask's full-episode frame lists
        # are appropriate for code submissions but unbounded during RL training.
        self._observation = copy.deepcopy(obs)
        self._frames = {}
        if self.mode != "render" or self.episode_dir is None:
            return
        for camera, frame in rgb_images(obs).items():
            if camera not in self.video_writers:
                import imageio.v2 as imageio

                path = self.episode_dir / f"raw_{camera}.mp4"
                self.video_writers[camera] = imageio.get_writer(
                    path, fps=25, codec="libx264", quality=8,
                    macro_block_size=1, ffmpeg_params=["-threads", "1"],
                )
                self.video_paths[camera] = str(path)
            self.video_writers[camera].append_data(np.ascontiguousarray(frame))

    def close_videos(self):
        # A recording error is not silently ignored or relabeled a task failure.
        writers, self.video_writers = self.video_writers, {}
        errors = []
        for writer in writers.values():
            try:
                writer.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("Could not finish raw RL video") from errors[0]

    def rl_state(self):
        with robodojo_cwd():
            sample = self.native.dynamics_sample()
            sample.pop("contacts", None)
            for key in ("finger_contacts", "gripper_opening", "gripper_position"):
                sample[key] = {
                    arm.removesuffix("_arm_joint_state"): value
                    for arm, value in sample[key].items()
                }
            plate_position, plate_orientation = self.native.plate.get_local_pose()
            velocities = None
            manager = self.native.robot_manager
            for robot, articulation in zip(manager.robot_list, manager.robot_key, strict=True):
                if robot.arm_name == "right_arm":
                    velocities = articulation.data.joint_vel[0, robot.arm_joint_indices]
                    velocities = velocities.detach().cpu().numpy()
                    break
            if velocities is None:
                raise RuntimeError("The RL task requires the native right arm")
            return json_value({
                "public_state": self._observation["state"],
                "sample": sample,
                "plate_position": plate_position,
                "plate_orientation": plate_orientation,
                "right_arm_joint_velocity": velocities,
                "scoring": self.native.placement_scorer.snapshot(),
            })

    def close(self):
        try:
            self.close_videos()
        finally:
            super().close()
