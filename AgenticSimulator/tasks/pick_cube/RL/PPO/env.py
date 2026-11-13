"""Vector Isaac Lab environment. Import/construct after AppLauncher starts."""

import math

import torch

from ..BC.scene import build_scene, sample_objects
from ..BC.schema import INITIAL_PREVIOUS_ACTION
from .rewards import TaskReward, physical_metrics
from .curriculum import STAGES, fractions

PHYSICS_HZ = 200
CONTROL_HZ = 50
DECIMATION = PHYSICS_HZ // CONTROL_HZ


class PickCubeEnv:
    def __init__(self, scene_config, config, reward_config, gamma, *, device, seed,
                 headless, app, episode_id=0, curriculum=None):
        import isaaclab.sim as sim_utils
        from .control import GuidanceController, PoseExecutor

        self.config, self.scene_config = config, scene_config
        self.seed, self.app, self.headless = seed, app, headless
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=1 / PHYSICS_HZ, render_interval=DECIMATION, device=device))
        if "camera" in scene_config:
            self.sim.set_camera_view(eye=scene_config["camera"]["eye"], target=scene_config["camera"]["target"])
        self.scene = build_scene(scene_config, config["num_envs"], config["env_spacing"])
        self.sim.reset()
        self.device, self.num_envs = self.sim.device, config["num_envs"]
        self.executor = PoseExecutor(self.scene["robot"], self.scene["cube"], self.scene["plate"], self.device)
        self.curriculum = curriculum
        self.expert_only = False  # enabled only by the explicit diagnostic entry
        self.online_supervision = False
        self.feedback_only = False  # explicit expert diagnostic, never policy inference
        self.curriculum_level = curriculum.get("start_level", 0) if curriculum else len(STAGES) - 1
        self.curriculum_progress_index = curriculum.get("start_progress_index", 0) if curriculum else 0
        self.teacher = (GuidanceController(self.scene["robot"], self.scene["cube"], self.scene["plate"], self.device)
                        if curriculum else None)
        self.monitor = TaskReward(self.num_envs, self.device, reward_config, gamma)
        self.all_ids = torch.arange(self.num_envs, device=self.device)
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_ids = torch.zeros_like(self.steps)
        self.levels = torch.zeros_like(self.steps)
        self.progress_indices = torch.zeros_like(self.steps)
        self.handoff_fractions = torch.zeros(self.num_envs, device=self.device)
        self.learner_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.paused = self.learner_started.clone()
        self.learner_steps = torch.zeros_like(self.steps)
        self.next_episode_id = episode_id
        self.max_steps = math.ceil(config["max_seconds"] * CONTROL_HZ)
        self.warmup_steps = round(config["warmup_seconds"] * CONTROL_HZ)
        self.initial_cube_z = torch.zeros(self.num_envs, device=self.device)
        self.episode_return = torch.zeros_like(self.initial_cube_z)
        self.reset(self.all_ids)

    @property
    def active(self):
        return (self.steps >= 0) & ~self.paused

    def observation(self):
        return self.executor.observation()

    def reset(self, env_ids):
        """Reset only selected worlds; no simulation steps affecting other worlds."""
        if not len(env_ids):
            return self.observation()
        scene, robot = self.scene, self.scene["robot"]
        count = len(env_ids)
        ids = list(range(self.next_episode_id, self.next_episode_id + count))
        if self.config["fixed_scene"]:
            ids = [self.config["fixed_episode_id"]] * count
        self.next_episode_id += count
        self.episode_ids[env_ids] = torch.tensor(ids, device=self.device)
        samples = sample_objects(self.scene_config, ids, self.seed)
        root = robot.data.default_root_state[env_ids].clone()
        root[:, :3] += scene.env_origins[env_ids]
        robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
        pos = robot.data.default_joint_pos[env_ids].clone()
        vel = robot.data.default_joint_vel[env_ids].clone()
        robot.write_joint_state_to_sim(pos, vel, env_ids=env_ids)
        robot.set_joint_position_target(pos, env_ids=env_ids)
        robot.set_joint_velocity_target(torch.zeros_like(vel), env_ids=env_ids)
        robot.set_joint_effort_target(torch.zeros_like(pos), env_ids=env_ids)
        for name, poses in samples.items():
            obj = scene[name]
            pose = torch.as_tensor(poses, device=self.device)
            pose[:, :3] += scene.env_origins[env_ids]
            obj.write_root_pose_to_sim(pose, env_ids=env_ids)
            obj.write_root_velocity_to_sim(torch.zeros(count, 6, device=self.device), env_ids=env_ids)
        scene.reset(env_ids)
        self.executor.reset(env_ids)
        # write_joint_state invalidates body caches; update without advancing time.
        scene.update(0.0)
        self.steps[env_ids] = -self.warmup_steps
        self.paused[env_ids] = False
        self.learner_steps[env_ids] = 0
        self.levels[env_ids] = self.curriculum_level
        self.progress_indices[env_ids] = self.curriculum_progress_index
        if self.curriculum and 0 < self.curriculum_level < len(STAGES) - 1:
            # Retain already learned suffixes while extending towards the start.
            retain = torch.rand(count, device=self.device) < self.curriculum.get("retention_fraction", 0.0)
            if self.curriculum_progress_index > 0:
                self.progress_indices[env_ids[retain]] -= 1
            else:
                previous_level = self.curriculum_level - 1
                self.levels[env_ids[retain]] = previous_level
                self.progress_indices[env_ids[retain]] = len(fractions(self.curriculum, previous_level)) - 1
        for level in range(len(STAGES)):
            selected = env_ids[self.levels[env_ids] == level]
            self.handoff_fractions[selected] = self.handoff_fractions.new_tensor(
                fractions(self.curriculum or {}, level))[self.progress_indices[selected]]
        self.learner_started[env_ids] = self.levels[env_ids] == len(STAGES) - 1
        if self.teacher is not None:
            self.teacher.reset_memory(env_ids)
        self.episode_return[env_ids] = 0
        obs = self.observation()
        self.initial_cube_z[env_ids] = obs[env_ids, 25]
        self.monitor.reset(env_ids, physical_metrics(obs, self.initial_cube_z))
        return obs

    def step(self, actions):
        """Return FINAL observations; caller resets done environments afterwards.

        Each world settles independently. Warmup actions are ignored, and valid
        masks exclude these transitions from PPO. No hidden memory crosses warmup.
        """
        if not self.app.is_running():
            raise KeyboardInterrupt("Simulator window closed")
        active = self.active.clone()
        previous_lift = self.monitor.seen_lift.clone()
        actions = actions.clone()
        tracking = active.clone()
        guide_failed = torch.zeros_like(active)
        expert_actions = torch.zeros_like(actions)
        expert_valid = torch.zeros_like(active)
        if self.teacher is not None:
            guiding = active & ~self.learner_started
            beginning = (guiding & (self.steps == 0)).nonzero().flatten()
            if len(beginning):
                self.teacher.reset_memory(beginning)
            # Freeze all non-guided FSMs, without touching the shared robot.
            supervising = guiding
            self.teacher.state[~supervising] = self.teacher.DONE
            if supervising.any():
                expert_actions = self.teacher.decide(1 / CONTROL_HZ)
                expert_valid = supervising & ~self.teacher.terminal
                target_phase = self.levels.new_tensor([s[1] for s in STAGES])[self.levels]
                guide_failed = guiding & self.teacher.terminal
                # Move the handoff gradually backwards within each motion, so
                # "above plate" does not jump directly to a 30 cm carry.
                rows = self.all_ids
                target = self.teacher.goals[rows, target_phase]
                origin = self.teacher.goals[rows, (target_phase - 1).clamp_min(0)]
                target_hand, _, _ = self.teacher._target_pose()
                actual_hand = self.scene["robot"].data.body_state_w[:, self.teacher.hand_id, :3]
                distance = (target_hand - actual_hand).norm(dim=-1)
                span = (target - origin).norm(dim=-1).clamp_min(1e-6)
                progress = (1 - distance / span).clamp(0, 1)
                progress = torch.where(target_phase == self.teacher.CLOSE,
                                       self.teacher.elapsed / 1.5, progress)
                arrived = ((self.teacher.state > target_phase)
                           | ((self.teacher.state == target_phase)
                              & (progress >= self.handoff_fractions)))
                handoff = guiding & ~guide_failed & arrived & (not self.expert_only)
                self.learner_started |= handoff
                guiding &= ~handoff
                actions[guiding] = expert_actions[guiding]
                tracking[guiding] = self.teacher._tracking[guiding]
        valid = active & self.learner_started
        if self.online_supervision:
            labels = self.teacher.feedback_labels(self.observation(), self.monitor.seen_lift,
                                                  self.initial_cube_z)
            replace = valid
            expert_actions[replace] = labels[replace]
            expert_valid[replace] = True
            if self.feedback_only:
                actions[replace] = labels[replace]
                valid = torch.zeros_like(valid)
        actions[~active] = actions.new_tensor(INITIAL_PREVIOUS_ACTION)
        self.executor.set_action(actions, tracking=tracking)
        for substep in range(DECIMATION):
            self.executor.physics_step(1 / PHYSICS_HZ)
            self.scene.write_data_to_sim()
            self.sim.step(render=not self.headless and substep == 0)
            self.scene.update(1 / PHYSICS_HZ)
            obs = self.observation()
            metrics = physical_metrics(obs, self.initial_cube_z)
            self.monitor.observe(metrics, actions[:, 6].bool(), active, 1 / PHYSICS_HZ)
        self.steps += 1
        self.learner_steps += valid.long()
        terminated = active & (self.monitor.success | self.executor.terminal | guide_failed)
        truncated = active & (self.steps >= self.max_steps) & ~terminated
        reward = self.monitor.reward(metrics, active, terminated, previous_lift, 1 / CONTROL_HZ)
        self.episode_return += reward
        settled = (~active & self.active).nonzero().flatten()
        if len(settled):
            self.initial_cube_z[settled] = obs[settled, 25]
            self.monitor.reset(settled, physical_metrics(obs, self.initial_cube_z))
        info = dict(valid=valid, metrics=metrics, success=self.monitor.success.clone(),
                    seen_lift=self.monitor.seen_lift.clone(),
                    episode_return=self.episode_return.clone(), episode_ids=self.episode_ids.clone(),
                    steps=self.steps.clone(), invalid_ik=self.executor.terminal.clone(),
                    curriculum_levels=self.levels.clone(), learner_steps=self.learner_steps.clone(),
                    curriculum_progress_indices=self.progress_indices.clone(),
                    handoff_fractions=self.handoff_fractions.clone(),
                    guide_failed=guide_failed, executed_actions=actions,
                    expert_actions=expert_actions, expert_valid=expert_valid)
        return obs, reward, terminated, truncated, info
