"""Isaac Lab manages the RL loop; this environment tracks each shot at physics rate."""

import weakref

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

from .mdp import predict_launch_offset
from .scene import (
    BOARD_THICKNESS, PADDLE_OFFSET, check_passage, make_contact_view,
    paddle_contacts,
)


class StrikeBallEnv(ManagerBasedRLEnv):
    record_contact_geometry = False

    def load_managers(self):
        # The observation manager evaluates terms once here to infer their shapes.
        # Isaac Lab 2 creates its episode_length_buf after that first evaluation.
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.robot, self.ball = self.scene["robot"], self.scene["ball"]
        self.wrist = self.robot.body_names.index("wrist_3_link")
        root_pattern = self.scene.env_regex_ns.replace(".*", "*")
        self.physics_view, self.contacts = make_contact_view(self.robot, root_pattern)
        origins = self.scene.env_origins
        self.hole = origins + origins.new_tensor(self.cfg.scene.board.init_state.pos)
        self.hole_top = self.hole + origins.new_tensor((0., 0., BOARD_THICKNESS / 2))
        self.paddle_offset = origins.new_tensor(PADDLE_OFFSET).expand_as(origins)
        self.local_normal = origins.new_tensor([0., 0., 1.]).expand_as(origins)
        self.held_state = self.ball.data.default_root_state.clone()
        masses = self.ball.root_physx_view.get_masses().to(self.device).reshape(-1, 1, 1)
        self.gravity = origins.new_tensor(self.cfg.sim.gravity)
        self.support = -masses * self.gravity
        self.zero_torque = torch.zeros_like(self.support)
        self.previous_pos = origins.clone()
        self.hit = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.awaiting_separation = torch.zeros_like(self.hit)
        self.in_hole = torch.zeros_like(self.hit)
        self.passed = torch.zeros_like(self.hit)
        self.crossing_error = origins.new_full((self.num_envs,), float("inf"))
        self.contact_substep = torch.zeros_like(self.episode_length_buf)
        self.contact_plane_distance = origins.new_zeros(self.num_envs)
        self.launch_entrance_offset_this_step = torch.full_like(origins, float("inf"))
        self.launch_descent_slope_this_step = self.crossing_error.clone()
        super().load_managers()
        # The ball is updated once after every PhysX step by InteractiveScene.
        env = weakref.proxy(self)
        self.ball.after_physics_step = lambda: env._observe_physics()

    def paddle_center(self):
        quat = self.robot.data.body_link_quat_w[:, self.wrist]
        return self.robot.data.body_link_pos_w[:, self.wrist] + quat_apply(quat, self.paddle_offset)

    def step(self, action):
        self.extras.pop("shot", None)
        self.extras.pop("log", None)
        self.launch_entrance_offset_this_step.fill_(float("inf"))
        self.launch_descent_slope_this_step.fill_(float("inf"))
        self._apply_support()
        return super().step(action.clamp(-10., 10.))

    def _apply_support(self):
        self.ball.set_external_force_and_torque(self.support * (~self.hit)[:, None, None], self.zero_torque)

    def _observe_physics(self):
        touching = paddle_contacts(self.contacts, self.robot, self.wrist)
        first_hit = touching & ~self.hit
        decimation = self.cfg.decimation
        elapsed_substeps = (self._sim_step_counter - 1) % decimation + 1
        self.contact_substep = torch.where(
            first_hit, self.episode_length_buf * decimation + elapsed_substeps, self.contact_substep,
        )
        if self.record_contact_geometry:
            separation = self.paddle_center() - self.held_state[:, :3]
            normal = quat_apply(self.robot.data.body_link_quat_w[:, self.wrist], self.local_normal)
            plane_distance = (separation * normal).sum(-1).abs()
            self.contact_plane_distance = torch.where(first_hit, plane_distance, self.contact_plane_distance)
        self.hit |= first_hit
        self._apply_support()

        # A contact elsewhere on the arm must not displace a held ball.
        state = self.ball.data.root_state_w
        ids = ((~self.hit) & (state != self.held_state).any(dim=1)).nonzero().flatten()
        if len(ids):
            self.ball.write_root_state_to_sim(self.held_state[ids], env_ids=ids)

        self.awaiting_separation |= first_hit
        separated = self.awaiting_separation & ~touching
        pos = self.ball.data.root_pos_w
        release_ids = separated.nonzero().flatten()
        if len(release_ids):
            # episode_length_buf still counts only fully completed control steps here.
            remaining = (self.max_episode_length - self.episode_length_buf[release_ids]) * decimation - elapsed_substeps
            velocity = self.ball.data.root_lin_vel_w[release_ids]
            from_entrance = pos[release_ids] - self.hole_top[release_ids]
            entrance_offset = predict_launch_offset(
                from_entrance, velocity, self.gravity, self.physics_dt, remaining,
            )
            self.launch_entrance_offset_this_step[release_ids] = entrance_offset
            vertical_speed = velocity[:, 2] + 0.5 * self.gravity[2] * self.physics_dt
            falling_speed = (vertical_speed.square() - 2 * self.gravity[2] * from_entrance[:, 2]).clamp_min(0.).sqrt()
            self.launch_descent_slope_this_step[release_ids] = velocity[:, :2].norm(dim=1) / falling_speed.clamp_min(0.4)
        self.awaiting_separation &= ~separated

        down, error, pending, cleared = check_passage(self.previous_pos, pos, self.hole, self.in_hole)
        first_crossing = self.hit & down & self.crossing_error.isinf()
        self.crossing_error = torch.where(first_crossing, error, self.crossing_error)
        self.in_hole = self.hit & pending
        self.passed |= self.hit & cleared
        self.previous_pos.copy_(pos)

    def reset_shot(self, env_ids):
        # The reset event runs after the native robot and ball reset terms.
        state = self.ball.data.root_state_w[env_ids]
        self.held_state[env_ids] = state
        self.previous_pos[env_ids] = state[:, :3]
        self.robot.set_joint_position_target(self.robot.data.joint_pos[env_ids], env_ids=env_ids)
        self.hit[env_ids] = False
        self.contact_substep[env_ids] = 0
        self.contact_plane_distance[env_ids] = 0.
        self.passed[env_ids] = False
        self.in_hole[env_ids] = False
        self.awaiting_separation[env_ids] = False
        self.crossing_error[env_ids] = float("inf")

    def _reset_idx(self, env_ids):
        finished = env_ids[self.episode_length_buf[env_ids] >= self.max_episode_length]
        if len(finished):
            hit, passed = self.hit[finished], self.passed[finished]
            errors = self.crossing_error[finished]
            self.extras["shot"] = {
                "hit": hit, "passed": passed, "crossing_error": errors,
                "contact_substep": self.contact_substep[finished],
                "contact_plane_distance": self.contact_plane_distance[finished],
            }
        super()._reset_idx(env_ids)
