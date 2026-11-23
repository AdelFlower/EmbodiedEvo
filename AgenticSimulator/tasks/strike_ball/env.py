"""Isaac Lab manages the RL loop; this environment tracks each shot at physics rate."""

import weakref

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

from .mdp import predict_launch_aim_error, predict_launch_offset, reference_velocity
from .scene import (
    BOARD_THICKNESS, PADDLE_OFFSET, check_passage, make_board_contact_view, make_contact_view,
    paddle_contacts,
)


class StrikeBallEnv(ManagerBasedRLEnv):
    action_repeat = 1
    record_contact_geometry = False

    def load_managers(self):
        # The observation manager evaluates terms once here to infer their shapes.
        # Isaac Lab 2 creates its episode_length_buf after that first evaluation.
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.robot, self.ball = self.scene["robot"], self.scene["ball"]
        self.wrist = self.robot.body_names.index("wrist_3_link")
        root_pattern = self.scene.env_regex_ns.replace(".*", "*")
        self.physics_view, self.contacts = make_contact_view(self.robot, root_pattern)
        self.board_contacts = make_board_contact_view(self.physics_view, root_pattern, self.num_envs)
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
        self.best_distance = origins.new_zeros(self.num_envs)
        self.prehit_action_count = torch.zeros_like(self.episode_length_buf)
        self.prehit_saturated_count = torch.zeros_like(self.episode_length_buf)
        self.previous_best_distance = self.best_distance.clone()
        self.crossing_error = torch.full_like(self.best_distance, float("inf"))
        self.first_hit_this_step = torch.zeros_like(self.hit)
        self.contact_substep = torch.zeros_like(self.episode_length_buf)
        self.contact_plane_distance = torch.zeros_like(self.best_distance)
        self.first_pass_this_step = torch.zeros_like(self.hit)
        self.launch_aim_error_this_step = self.crossing_error.clone()
        self.launch_speed_error_this_step = self.crossing_error.clone()
        self.launch_entrance_error_this_step = self.crossing_error.clone()
        self.launch_entrance_offset_this_step = torch.full_like(origins, float("inf"))
        self.launch_state_this_step = origins.new_zeros((self.num_envs, 6))
        self.launch_descent_slope_this_step = self.crossing_error.clone()
        self.touching_board = torch.zeros_like(self.hit)
        self.board_impacts_this_step = torch.zeros_like(self.best_distance)
        super().load_managers()
        # The ball is updated once after every PhysX step by InteractiveScene.
        env = weakref.proxy(self)
        self.ball.after_physics_step = lambda: env._observe_physics()

    def paddle_center(self):
        quat = self.robot.data.body_link_quat_w[:, self.wrist]
        return self.robot.data.body_link_pos_w[:, self.wrist] + quat_apply(quat, self.paddle_offset)

    def step(self, action):
        if self.action_repeat == 1:
            return self._step_native(action)
        # Only the deadline ends a shot. Starting at reset and using a divisor
        # of the native episode length keeps the last repeat on the deadline.
        obs, reward, terminated, truncated, extras = self._step_native(action)
        total = reward.clone()
        for _ in range(1, self.action_repeat):
            obs, reward, terminated, truncated, extras = self._step_native(action)
            total += reward
        return obs, total, terminated, truncated, extras

    def _step_native(self, action):
        self.extras.pop("shot", None)
        self.extras.pop("log", None)
        distance = (self.paddle_center() - self.held_state[:, :3]).norm(dim=-1)
        self.best_distance = torch.where(self.episode_length_buf == 0, distance, self.best_distance)
        self.previous_best_distance.copy_(self.best_distance)
        self.first_hit_this_step.zero_()
        self.first_pass_this_step.zero_()
        self.launch_aim_error_this_step.fill_(float("inf"))
        self.launch_speed_error_this_step.fill_(float("inf"))
        self.launch_entrance_error_this_step.fill_(float("inf"))
        self.launch_entrance_offset_this_step.fill_(float("inf"))
        self.launch_descent_slope_this_step.fill_(float("inf"))
        self.board_impacts_this_step.zero_()
        # The RSL wrapper already clips actions, so count components at either bound.
        before_hit = ~self.hit
        self.prehit_action_count += before_hit * action.shape[-1]
        self.prehit_saturated_count += before_hit * (action.abs() >= 10.).sum(dim=-1)
        self._apply_support()
        return super().step(action.clamp(-10., 10.))

    def _apply_support(self):
        self.ball.set_external_force_and_torque(self.support * (~self.hit)[:, None, None], self.zero_torque)

    def _observe_physics(self):
        touching = paddle_contacts(self.contacts, self.robot, self.wrist)
        first_hit = touching & ~self.hit
        separation = self.paddle_center() - self.held_state[:, :3]
        distance = separation.norm(dim=-1)
        self.best_distance = torch.where(self.hit, self.best_distance, torch.minimum(self.best_distance, distance))
        self.first_hit_this_step |= first_hit
        decimation = self.cfg.decimation
        elapsed_substeps = (self._sim_step_counter - 1) % decimation + 1
        self.contact_substep = torch.where(
            first_hit, self.episode_length_buf * decimation + elapsed_substeps, self.contact_substep,
        )
        if self.record_contact_geometry:
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
            self.launch_aim_error_this_step[release_ids] = predict_launch_aim_error(
                pos[release_ids] - self.hole[release_ids], self.ball.data.root_lin_vel_w[release_ids],
                self.gravity, self.physics_dt, remaining,
            )
            wanted = reference_velocity(self, pos[release_ids], release_ids)
            self.launch_speed_error_this_step[release_ids] = (
                self.ball.data.root_lin_vel_w[release_ids] - wanted
            ).norm(dim=1)
            velocity = self.ball.data.root_lin_vel_w[release_ids]
            from_entrance = pos[release_ids] - self.hole_top[release_ids]
            self.launch_state_this_step[release_ids] = torch.cat((from_entrance, velocity), dim=1)
            entrance_offset = predict_launch_offset(
                from_entrance, velocity, self.gravity, self.physics_dt, remaining,
            )
            self.launch_entrance_offset_this_step[release_ids] = entrance_offset
            self.launch_entrance_error_this_step[release_ids] = entrance_offset.norm(dim=1)
            vertical_speed = velocity[:, 2] + 0.5 * self.gravity[2] * self.physics_dt
            falling_speed = (vertical_speed.square() - 2 * self.gravity[2] * from_entrance[:, 2]).clamp_min(0.).sqrt()
            self.launch_descent_slope_this_step[release_ids] = velocity[:, :2].norm(dim=1) / falling_speed.clamp_min(0.4)
        self.awaiting_separation &= ~separated

        # Merge simultaneous contacts on adjacent wedges before counting a new impact.
        board_impulses = self.board_contacts.get_contact_force_matrix(dt=1.)
        touching_board = (board_impulses != 0).any(dim=(1, 2))
        self.board_impacts_this_step += self.hit & ~self.passed & touching_board & ~self.touching_board
        self.touching_board.copy_(touching_board)

        down, error, pending, cleared = check_passage(self.previous_pos, pos, self.hole, self.in_hole)
        first_crossing = self.hit & down & self.crossing_error.isinf()
        self.crossing_error = torch.where(first_crossing, error, self.crossing_error)
        self.in_hole = self.hit & pending
        self.first_pass_this_step |= self.hit & cleared & ~self.passed
        self.passed |= self.hit & cleared
        self.previous_pos.copy_(pos)

    def reset_shot(self, env_ids):
        # The reset event runs after the native robot and ball reset terms.
        state = self.ball.data.root_state_w[env_ids]
        self.held_state[env_ids] = state
        self.previous_pos[env_ids] = state[:, :3]
        self.prehit_action_count[env_ids] = 0
        self.prehit_saturated_count[env_ids] = 0
        self.robot.set_joint_position_target(self.robot.data.joint_pos[env_ids], env_ids=env_ids)
        self.hit[env_ids] = False
        self.contact_substep[env_ids] = 0
        self.contact_plane_distance[env_ids] = 0.
        self.passed[env_ids] = False
        self.in_hole[env_ids] = False
        self.awaiting_separation[env_ids] = False
        self.touching_board[env_ids] = False
        self.board_impacts_this_step[env_ids] = 0
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
            measured = errors[errors.isfinite()]
            saturation = self.prehit_saturated_count[finished].sum() / self.prehit_action_count[finished].sum().clamp_min(1)
        super()._reset_idx(env_ids)
        if len(finished):
            self.extras["log"].update({
                "Strike/hit_rate": hit.float().mean(),
                "Strike/pass_rate": passed.float().mean(),
                "Strike/pass_given_hit": passed.sum() / hit.sum().clamp_min(1),
                "Strike/prehit_action_saturation": saturation,
            })
            if len(measured):
                self.extras["log"]["Strike/median_crossing_error_mm"] = 1000. * torch.quantile(measured, 0.5)
