"""Isaac Lab manages the RL loop; this environment tracks each shot at physics rate."""

import weakref

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

from .scene import (
    BOARD_THICKNESS, PADDLE_OFFSET, check_passage, make_board_contact_view, make_contact_view,
    paddle_contacts, target_velocity,
)


class StrikeBallEnv(ManagerBasedRLEnv):
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
        self.support = -masses * origins.new_tensor(self.cfg.sim.gravity)
        self.zero_torque = torch.zeros_like(self.support)
        self.previous_pos = origins.clone()
        self.reference_velocity = torch.zeros_like(origins)
        self.hit = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.awaiting_separation = torch.zeros_like(self.hit)
        self.in_hole = torch.zeros_like(self.hit)
        self.passed = torch.zeros_like(self.hit)
        self.best_distance = origins.new_zeros(self.num_envs)
        self.previous_best_distance = self.best_distance.clone()
        self.crossing_error = torch.full_like(self.best_distance, float("inf"))
        self.first_hit_this_step = torch.zeros_like(self.hit)
        self.first_pass_this_step = torch.zeros_like(self.hit)
        self.launch_error_this_step = self.crossing_error.clone()
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
        self.extras.pop("shot", None)
        self.extras.pop("log", None)
        distance = (self.paddle_center() - self.held_state[:, :3]).norm(dim=-1)
        self.best_distance = torch.where(self.episode_length_buf == 0, distance, self.best_distance)
        self.previous_best_distance.copy_(self.best_distance)
        self.first_hit_this_step.zero_()
        self.first_pass_this_step.zero_()
        self.launch_error_this_step.fill_(float("inf"))
        self.board_impacts_this_step.zero_()
        self._apply_support()
        return super().step(action.clamp(-10., 10.))

    def _apply_support(self):
        self.ball.set_external_force_and_torque(self.support * (~self.hit)[:, None, None], self.zero_torque)

    def _observe_physics(self):
        touching = paddle_contacts(self.contacts, self.robot, self.wrist)
        first_hit = touching & ~self.hit
        distance = (self.paddle_center() - self.held_state[:, :3]).norm(dim=-1)
        self.best_distance = torch.where(self.hit, self.best_distance, torch.minimum(self.best_distance, distance))
        self.first_hit_this_step |= first_hit
        self.hit |= first_hit
        self._apply_support()

        # A contact elsewhere on the arm must not displace a held ball.
        state = self.ball.data.root_state_w
        ids = ((~self.hit) & (state != self.held_state).any(dim=1)).nonzero().flatten()
        if len(ids):
            self.ball.write_root_state_to_sim(self.held_state[ids], env_ids=ids)

        self.awaiting_separation |= first_hit
        separated = self.awaiting_separation & ~touching
        velocity_error = (self.ball.data.root_lin_vel_w - self.reference_velocity).norm(dim=-1)
        self.launch_error_this_step = torch.where(separated, velocity_error, self.launch_error_this_step)
        self.awaiting_separation &= ~separated

        # Merge simultaneous contacts on adjacent wedges before counting a new impact.
        board_impulses = self.board_contacts.get_contact_force_matrix(dt=1.)
        touching_board = (board_impulses != 0).any(dim=(1, 2))
        self.board_impacts_this_step += self.hit & ~self.passed & touching_board & ~self.touching_board
        self.touching_board.copy_(touching_board)

        pos = self.ball.data.root_pos_w
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
        self.reference_velocity[env_ids] = target_velocity(
            state[:, :3], self.hole[env_ids], self.cfg.sim.gravity, self.physics_dt,
        )
        self.robot.set_joint_position_target(self.robot.data.joint_pos[env_ids], env_ids=env_ids)
        self.hit[env_ids] = False
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
            self.extras["shot"] = {"hit": hit, "passed": passed, "crossing_error": self.crossing_error[finished]}
        super()._reset_idx(env_ids)
        if len(finished):
            self.extras["log"].update({
                "Strike/hit_rate": hit.float().mean(),
                "Strike/pass_rate": passed.float().mean(),
                "Strike/pass_given_hit": passed.sum() / hit.sum().clamp_min(1),
            })
