"""CPU checks for PPO math and episode boundaries, without Isaac Sim imports."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("torch", "yaml"))
if HAS_DEPS:
    import torch
    import yaml
    from AgenticSimulator.tasks.pick_cube.RL.PPO.buffer import RolloutBuffer, generalized_advantage
    from AgenticSimulator.tasks.pick_cube.RL.PPO.model import ActorCritic
    from AgenticSimulator.tasks.pick_cube.RL.PPO.ppo import PPO
    from AgenticSimulator.tasks.pick_cube.RL.PPO.rewards import TaskReward, physical_metrics
    from AgenticSimulator.tasks.pick_cube.RL.PPO.runtime import ROOT, save_checkpoint
    from AgenticSimulator.tasks.pick_cube.RL.PPO.curriculum import Curriculum


@unittest.skipUnless(HAS_DEPS, "PPO CPU checks require torch and PyYAML")
class TestPickCubePPO(unittest.TestCase):
    def test_absolute_goal_holds_pose_and_corrects_displacement(self):
        policy = ActorCritic(hidden_dim=16, recurrent_dim=8, absolute_targets=True)
        raw = torch.zeros(2, 9, requires_grad=True)
        obs = torch.zeros(2, 48)
        obs[:, 16:19] = torch.tensor([0.4, 0., 0.25])
        obs[:, 20] = 1.0  # downward-facing reference
        obs[1, 16] += 0.01
        obs[1, 20] = torch.cos(torch.tensor(0.1))
        obs[1, 21] = torch.sin(torch.tensor(0.1))  # world-z rotation +0.2 rad
        residual = policy.target_residual(raw, obs)
        physical = residual[:, :6] * policy.pose_scale
        torch.testing.assert_close(physical[0], torch.zeros(6))
        torch.testing.assert_close(physical[1], torch.tensor([-0.01, 0., 0., 0., 0., -0.2]))
        residual.square().sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_curriculum_ignores_retained_levels_and_waits_for_window(self):
        curriculum = Curriculum(dict(window=3, promotion_rate=2/3, evaluation_interval=5), level=2)
        curriculum.observe([dict(curriculum_level=1, success=True)] * 10)
        self.assertFalse(curriculum.ready(10))
        curriculum.observe([dict(curriculum_level=2, success=s) for s in (True, False, True)])
        self.assertFalse(curriculum.ready(4))
        self.assertTrue(curriculum.ready(5))
        curriculum.promote()
        self.assertEqual(curriculum.level, 3)
        self.assertFalse(curriculum.ready(10))

    def test_expert_prefix_does_not_propagate_advantage(self):
        reward = torch.tensor([[100.], [100.], [1.], [2.]])
        values = torch.zeros_like(reward)
        done = torch.zeros_like(reward, dtype=torch.bool)
        valid = torch.tensor([[False], [False], [True], [True]])
        advantage, _ = generalized_advantage(reward, values, values, done, done, valid, 1., 1.)
        torch.testing.assert_close(advantage, torch.tensor([[0.], [0.], [3.], [2.]]))

    def test_completion_outweighs_holding_dense_reward_forever(self):
        c = self.config["reward"]
        maximum = sum(c[k] for k in ("reach", "alignment", "lift", "transport", "place", "retreat"))
        stationary_return = c["dense_scale"] * maximum / 50 / (1 - self.config["ppo"]["gamma"])
        self.assertGreater(c["success_bonus"], stationary_return)

    def test_placed_cube_rewards_retreat_not_pressing(self):
        monitor = TaskReward(2, "cpu", self.config["reward"], 0.998)
        m = dict(grasp_distance=torch.tensor([0.01, 0.15]), cube_rise=torch.zeros(2),
                 xy_distance=torch.full((2,), 0.01), height_error=torch.full((2,), 0.009),
                 alignment=torch.ones(2))
        potential = monitor.compute_potential(m, torch.ones(2, dtype=torch.bool))
        self.assertGreater(potential[1], potential[0])

    def test_curriculum_expands_suffix_before_changing_stage(self):
        c = Curriculum(dict(window=1, promotion_rate=1, evaluation_interval=1,
                            handoff_fractions=[0.9, 0.5, 0.0]), level=2)
        c.promote()
        self.assertEqual((c.level, c.progress_index), (2, 1))
        c.observe([dict(curriculum_level=2, curriculum_progress_index=0, success=True)])
        self.assertFalse(c.ready(1))
        c.observe([dict(curriculum_level=2, curriculum_progress_index=1, success=True)])
        self.assertTrue(c.ready(1))
        c.promote()
        c.promote()
        self.assertEqual((c.level, c.progress_index), (3, 0))

    def setUp(self):
        torch.manual_seed(7)
        self.config = yaml.safe_load((ROOT / "config.yaml").read_text())
        self.model = ActorCritic(hidden_dim=16, recurrent_dim=8)

    def test_action_density_and_recurrent_reset(self):
        obs = torch.randn(2, 5, 48)
        starts = torch.tensor([[True, False, False, True, False],
                               [True, False, True, False, False]])
        hidden = torch.randn(1, 2, 8)
        raw, values, final = self.model(obs, hidden, starts)
        for env, boundary in ((0, 3), (1, 2)):
            expected, expected_value, expected_hidden = self.model(obs[env:env+1, boundary:])
            torch.testing.assert_close(raw[env:env+1, boundary:], expected)
            torch.testing.assert_close(values[env:env+1, boundary:], expected_value)
            torch.testing.assert_close(final[:, env:env+1], expected_hidden)
        action, old_logp, _, _ = self.model.act(obs, hidden, starts)
        new_logp, _ = self.model.action_statistics(raw, action)
        torch.testing.assert_close((new_logp - old_logp).exp(), torch.ones_like(old_logp))
        self.assertTrue(torch.all((action[..., 6] == 0) | (action[..., 6] == 1)))
        self.assertTrue(torch.all(action[..., 7].abs() <= 1))
        self.assertTrue(torch.all(action[..., 7][action[..., 6] == 0] == -1))

    def test_force_is_sampled_scored_and_supervised_only_when_closed(self):
        obs = torch.zeros(64, 48)
        with torch.no_grad():
            self.model.actor.weight.zero_()
            self.model.actor.bias[6] = 20  # deterministically close for this check
        action, logp, _, _ = self.model.act(obs)
        self.assertGreater(action[:, 7].std().item(), 0.1)
        raw, _, _ = self.model(obs)
        torch.testing.assert_close(logp, self.model.action_statistics(raw, action)[0])
        deterministic = self.model.act(obs, deterministic=True)[0]
        torch.testing.assert_close(deterministic[:, 7], torch.zeros(64))  # starts at 10 N
        changed = raw.clone()
        changed[:, 7] += 2.0
        self.assertFalse(torch.allclose(self.model.action_statistics(raw, action)[0],
                                        self.model.action_statistics(changed, action)[0]))
        action[:, 6] = 0
        action[:, 7] = -1
        torch.testing.assert_close(self.model.action_statistics(raw, action)[0],
                                   self.model.action_statistics(changed, action)[0])
        with torch.no_grad():
            self.model.actor.bias[7] += 2.0
        labels = action[:2, None].clone()
        labels[..., 6:] = labels.new_tensor([1., 0.])  # expert closes at 10 N
        self.model.imitation_loss(obs[:2, None], labels, torch.ones(2, 1, dtype=torch.bool)).backward()
        self.assertGreater(self.model.actor.bias.grad[7:].abs().sum().item(), 0)
        self.model.zero_grad()
        labels[..., 6] = 0
        self.model.imitation_loss(obs[:2, None], labels, torch.ones(2, 1, dtype=torch.bool)).backward()
        torch.testing.assert_close(self.model.actor.bias.grad[7:], torch.zeros(2))

    def test_held_grip_has_no_bernoulli_likelihood_but_force_is_still_sampled(self):
        obs = torch.zeros(32, 48)
        obs[:, 46] = 1  # actual previous command was closed
        decisions = torch.zeros(32, dtype=torch.bool)
        action, logp, _, _ = self.model.act(obs, grip_decisions=decisions)
        self.assertTrue(torch.all(action[:, 6] == 1))
        self.assertGreater(action[:, 7].std().item(), 0.1)
        raw, _, _ = self.model(obs)
        changed = raw.clone()
        changed[:, 6] += 20
        score, entropy = self.model.action_statistics(raw, action, decisions)
        new_score, new_entropy = self.model.action_statistics(changed, action, decisions)
        torch.testing.assert_close(score, logp)
        torch.testing.assert_close(score, new_score)
        torch.testing.assert_close(entropy, new_entropy)
        self.model.grip_decimation = 10
        torch.testing.assert_close(self.model.grip_decision_mask(torch.tensor([0, 1, 9, 10])),
                                   torch.tensor([True, False, False, True]))

    def test_shared_executor_applies_policy_force_per_environment(self):
        import importlib
        from types import ModuleType
        from unittest.mock import Mock, patch

        # Import the real executor with only Isaac's imports stubbed; exercise
        # physics_step -> _send -> robot effort target without a GPU simulator.
        modules = {name: ModuleType(name) for name in
                   ("isaaclab", "isaaclab.controllers", "isaaclab.utils", "isaaclab.utils.math")}
        for name in ("DifferentialIKController", "DifferentialIKControllerCfg"):
            setattr(modules["isaaclab.controllers"], name, Mock())
        for name in ("apply_delta_pose", "combine_frame_transforms", "compute_pose_error",
                     "quat_apply", "subtract_frame_transforms", "matrix_from_quat"):
            setattr(modules["isaaclab.utils.math"], name, Mock())
        with patch.dict("sys.modules", modules):
            control = importlib.import_module("AgenticSimulator.tasks.pick_cube.RL.BC.control")
            executor = control.BCExpertController.__new__(control.BCExpertController)
            executor.robot = Mock()
            executor.robot.root_physx_view.get_gravity_compensation_forces.return_value = torch.zeros(3, 9)
            executor.arm_ids, executor.finger_ids = list(range(7)), [7, 8]
            executor.last_arm = torch.zeros(3, 7)
            executor.grip_force = 10.0  # expert default must not override policy
            executor.state = torch.zeros(3, dtype=torch.long)
            executor.finger_force_mode = torch.tensor([True, True, False])
            executor._has_closed = executor.finger_force_mode.clone()
            executor._tracking = torch.ones(3, dtype=torch.bool)
            executor.width = torch.full((3,), 0.04)
            executor._held_position = torch.zeros(3, 3)
            executor._held_quaternion = torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
            executor.previous_action = torch.zeros(3, 8)
            executor.previous_action[:, 7] = torch.tensor([-0.5, 0.5, 0.8])
            executor._track_pose = Mock()
            executor._sync_gripper_modes = Mock()
            executor.physics_step(0.005)
            sent = executor.robot.set_joint_effort_target.call_args
            torch.testing.assert_close(sent.args[0], torch.tensor([[-5., -5.], [-15., -15.], [0., 0.]]))
            self.assertEqual(sent.kwargs["joint_ids"], [7, 8])
            # PPO opening must not pull asymmetric force-controlled fingers
            # back to a common zero target and squeeze/recenter the cube.
            ppo_control = importlib.import_module("AgenticSimulator.tasks.pick_cube.RL.PPO.control")
            executor.__class__ = ppo_control.PoseExecutor
            executor.robot.data.joint_pos = torch.zeros(3, 9)
            executor.robot.data.joint_pos[2, 7:] = torch.tensor([0.025, 0.0001])
            executor.opening_widths = torch.zeros(3, 2)
            executor.opening_limits = torch.full((3, 2), 0.03)
            executor.finger_force_mode[2] = True
            actions = torch.zeros(3, 8)
            actions[:2, 6] = 1
            with patch.object(control.BCExpertController, "set_action"):
                executor.set_action(actions)
            executor.finger_force_mode[2] = False
            executor._has_closed[2] = True
            executor.physics_step(0.005)
            target = executor.robot.set_joint_position_target.call_args.args[0]
            torch.testing.assert_close(target[2], torch.tensor([0.02525, 0.00035]))

    def test_gae_timeout_bootstraps_but_cuts_reset(self):
        rewards = torch.tensor([[1., 1., 100.], [99., 99., 2.]])
        values = torch.zeros_like(rewards)
        next_values = torch.tensor([[10., 10., 10.], [0., 0., 0.]])
        terminated = torch.tensor([[True, False, False], [True, True, True]])
        truncated = torch.tensor([[False, True, False], [False, False, False]])
        valid = torch.tensor([[True, True, False], [True, True, True]])
        advantage, returns = generalized_advantage(
            rewards, values, next_values, terminated, truncated, valid, 0.9, 1.)
        torch.testing.assert_close(advantage, torch.tensor([[1., 10., 0.], [99., 99., 2.]]))
        torch.testing.assert_close(returns, advantage)

    def test_sequence_buffer_preserves_order_and_initial_hidden(self):
        buffer = RolloutBuffer(8, 2, 8, "cpu")
        for value in buffer.data.values():
            value.zero_()
        buffer.data["valid"].fill_(True)
        for t in range(8):
            buffer.data["observations"][t, :, 0] = torch.tensor([t, t + 100])
            buffer.data["hidden"][t, :, 0] = torch.tensor([t + 200, t + 300])
        buffer.finish(0.99, 0.95)
        batch = next(buffer.batches(4, 4))
        torch.testing.assert_close(batch["observations"][:, 1:, 0] - batch["observations"][:, :-1, 0], torch.ones(4, 3))
        torch.testing.assert_close(batch["hidden"][0, :, 0], batch["observations"][:, 0, 0] + 200)

    def test_auxiliary_update_and_checkpoint_roundtrip(self):
        buffer = RolloutBuffer(8, 2, 8, "cpu")
        hidden = self.model.initial_hidden(2)
        for t in range(8):
            obs = torch.randn(2, 48)
            starts = torch.full((2,), t == 0 or t == 4)
            action, logp, value, next_hidden = self.model.act(obs, hidden, starts)
            buffer.add(t, observations=obs, actions=action, logp=logp, values=value,
                       hidden=hidden[0], starts=starts, rewards=torch.ones(2),
                       next_values=torch.zeros(2), terminated=torch.full((2,), t == 3 or t == 7),
                       truncated=torch.zeros(2, dtype=torch.bool), valid=torch.ones(2, dtype=torch.bool))
            hidden = next_hidden
        buffer.finish(0.99, 0.95)
        demos = torch.randn(2, 5, 48)
        sampled_actions = buffer.data["actions"].clone()
        buffer.data["expert_valid"].fill_(True)
        buffer.data["expert_actions"][..., 6] = 1.0
        actions = torch.zeros(2, 5, 8)
        actions[..., 7] = -1
        mask = torch.ones(2, 5, dtype=torch.bool)
        cfg = dict(self.config["ppo"], sequence_length=4, sequences_per_batch=2, epochs=2)
        trainer = PPO(self.model, cfg, [(demos, actions, mask)])
        before = self.model.actor.weight.detach().clone()
        metrics = trainer.update(buffer, 0.1)
        self.assertGreater(metrics["optimizer_steps"], 0)
        self.assertGreater(metrics["bc_loss"], 0)
        self.assertGreater(metrics["online_expert_steps"], 0)
        self.assertGreater(metrics["online_expert_loss"], 0)
        torch.testing.assert_close(buffer.data["actions"], sampled_actions)
        self.assertFalse(torch.equal(before, self.model.actor.weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in self.model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(path, self.model, trainer.optimizer, self.config, 1, 16)
            loaded = torch.load(path, weights_only=True)
            other = ActorCritic(hidden_dim=16, recurrent_dim=8)
            other.load_state_dict(loaded["model"])
            torch.testing.assert_close(self.model.act(demos, deterministic=True)[0],
                                       other.act(demos, deterministic=True)[0])

    def test_success_requires_lift_release_and_stability_and_resets_locally(self):
        monitor = TaskReward(2, "cpu", self.config["reward"], 0.998)
        m = {k: torch.zeros(2) for k in ("cube_rise", "grasp_distance", "xy_distance", "height_error", "alignment")}
        monitor.reset(torch.arange(2), m)
        active = torch.ones(2, dtype=torch.bool)
        opened = torch.zeros(2, dtype=torch.bool)
        for _ in range(30):
            monitor.observe(m, opened, active, 0.02)
        self.assertFalse(monitor.success.any())
        m["cube_rise"].fill_(0.06)
        monitor.observe(m, ~opened, active, 0.02)
        m["cube_rise"].zero_()
        for _ in range(30):
            monitor.observe(m, opened, active, 0.02)
        self.assertTrue(monitor.success.all())
        monitor.reset(torch.tensor([0]), m)
        self.assertFalse(monitor.seen_lift[0])
        self.assertTrue(monitor.success[1])
        self.assertGreater(monitor.stable_seconds[1], 0.5)

    def test_physical_metrics_and_fixed_normalization(self):
        obs = torch.zeros(2, 48)
        obs[:, 20] = 1  # hand qx=1: top-down
        obs[:, 26] = obs[:, 36] = 1  # cube/plate identity
        obs[:, 18] = 0.2034
        obs[:, 25] = 0.1
        obs[:, 35] = 0.1053308  # equal bottom centers
        m = physical_metrics(obs, torch.full((2,), 0.04))
        torch.testing.assert_close(m["grasp_distance"], torch.zeros(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(m["height_error"], torch.zeros(2), atol=1e-6, rtol=0)
        torch.testing.assert_close(m["alignment"], torch.ones(2))
        self.model.set_normalization(dict(obs_mean=torch.zeros(48), obs_std=torch.full((48,), 1e-8)))
        self.assertGreaterEqual(self.model.obs_std.min().item(), 0.009)
        self.assertTrue(torch.isfinite(self.model(obs)[0]).all())


if __name__ == "__main__":
    unittest.main()
