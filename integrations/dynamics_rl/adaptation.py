"""Continue the policy chain with PPO and a per-rollout action-distribution trust region."""

import copy
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance

from integrations.dynamics_rl.learning import POLICY_FORMAT, ROBUST_INACTIVE_COLUMNS, load_model
from orchestrator.files import append_jsonl, write_json


PPO_KEYS = ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range",
            "target_kl", "max_grad_norm")


class TanhGaussianPolicy(ActorCriticPolicy):
    """A fixed-variance Gaussian whose mean exactly matches the SAC inference actor."""

    def _build(self, lr_schedule):
        self.net_arch = dict(pi=[256, 256], vf=[256, 256])
        self.activation_fn = torch.nn.ReLU
        super()._build(lr_schedule)
        self.log_std.requires_grad_(False)
        self.optimizer = torch.optim.Adam([
            {"params": [*self.mlp_extractor.policy_net.parameters(), *self.action_net.parameters()]},
            {"params": [*self.mlp_extractor.value_net.parameters(), *self.value_net.parameters()]},
        ], lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(torch.tanh(self.action_net(latent_pi)), self.log_std)


class AdaptationPPO(PPO):
    """Warm the value function each round, then constrain each complete PPO policy step."""

    def __init__(self, *args, training=None, **kwargs):
        self.training = training
        self.policy_updates = 0
        self.first_rollout = True
        super().__init__(*args, **kwargs)

    def _update_learning_rate(self, optimizers):
        for group, key in zip(self.policy.optimizer.param_groups,
                              ("actor_learning_rate", "value_learning_rate"), strict=True):
            group["lr"] = self.training[key]

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "parent_actor", "run_dir", "started",
                "initial_steps", "checkpoint_interval"]

    def train(self):
        observations = torch.as_tensor(self.rollout_buffer.observations.reshape(-1, 57), device=self.device)
        actor = self.policy.optimizer.param_groups[0]["params"]
        with torch.no_grad():
            old_mean = self.policy.get_distribution(observations).distribution.mean.clone()
            parent_mean = self.parent_actor(observations)
        value_only = self.first_rollout
        self._update_learning_rate(self.policy.optimizer)
        exact_kl, fraction = 0.0, 0.0
        if value_only:
            self.policy.set_training_mode(True)
            losses = []
            for _ in range(self.training["value_warmup_epochs"]):
                for batch in self.rollout_buffer.get(self.batch_size):
                    loss = torch.nn.functional.mse_loss(
                        self.policy.predict_values(batch.observations).flatten(), batch.returns)
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.policy.optimizer.step()
                    losses.append(loss.item())
            metrics = {"epochs_completed": self.training["value_warmup_epochs"], "policy_loss": 0.0,
                       "value_loss": float(np.mean(losses)), "approx_kl": 0.0, "clip_fraction": 0.0,
                       "explained_variance": float(explained_variance(
                           self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))}
        else:
            old = [parameter.detach().clone() for parameter in actor]
            first = self.policy.mlp_extractor.policy_net[0].weight
            mask = torch.ones_like(first)
            mask[:, ROBUST_INACTIVE_COLUMNS] = 0
            hook = first.register_hook(lambda gradient: gradient * mask)
            epochs_before = self._n_updates
            try:
                super().train()
            finally:
                hook.remove()
            with torch.no_grad():
                delta = [parameter - before for parameter, before in zip(actor, old, strict=True)]
                fraction = 1.0
                for attempt in range(12):
                    mean = self.policy.get_distribution(observations).distribution.mean
                    exact_kl = float(((mean - old_mean).square() /
                                      (2 * self.policy.log_std.exp().square())).sum(dim=1).mean())
                    if exact_kl <= self.training["max_update_kl"]:
                        break
                    fraction = fraction / 2 if attempt < 10 else 0.0
                    for parameter, before, change in zip(actor, old, delta, strict=True):
                        parameter.copy_(before + fraction * change)
                self.policy_updates += int(fraction > 0)
            logged = self.logger.name_to_value
            metrics = {"epochs_completed": self._n_updates - epochs_before,
                       "policy_loss": float(logged["train/policy_gradient_loss"]),
                       **{key: float(logged[f"train/{key}"]) for key in
                          ("value_loss", "approx_kl", "clip_fraction", "explained_variance")}}
        self.first_rollout = False
        with torch.no_grad():
            mean = self.policy.get_distribution(observations).distribution.mean
            drift = float((mean - parent_mean).abs().max(dim=1).values.mean())
        append_jsonl(self.run_dir / "rl-metrics.jsonl", {
            "online_steps": self.num_timesteps, "value_only": value_only, **metrics,
            "exact_kl": exact_kl, "step_fraction": fraction, "drift_from_parent": drift,
            "elapsed_seconds": time.time() - self.started,
        })
        if (self.num_timesteps - self.initial_steps) % self.checkpoint_interval == 0:
            directory = self.run_dir / "checkpoints"
            directory.mkdir(parents=True, exist_ok=True)
            self.save(directory / f"step_{self.num_timesteps:06d}.zip")


def convert_p0(sac_training_zip, training, seed, output_zip):
    """Transplant P0's deterministic actor into a newly seeded PPO learner."""
    torch.set_num_threads(training["learner_threads"])
    source = load_model(sac_training_zip)
    env = gym.Env()
    env.observation_space = gym.spaces.Box(-np.inf, np.inf, (57,), dtype=np.float32)
    env.action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)
    model = AdaptationPPO(TanhGaussianPolicy, env, training=dict(training), seed=seed, device="cpu",
                          learning_rate=training["value_learning_rate"], ent_coef=0,
                          **{key: training[key] for key in PPO_KEYS})
    model.policy.mlp_extractor.policy_net.load_state_dict(source.actor.latent_pi.state_dict())
    model.policy.action_net.load_state_dict(source.actor.mu.state_dict())
    model.policy.log_std.data.fill_(training["exploration_log_std"])
    model._update_learning_rate(model.policy.optimizer)
    model.num_timesteps = source.num_timesteps
    model.save(output_zip)


def load_round_model(training_zip, env, training, seed):
    """Restore the chain and apply the frozen configuration to a fresh round."""
    model = AdaptationPPO.load(training_zip, env=env, device="cpu", custom_objects={
        **{key: training[key] for key in PPO_KEYS}, "ent_coef": 0,
        "training": dict(training), "learning_rate": training["value_learning_rate"],
    })
    # SB3 restores optimizer groups and log_std after constructing the policy.
    model._update_learning_rate(model.policy.optimizer)
    model.policy.log_std.data.fill_(training["exploration_log_std"])
    model._last_obs = None
    model.set_random_seed(seed)
    model.first_rollout = True
    model.parent_actor = copy.deepcopy(torch.nn.Sequential(
        model.policy.mlp_extractor.policy_net, model.policy.action_net, torch.nn.Tanh())).requires_grad_(False)
    return model


def train_round(model, actions, *, on_episode, run_dir, checkpoint_interval):
    """Collect exact rollout multiples and retain metrics, checkpoints and partial episodes."""
    model.run_dir = Path(run_dir)
    model.started, model.initial_steps = time.time(), model.num_timesteps
    model.checkpoint_interval = checkpoint_interval
    env = model.get_env().envs[0].unwrapped

    def episode_finished(record):
        append_jsonl(model.run_dir / "online-episodes.jsonl", record)
        on_episode(record)

    env.on_episode = episode_finished
    model.set_logger(configure(str(model.run_dir / "learner"), ["csv"]))
    model.learn(actions, reset_num_timesteps=False, callback=lambda _locals, _globals: True)
    if env.active:
        reason = ("Unused vector auto-reset" if env.episode_actions == 0 else
                  "Online action budget reached before episode termination")
        env._finish_episode(complete=False, error=reason)
    model._last_obs = None
    model.logger.close()


def save_network(model, directory, metadata):
    """Save the PPO state and the existing inference payload format."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "training.zip")
    torch.save({
        "format": POLICY_FORMAT, "observation_dim": 57, "action_dim": 7, "hidden_sizes": [256, 256],
        "latent_state": model.policy.mlp_extractor.policy_net.state_dict(),
        "mean_state": model.policy.action_net.state_dict(),
        "online_steps": int(model.num_timesteps), "rl_updates": model.policy_updates,
        "metadata": {**metadata, "learner": {"algorithm": "ppo", **model.training,
                                              "masked_actor_columns": list(ROBUST_INACTIVE_COLUMNS)}},
    }, directory / "policy.pt")
    write_json(directory / "network.json", {
        **metadata, "online_steps": int(model.num_timesteps),
        "policy_updates": model.policy_updates, "created_at": time.time(),
    })
