"""Clock-free actor learning and exports for the fixability study."""

import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecEnvWrapper

from integrations.dynamics_rl.adaptation import PPO_KEYS
from orchestrator.files import append_jsonl, write_json


def hidden_layers(width):
    return torch.nn.Sequential(torch.nn.Linear(width, 256), torch.nn.ReLU(),
                               torch.nn.Linear(256, 256), torch.nn.ReLU())


class StateExtractor(torch.nn.Module):
    """Give the actor state alone and the critic the complete observation."""

    def __init__(self):
        super().__init__()
        self.latent_dim_pi = self.latent_dim_vf = 256
        self.policy_net, self.value_net = hidden_layers(49), hidden_layers(57)

    def forward_actor(self, observations):
        return self.policy_net(observations[:, :49])

    def forward_critic(self, observations):
        return self.value_net(observations)

    def forward(self, observations):
        return self.forward_actor(observations), self.forward_critic(observations)


class FixabilityPolicy(ActorCriticPolicy):
    """Separate actor and critic MLPs with a fixed diagonal Gaussian variance."""

    def _build_mlp_extractor(self):
        self.mlp_extractor = StateExtractor().to(self.device)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.log_std.requires_grad_(False)
        self.optimizer = torch.optim.Adam([
            {"params": [*self.mlp_extractor.policy_net.parameters(), *self.action_net.parameters()]},
            {"params": [*self.mlp_extractor.value_net.parameters(), *self.value_net.parameters()]},
        ], lr=lr_schedule(1), **self.optimizer_kwargs)


class RewardScale(VecEnvWrapper):
    """Scale rewards after RemoteEnv has recorded the raw episode evidence."""

    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs, rewards * self.scale, dones, infos


class FixabilityPPO(PPO):
    """Warm the critic and backtrack each actor update to its exact KL bound."""

    def __init__(self, *args, training=None, **kwargs):
        self.training = training
        self.warmup_remaining = 0 if training is None else training["value_warmup_rollouts"]
        self.rollout_episodes = []
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        self.warmup_remaining = self.training["value_warmup_rollouts"]
        self.rollout_episodes = []

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "run_dir", "started", "rollout_episodes"]

    def _update_learning_rate(self, optimizers):
        for group, key in zip(self.policy.optimizer.param_groups,
                              ("actor_learning_rate", "value_learning_rate"), strict=True):
            group["lr"] = self.training[key]

    def train(self):
        observations = torch.as_tensor(self.rollout_buffer.observations.reshape(-1, 57), device=self.device)
        actor = self.policy.optimizer.param_groups[0]["params"]
        with torch.no_grad():
            old_mean = self.policy.get_distribution(observations).distribution.mean.clone()
        value_only = self.warmup_remaining > 0
        self._update_learning_rate(self.policy.optimizer)
        exact_kl, fraction = 0.0, 0.0
        if value_only:
            self.policy.set_training_mode(True)
            losses = []
            for _ in range(self.n_epochs):
                for batch in self.rollout_buffer.get(self.batch_size):
                    loss = torch.nn.functional.mse_loss(
                        self.policy.predict_values(batch.observations).flatten(), batch.returns)
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    losses.append(loss.item())
            self._n_updates += self.n_epochs
            self.warmup_remaining -= 1
            metrics = {"policy_loss": 0.0, "value_loss": float(np.mean(losses)),
                       "approx_kl": 0.0, "clip_fraction": 0.0,
                       "explained_variance": float(explained_variance(
                           self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))}
        else:
            old = [parameter.detach().clone() for parameter in actor]
            super().train()
            with torch.no_grad():
                delta = [parameter - before for parameter, before in zip(actor, old, strict=True)]
                fraction = 1.0
                for attempt in range(12):
                    mean = self.policy.get_distribution(observations).distribution.mean
                    exact_kl = float(((mean - old_mean).square() /
                                      (2 * self.policy.log_std.exp().square())).sum(1).mean())
                    if exact_kl <= self.training["max_update_kl"]:
                        break
                    fraction = fraction / 2 if attempt < 10 else 0.0
                    for parameter, before, change in zip(actor, old, delta, strict=True):
                        parameter.copy_(before + fraction * change)
            logged = self.logger.name_to_value
            metrics = {"policy_loss": float(logged["train/policy_gradient_loss"]),
                       **{key: float(logged[f"train/{key}"]) for key in
                          ("value_loss", "approx_kl", "clip_fraction", "explained_variance")}}
        completed = [row for row in self.rollout_episodes if row["complete"]]
        append_jsonl(self.run_dir / "rl-metrics.jsonl", {
            "online_steps": self.num_timesteps, "value_only": value_only, **metrics,
            "exact_kl": exact_kl, "step_fraction": fraction,
            "online_episodes": len(completed), "online_successes": sum(r["success"] for r in completed),
            "elapsed_seconds": time.time() - self.started,
        })
        self.rollout_episodes.clear()


def create_model(config, env=None):
    """Construct the same CPU PPO learner for BC and both physics conditions."""
    torch.set_num_threads(config["learner_threads"])
    if env is None:
        env = gym.Env()
        env.observation_space = gym.spaces.Box(-np.inf, np.inf, (57,), dtype=np.float32)
        env.action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)
    model = FixabilityPPO(FixabilityPolicy, env, training=dict(config), device="cpu",
                          seed=config["seed"], learning_rate=config["value_learning_rate"],
                          **{key: config[key] for key in PPO_KEYS})
    model.policy.log_std.data.copy_(torch.tensor(config["log_std"]))
    model._update_learning_rate(model.policy.optimizer)
    return model


def load_learner(path, env, config):
    """Restore weights and optimizer, applying the frozen study configuration."""
    torch.set_num_threads(config["learner_threads"])
    model = FixabilityPPO.load(path, env=env, device="cpu", custom_objects={
        **{key: config[key] for key in PPO_KEYS}, "training": dict(config),
        "learning_rate": config["value_learning_rate"], "n_envs": env.num_envs,
    })
    model.policy.log_std.data.copy_(torch.tensor(config["log_std"]))
    model._update_learning_rate(model.policy.optimizer)
    model.set_random_seed(config["seed"])
    return model


def bc_train(observations, labels, config, *, run_dir):
    """Fit clean action labels uniformly, leaving the critic untrained."""
    model = create_model(config)
    actor = model.policy.optimizer.param_groups[0]["params"]
    optimizer = torch.optim.Adam(actor, lr=config["bc_learning_rate"])
    rng = np.random.default_rng(config["seed"])
    observations, labels = torch.as_tensor(observations), torch.as_tensor(labels)
    for update in range(1, config["bc_updates"] + 1):
        indices = rng.integers(len(observations), size=config["bc_batch_size"])
        mean = model.policy.get_distribution(observations[indices]).distribution.mean
        loss = torch.nn.functional.mse_loss(mean, labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update % 1000 == 0 or update == config["bc_updates"]:
            append_jsonl(Path(run_dir) / "bc.jsonl", {"updates": update, "mse": loss.item()})
    return model


def save_network(model, directory, metadata):
    """Save the learner, standalone actor and human-readable network record."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "training.zip")
    actor = torch.nn.Sequential(model.policy.mlp_extractor.policy_net, model.policy.action_net)
    torch.save({"format": "dynamics-fixability-actor-v1", "actor_state": actor.state_dict(),
                "log_std": model.policy.log_std.detach().cpu(), "actor_columns": 49,
                "metadata": metadata}, directory / "policy.pt")
    write_json(directory / "network.json", {**metadata, "online_steps": model.num_timesteps})


class LoadedActor:
    """Clipped actor inference with a private seeded NumPy exploration stream."""

    def __init__(self, payload, seed):
        self.actor = torch.nn.Sequential(hidden_layers(49), torch.nn.Linear(256, 7))
        self.actor.load_state_dict(payload["actor_state"])
        self.actor.eval()
        self.sigma = np.exp(payload["log_std"].numpy())
        self.rng = np.random.default_rng(seed)
        self.metadata = payload["metadata"]

    def predict(self, obs, deterministic=True):
        with torch.inference_mode():
            mean = self.actor(torch.as_tensor(np.asarray(obs, dtype=np.float32)[..., :49])).numpy()
        if not deterministic:
            mean = mean + self.sigma * self.rng.standard_normal(mean.shape)
        return np.clip(mean, -1, 1), None


def load_actor(path, seed=0):
    return LoadedActor(torch.load(path, map_location="cpu", weights_only=True), seed)
