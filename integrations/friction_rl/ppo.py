"""CPU Gaussian policy cloning and PPO continuation with a motion prior."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from integrations.friction_rl.network import OBS_DIM, ACT_DIM

EXPLORE = (3, 4)


class ActorCritic(nn.Module):
    """Keep a Gaussian actor and a separate value network."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256),
                                   nn.Tanh(), nn.Linear(256, act_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256),
                                    nn.Tanh(), nn.Linear(256, 1))
        self.log_std = nn.Parameter(torch.zeros(act_dim))


def save(path, model):
    """Save only the policy and critic state dictionary."""
    torch.save(model.state_dict(), path)


def load(path):
    """Load a state dictionary into the fixed CPU architecture."""
    model = ActorCritic(OBS_DIM, ACT_DIM)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def act(model, obs, rng, stochastic):
    """Explore gripper and force, retaining raw samples before command clipping."""
    with torch.no_grad():
        mean = model.actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        std = model.log_std.exp().numpy()
    stored = mean.copy()
    if stochastic:
        stored[list(EXPLORE)] += std[list(EXPLORE)] * rng.standard_normal(len(EXPLORE))
    return stored, np.clip(stored, -1, 1)


def bc_fit(model, obs, labels, steps, batch, lr, seed, init_std):
    """Fit every actor output, record each hundredth loss and initialize exploration."""
    torch.manual_seed(seed)
    obs, labels = (torch.as_tensor(x, dtype=torch.float32) for x in (obs, labels))
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=lr)
    records = []
    for step in range(steps):
        indices = torch.randint(len(obs), (batch,))
        loss = (model.actor(obs[indices]) - labels[indices]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            records.append({"step": step + 1, "loss": float(loss.detach())})
    with torch.no_grad():
        model.log_std.copy_(torch.as_tensor(init_std, dtype=torch.float32).log())
    return records


def gae(rewards, values, dones, gamma, lam):
    """Compute terminal-aware advantages and returns for concatenated episodes."""
    advantages = np.empty(len(rewards), dtype=np.float32)
    advantage = next_value = 0.0
    for t in reversed(range(len(rewards))):
        live = 1 - float(dones[t])
        delta = rewards[t] + gamma * next_value * live - values[t]
        advantage = delta + gamma * lam * live * advantage
        advantages[t] = advantage
        next_value = values[t]
    return advantages, advantages + np.asarray(values)


def optimizers(model, cfg):
    """Create the actor and critic Adam optimizers once per training run."""
    actor_optimizer = torch.optim.Adam([*model.actor.parameters(), model.log_std], lr=cfg["actor_lr"])
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=cfg["critic_lr"])
    return actor_optimizer, critic_optimizer


def ppo_update(model, batch, demos, cfg, actor_optimizer, critic_optimizer, actor_frozen):
    """Update the critic and optionally the actor using PPO and a force-free motion prior."""
    obs, stored = (torch.as_tensor(batch[key], dtype=torch.float32) for key in ("obs", "stored"))
    demo_obs, labels = (torch.as_tensor(demos[key], dtype=torch.float32) for key in ("obs", "labels"))
    with torch.no_grad():
        logp_old = Normal(model.actor(obs)[:, EXPLORE], model.log_std[list(EXPLORE)].exp()).log_prob(
            stored[:, EXPLORE]).sum(-1)
        values = model.critic(obs).squeeze(-1).numpy()
    advantages, returns = gae(batch["reward"], values, batch["done"], cfg["gamma"], cfg["gae_lambda"])
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    returns = torch.as_tensor(returns, dtype=torch.float32)
    actor_params = [*model.actor.parameters(), model.log_std]
    metrics = []
    updates = 0
    stop = False
    for epoch in range(cfg["epochs"]):
        for indices in torch.randperm(len(obs)).split(cfg["minibatch"]):
            logp = Normal(model.actor(obs[indices])[:, EXPLORE], model.log_std[list(EXPLORE)].exp()).log_prob(
                stored[indices][:, EXPLORE]).sum(-1)
            log_ratio = logp - logp_old[indices]
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(ratio * advantages[indices],
                                        ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"])
                                        * advantages[indices]).mean()
            value_loss = (model.critic(obs[indices]).squeeze(-1) - returns[indices]).square().mean()
            demo_indices = torch.randint(len(demo_obs), (len(indices),))
            bc_loss = (model.actor(demo_obs[demo_indices])[:, :4] - labels[demo_indices, :4]).square().mean()
            approx_kl = ((ratio - 1) - log_ratio).mean()
            metrics.append([float(x.detach()) for x in (policy_loss, value_loss, bc_loss, approx_kl)])
            if float(approx_kl.detach()) > 1.5 * cfg["target_kl"]:
                stop = True
                break
            if not actor_frozen:
                actor_optimizer.zero_grad()
                (policy_loss + cfg["bc_weight"] * bc_loss).backward()
                nn.utils.clip_grad_norm_(actor_params, .5)
                actor_optimizer.step()
                updates += 1
            critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(model.critic.parameters(), .5)
            critic_optimizer.step()
        if stop:
            break
    return {**dict(zip(("policy_loss", "value_loss", "bc_loss", "approx_kl"),
                      np.mean(metrics, axis=0).tolist())), "epochs_run": epoch + 1, "updates": updates}
