"""On-policy sequences with separately masked teacher labels for auxiliary loss."""

import torch


def generalized_advantage(rewards, values, next_values, terminated, truncated,
                          valid, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (~terminated[t]) * next_values[t] - values[t]
        # Timeouts bootstrap from the FINAL observation, but do not propagate
        # advantages across the reset into a different episode.
        carry = delta + gamma * gae_lambda * (~(terminated[t] | truncated[t])) * carry
        carry = torch.where(valid[t], carry, 0.0)
        advantages[t] = carry
    return advantages, advantages + values


class RolloutBuffer:
    def __init__(self, steps, num_envs, hidden_dim, device):
        shape = (steps, num_envs)
        self.data = {
            "observations": torch.empty(*shape, 48, device=device),
            "actions": torch.empty(*shape, 8, device=device),
            "expert_actions": torch.zeros(*shape, 8, device=device),
            "hidden": torch.empty(*shape, hidden_dim, device=device),
        }
        for key in ("logp", "values", "rewards", "next_values"):
            self.data[key] = torch.empty(*shape, device=device)
        for key in ("starts", "terminated", "truncated", "valid"):
            self.data[key] = torch.empty(*shape, dtype=torch.bool, device=device)
        self.data["grip_decisions"] = torch.ones(*shape, dtype=torch.bool, device=device)
        self.data["expert_valid"] = torch.zeros(*shape, dtype=torch.bool, device=device)
        self.steps, self.num_envs = shape

    def add(self, step, **values):
        for name, value in values.items():
            self.data[name][step].copy_(value)

    def finish(self, gamma, gae_lambda):
        d = self.data
        advantages, returns = generalized_advantage(
            d["rewards"], d["values"], d["next_values"], d["terminated"],
            d["truncated"], d["valid"], gamma, gae_lambda)
        real = advantages[d["valid"]]
        if real.numel():
            advantages = (advantages - real.mean()) / real.std(unbiased=False).clamp_min(1e-8)
        d["advantages"], d["returns"] = advantages, returns

    def batches(self, sequence_length, sequences_per_batch):
        if self.steps % sequence_length:
            raise ValueError("rollout_steps must be divisible by sequence_length")
        # Flatten ENVIRONMENT and CHUNK axes, preserving the time axis.
        chunks = {}
        for name, tensor in self.data.items():
            chunks[name] = tensor.transpose(0, 1).reshape(
                -1, sequence_length, *tensor.shape[2:])
        eligible = chunks["valid"].any(-1).nonzero().flatten()
        order = eligible[torch.randperm(len(eligible), device=eligible.device)]
        for start in range(0, len(order), sequences_per_batch):
            indices = order[start:start + sequences_per_batch]
            batch = {name: tensor[indices] for name, tensor in chunks.items()}
            batch["hidden"] = batch["hidden"][:, 0].unsqueeze(0).contiguous()
            yield batch
