"""Clipped recurrent PPO with a separate auxiliary expert loss."""

import math
import torch


class PPO:
    def __init__(self, policy, config, expert_loader=None):
        self.policy, self.config = policy, config
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config["lr"])
        self.expert_loader = expert_loader
        self.expert_iterator = iter(expert_loader) if expert_loader is not None else None

    def expert_batch(self):
        try:
            batch = next(self.expert_iterator)
        except StopIteration:
            self.expert_iterator = iter(self.expert_loader)
            batch = next(self.expert_iterator)
        return tuple(t.to(self.policy.obs_mean.device) for t in batch)

    def update(self, buffer, bc_weight):
        cfg, policy = self.config, self.policy
        policy.train()
        totals, count, stop, stop_kl, processed = {}, 0, False, None, 0
        length = cfg["sequence_length"]
        if "num_mini_batches" in cfg:
            eligible = buffer.data["valid"].transpose(0, 1).reshape(-1, length).any(-1).sum().item()
            batch_size = max(1, math.ceil(eligible / cfg["num_mini_batches"]))
        else:
            batch_size = cfg["sequences_per_batch"]
        for _ in range(cfg["epochs"]):
            for b in buffer.batches(length, batch_size):
                mask = b["valid"]
                raw, value, _ = policy(b["observations"], b["hidden"], b["starts"])
                logp, entropy = policy.action_statistics(raw, b["actions"], b["grip_decisions"])
                log_ratio = logp[mask] - b["logp"][mask]
                ratio = log_ratio.exp()
                kl = ((ratio - 1) - log_ratio).mean()
                lr = self.optimizer.param_groups[0]["lr"]
                desired = cfg["target_kl"]
                if desired and kl.item() > desired * 2:
                    lr = max(cfg.get("min_lr", 1e-6), lr / 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = lr
                elif desired and 0 < kl.item() < desired / 2 and count > 0:
                    lr = min(cfg["lr"], lr * 1.1)
                    for group in self.optimizer.param_groups:
                        group["lr"] = lr
                if desired and kl.item() > desired * 4:
                    stop_kl = kl.item()
                    stop = True
                    break
                advantage = b["advantages"][mask]
                actor_loss = -torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1 - cfg["clip_ratio"], 1 + cfg["clip_ratio"]) * advantage).mean()
                value_loss = 0.5 * (value[mask] - b["returns"][mask]).square().mean()
                entropy_mean = entropy[mask].mean()
                bc_loss = value.new_zeros(())
                if self.expert_loader is not None and bc_weight > 0:
                    bc_loss = policy.imitation_loss(*self.expert_batch())
                online_loss = value.new_zeros(())
                supervised = mask & b["expert_valid"]
                if (cfg.get("online_expert_weight", 0) > 0
                        and not cfg.get("online_expert_epochs", 0) and supervised.any()):
                    online_loss = policy.supervision_loss(raw, b["expert_actions"], supervised)
                loss = (actor_loss + cfg["value_coef"] * value_loss
                        - cfg["entropy_coef"] * entropy_mean + bc_weight * bc_loss
                        + cfg.get("online_expert_weight", 0) * online_loss)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()
                processed += int(mask.sum())
                for key, item in dict(actor_loss=actor_loss, value_loss=value_loss,
                                      entropy=entropy_mean, bc_loss=bc_loss, online_loss=online_loss, kl=kl,
                                      grad_norm=grad_norm).items():
                    totals[key] = totals.get(key, 0.0) + item.detach().item()
                count += 1
            if stop:
                break
        # DAgger-style regression after PPO. These labels are never treated as
        # sampled actions. The next rollout is collected from the updated policy.
        expert_steps, expert_total = 0, 0.0
        ppo_lr = self.optimizer.param_groups[0]["lr"]
        if cfg.get("online_expert_weight", 0) > 0:
            for group in self.optimizer.param_groups:
                group["lr"] = cfg.get("online_expert_lr", cfg["lr"])
            for _ in range(cfg.get("online_expert_epochs", 0)):
                for b in buffer.batches(length, batch_size):
                    supervised = b["valid"] & b["expert_valid"]
                    if not supervised.any():
                        continue
                    raw, _, _ = policy(b["observations"], b["hidden"], b["starts"])
                    auxiliary = policy.supervision_loss(raw, b["expert_actions"], supervised)
                    self.optimizer.zero_grad(set_to_none=True)
                    (cfg["online_expert_weight"] * auxiliary).backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
                    self.optimizer.step()
                    expert_total += auxiliary.detach().item()
                    expert_steps += 1
            for group in self.optimizer.param_groups:
                group["lr"] = ppo_lr
        return {**{k: v / count for k, v in totals.items()},
                "optimizer_steps": count, "kl_early_stop": stop, "stop_kl": stop_kl,
                "valid_frames": int(buffer.data["valid"].sum()), "processed_frames": processed,
                "learning_rate": self.optimizer.param_groups[0]["lr"], "bc_weight": bc_weight,
                "online_expert_steps": expert_steps,
                "online_expert_loss": expert_total / max(expert_steps, 1)}
