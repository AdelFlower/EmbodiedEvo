"""Record native task outcomes before automatic episode resets."""

import torch


class EpisodeMetrics:
    def __init__(self, num_envs, episodes_per_env, device):
        self.limit = episodes_per_env
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.success = self.lifted.clone()
        self.steps = self.count.clone()
        self.records = []

    @property
    def complete(self):
        return bool((self.count >= self.limit).all())

    def update(self, lifted, success, done):
        self.lifted |= lifted
        self.success |= success
        self.steps += 1
        for i in done.nonzero().flatten().tolist():
            if self.count[i] < self.limit:
                self.records.append({"env_id": i, "episode": self.count[i].item(),
                                     "lifted": self.lifted[i].item(), "success": self.success[i].item(),
                                     "steps": self.steps[i].item()})
                self.count[i] += 1
            self.lifted[i] = self.success[i] = False
            self.steps[i] = 0

    def summary(self):
        n = len(self.records)
        return {"episodes": n, "complete": self.complete,
                "lift_rate": sum(r["lifted"] for r in self.records)/n if n else None,
                "success_rate": sum(r["success"] for r in self.records)/n if n else None,
                "records": self.records}


def attach_evaluation(env, episodes_per_env):
    from isaaclab_arena import mdp

    metrics = EpisodeMetrics(env.num_envs, episodes_per_env, env.device)
    original = env.reward_manager.compute
    success_params = dict(env.cfg.terminations.success.params, rl_training=False)
    lift_params = env.cfg.rewards.lifting_object.params

    def compute(dt):
        rewards = original(dt)
        # ManagerBasedRLEnv computes rewards after terminations but before resets.
        # Observe the native checks here without modifying rewards or termination.
        metrics.update(mdp.object_is_lifted(env, **lift_params).bool(),
                       mdp.lift_object_rl_success(env, **success_params), env.reset_buf)
        return rewards

    env.reward_manager.compute = compute
    return metrics
