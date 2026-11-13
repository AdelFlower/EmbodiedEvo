"""Frozen-policy validation, with explicit reporting of expert assistance."""

import numpy as np
import torch

from .curriculum import STAGES, fractions
from .runtime import append_jsonl, episode_rows, write_json


@torch.no_grad()
def evaluate_batch(env, policy, *, count, level, seed, run_dir, tag, expert_only=False, progress_index=0):
    """Uses the existing simulator; caller records interrupted training attempts.

    At level=8 no guide action is executed. Each counted world runs one episode;
    completed worlds pause instead of being repeatedly sampled until success.
    """
    count = min(count, env.num_envs)
    old = (env.seed, env.next_episode_id, env.curriculum_level, env.config["fixed_scene"],
           env.curriculum.get("retention_fraction", 0) if env.curriculum else 0,
           env.curriculum_progress_index)
    env.seed, env.next_episode_id, env.curriculum_level = seed, 0, level
    env.config["fixed_scene"] = False
    if env.curriculum:
        env.curriculum["retention_fraction"] = 0.0
    env.expert_only = expert_only
    env.curriculum_progress_index = progress_index
    directory = run_dir / "validations" / tag
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    trace = []
    trace_count = min(8, count)
    report = dict(status="running", level=level, stage=STAGES[level][0],
                  progress_index=progress_index,
                  handoff_fraction=fractions(env.curriculum or {}, level)[progress_index],
                  seed=seed, requested_episodes=count, expert_only=expert_only,
                  expert_assisted=level < 8 or expert_only)
    write_json(directory / "result.json", report)
    try:
        observation = env.reset(env.all_ids)
        env.paused[count:] = True
        hidden = policy.initial_hidden(env.num_envs)
        starts = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        policy.eval()
        with (directory / "episodes.jsonl").open("w") as output:
            for _ in range(env.max_steps + env.warmup_steps + 1):
                action, _, _, hidden = policy.act(
                    observation, hidden, starts, deterministic=True,
                    grip_decisions=policy.grip_decision_mask(env.learner_steps))
                before = observation[:trace_count].cpu().numpy().copy()
                active = env.active[:trace_count].cpu().numpy().copy()
                observation, _, terminated, truncated, info = env.step(action)
                trace.append((before, info["executed_actions"][:trace_count].cpu().numpy().copy(),
                              active, info["valid"][:trace_count].cpu().numpy().copy()))
                done = terminated | truncated
                ended = episode_rows(info, done, seed)
                for row in ended:
                    row["expert_only"] = expert_only
                    row["expert_assisted"] |= expert_only
                rows.extend(ended)
                append_jsonl(output, ended)
                env.paused |= done
                starts = done | ~info["valid"]
                if len(rows) == count:
                    break
        report.update(status="complete", completed_episodes=len(rows),
                      successes=sum(r["success"] for r in rows),
                      success_rate=sum(r["success"] for r in rows) / count,
                      lift_rate=sum(r["seen_lift"] for r in rows) / count)
        return report
    finally:
        if trace:
            np.savez_compressed(directory / "rollout.npz",
                                observations=np.stack([r[0] for r in trace]),
                                actions=np.stack([r[1] for r in trace]),
                                active=np.stack([r[2] for r in trace]),
                                policy_controlled=np.stack([r[3] for r in trace]))
        if report["status"] != "complete":
            report.update(status="interrupted_or_error", completed_episodes=len(rows))
        write_json(directory / "result.json", report)
        (env.seed, env.next_episode_id, env.curriculum_level, env.config["fixed_scene"],
         retention, env.curriculum_progress_index) = old
        if env.curriculum:
            env.curriculum["retention_fraction"] = retention
        env.expert_only = False
        env.paused.zero_()
