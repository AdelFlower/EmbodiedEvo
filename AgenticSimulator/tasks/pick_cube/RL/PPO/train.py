"""Online recurrent PPO with optional successful-demonstration supervision."""

import time
import traceback
from pathlib import Path

import yaml

from .runtime import (append_jsonl, close_application, episode_rows, parse_args,
                      save_checkpoint, snapshot_environment, write_json, install_interrupt_handlers)
from .curriculum import Curriculum, STAGES


def train(args, config, scene_config, report):
    import torch

    from ..BC.dataset import build_dataloaders
    from .buffer import RolloutBuffer
    from .env import PickCubeEnv
    from .model import ActorCritic
    from .ppo import PPO
    from .validation import evaluate_batch

    torch.manual_seed(config["seed"])
    torch.cuda.set_device(args.device)
    policy = ActorCritic(**config["model"]).to(args.device)
    expert_loader = None
    if args.dataset is not None:
        expert = config["expert"]
        expert_loader, val_loader, stats, split = build_dataloaders(
            args.dataset, batch_size=expert["batch_size"], val_fraction=expert["val_fraction"],
            seed=config["seed"], num_workers=expert["num_workers"])
        policy.set_normalization(stats)
        write_json(args.run_dir / "expert_split.json", split)
        del val_loader
    trainer = PPO(policy, config["ppo"], expert_loader)
    first_update = 0
    curriculum = Curriculum(config["curriculum"], config["curriculum"]["start_level"])
    if args.resume:
        restored = torch.load(args.resume, map_location=args.device, weights_only=True)
        policy.load_state_dict(restored["model"])
        trainer.optimizer.load_state_dict(restored["optimizer"])
        first_update = restored["update"]
        if args.curriculum_level is None:
            curriculum.level = restored["config"]["curriculum"]["start_level"]
            config["curriculum"]["start_level"] = curriculum.level
            if args.curriculum_progress_index is None:
                curriculum.progress_index = restored["config"]["curriculum"].get("start_progress_index", 0)
                config["curriculum"]["start_progress_index"] = curriculum.progress_index
    else:
        restored = None
    curriculum.last_evaluation = first_update
    if args.position_std is not None:
        with torch.no_grad():
            policy.log_std[:3].copy_((args.position_std / policy.pose_scale[:3]).log())
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            state = trainer.optimizer.state.get(policy.log_std, {})
            if key in state:
                state[key][:3].zero_()
    config["position_std_override_m"] = args.position_std
    if args.grip_logit_scale is not None:
        with torch.no_grad():
            for parameter in (policy.actor.weight, policy.actor.bias):
                parameter[6].mul_(args.grip_logit_scale)
                state = trainer.optimizer.state.get(parameter, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state:
                        state[key][6].zero_()
    config["grip_logit_scale_override"] = args.grip_logit_scale
    if args.reset_pose_head:
        with torch.no_grad():
            for parameter in (policy.actor.weight, policy.actor.bias):
                parameter[:6].zero_()
                state = trainer.optimizer.state.get(parameter, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state:
                        state[key][:6].zero_()
    config["reset_pose_head"] = args.reset_pose_head
    # A resumed curriculum level is resolved after reading the checkpoint.
    (args.run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    env = PickCubeEnv(scene_config, config["env"], config["reward"], config["ppo"]["gamma"],
                      device=args.device, seed=config["seed"], headless=args.headless, app=args.app,
                      curriculum=config["curriculum"])
    env.online_supervision = config["ppo"].get("online_expert_weight", 0) > 0
    install_interrupt_handlers()
    settings = config["train"]
    buffer = RolloutBuffer(settings["rollout_steps"], env.num_envs, policy.recurrent_dim, args.device)
    observation = env.observation()
    hidden = policy.initial_hidden(env.num_envs)
    starts = torch.ones(env.num_envs, dtype=torch.bool, device=args.device)
    completed = succeeded = lifted = steps = 0
    if restored:
        steps = restored["steps"]
    started = time.perf_counter()
    report.update(status="running", update=first_update, steps=steps, policy_success_verified=False)
    write_json(args.run_dir / "result.json", report)
    # Save an initial checkpoint too, so interrupted first updates remain inspectable.
    checkpoint_dir = args.run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    save_checkpoint(checkpoint_dir / "initial.pt", policy, trainer.optimizer, config, first_update, steps)
    with (args.run_dir / "episodes.jsonl").open("w") as episodes, (args.run_dir / "metrics.jsonl").open("w") as metrics_file:
        try:
            for update in range(first_update + 1, settings["updates"] + 1):
                if (args.run_dir / "STOP").exists():
                    raise KeyboardInterrupt("Stop requested through the run directory")
                policy.eval()
                new_episodes = []
                for t in range(settings["rollout_steps"]):
                    with torch.no_grad():
                        grip_decisions = policy.grip_decision_mask(env.learner_steps)
                        action, logp, value, next_hidden = policy.act(
                            observation, hidden, starts, grip_decisions=grip_decisions)
                        final_obs, reward, terminated, truncated, info = env.step(action)
                        # Use hidden AFTER obs_t with final obs_{t+1}, BEFORE reset.
                        _, next_value, _ = policy(final_obs, next_hidden)
                    buffer.add(t, observations=observation, actions=action, hidden=hidden[0],
                               logp=logp, values=value, rewards=reward, next_values=next_value,
                               starts=starts, terminated=terminated, truncated=truncated,
                               valid=info["valid"], grip_decisions=grip_decisions,
                               expert_actions=info["expert_actions"], expert_valid=info["expert_valid"])
                    done = terminated | truncated
                    rows = episode_rows(info, done, config["seed"])
                    append_jsonl(episodes, rows)
                    new_episodes.extend(rows)
                    curriculum.observe(rows)
                    completed += len(rows)
                    succeeded += sum(row["success"] for row in rows)
                    lifted += sum(row["seen_lift"] for row in rows)
                    steps += int(info["valid"].sum())
                    hidden = next_hidden
                    # Warmup does not enter the policy's history, including the
                    # last settling step immediately before its first action.
                    starts = done | ~info["valid"]
                    observation = env.reset(done.nonzero().flatten()) if done.any() else final_obs
                    report.update(steps=steps, completed_episodes=completed, succeeded=succeeded,
                                  unfinished_episodes=env.num_envs)
                buffer.finish(config["ppo"]["gamma"], config["ppo"]["gae_lambda"])
                expert = config["expert"]
                fraction = min((update - 1) / max(1, expert["bc_decay_updates"]), 1.0)
                weight = expert["bc_weight"] + fraction * (expert["bc_final_weight"] - expert["bc_weight"])
                if expert_loader is None or expert["bc_weight"] == 0:
                    weight = 0.0
                losses = trainer.update(buffer, weight)
                # Refresh the carried recurrent state after optimization using
                # the observed history, including the original reset boundaries.
                with torch.no_grad():
                    _, _, hidden = policy(buffer.data["observations"].transpose(0, 1),
                                          buffer.data["hidden"][0].unsqueeze(0),
                                          buffer.data["starts"].transpose(0, 1))
                full = [r for r in new_episodes if not r["expert_assisted"]]
                row = dict(update=update, steps=steps, wall_seconds=time.perf_counter() - started,
                           completed_episodes=completed, success_rate=succeeded / max(completed, 1),
                           lift_rate=lifted / max(completed, 1),
                           rollout_completed_episodes=len(new_episodes),
                           rollout_success_rate=(sum(r["success"] for r in new_episodes) / len(new_episodes)
                                                 if new_episodes else None),
                           full_policy_episodes=len(full),
                           full_policy_success_rate=sum(r["success"] for r in full) / len(full) if full else None,
                           curriculum=curriculum.state(), **losses)
                append_jsonl(metrics_file, [row])
                report.update(row)
                write_json(args.run_dir / "result.json", report)
                print(f"[ppo] update={update} stage={STAGES[curriculum.level][0]} steps={steps} "
                      f"handoff={curriculum.state()['handoff_fraction']:.2f} "
                      f"window_success={curriculum.state()['window_success_rate']} "
                      f"assisted_cumulative={row['success_rate']:.3f} "
                      f"optimizer_steps={losses['optimizer_steps']} kl={losses.get('kl', 0):.4f} "
                      f"lr={losses['learning_rate']:.2g}", flush=True)
                verified = False
                if curriculum.ready(update):
                    curriculum.last_evaluation = update
                    # Evaluation deliberately resets worlds; preserve censored
                    # training attempts rather than silently replacing them.
                    with (args.run_dir / "interrupted_episodes.jsonl").open("a") as interrupted:
                        append_jsonl(interrupted, [dict(
                            env_id=i, episode_id=int(env.episode_ids[i]), seed=env.seed,
                            control_steps=max(int(env.steps[i]), 0), success=None,
                            reason="frozen_validation_boundary") for i in range(env.num_envs)])
                    validation = evaluate_batch(
                        env, policy, count=config["curriculum"]["evaluation_episodes"],
                        level=curriculum.level, seed=config["seed"] + 10000,
                        progress_index=curriculum.progress_index,
                        run_dir=args.run_dir, tag=f"update_{update:06d}_{STAGES[curriculum.level][0]}")
                    print(f"[validation] {validation}", flush=True)
                    report["last_validation"] = validation
                    if validation["success_rate"] >= config["curriculum"]["promotion_rate"]:
                        save_checkpoint(checkpoint_dir / f"stage_{curriculum.level}_progress_{curriculum.progress_index}_passed.pt",
                                        policy, trainer.optimizer, config, update, steps)
                        if curriculum.level < len(STAGES) - 1:
                            curriculum.promote()
                        else:
                            final_results = [evaluate_batch(
                                env, policy, count=config["curriculum"]["evaluation_episodes"], level=8,
                                seed=seed, run_dir=args.run_dir, tag=f"update_{update:06d}_full_seed{seed}")
                                for seed in config["curriculum"]["final_seeds"]]
                            verified = all(v["success_rate"] >= config["curriculum"]["final_success_rate"]
                                           for v in final_results)
                            report.update(full_policy_evaluations=final_results, policy_success_verified=verified)
                    env.curriculum_level = curriculum.level
                    env.curriculum_progress_index = curriculum.progress_index
                    observation = env.reset(env.all_ids)
                    hidden = policy.initial_hidden(env.num_envs)
                    starts.fill_(True)
                config["curriculum"]["start_level"] = curriculum.level
                config["curriculum"]["start_progress_index"] = curriculum.progress_index
                save_checkpoint(checkpoint_dir / "latest.pt", policy, trainer.optimizer, config, update, steps)
                if update % settings["checkpoint_interval"] == 0:
                    save_checkpoint(checkpoint_dir / f"update_{update:06d}.pt", policy, trainer.optimizer, config, update, steps)
                    # Small raw traces make policy/teacher frame and timing errors inspectable.
                    import numpy as np
                    diagnostic_dir = args.run_dir / "diagnostics"
                    diagnostic_dir.mkdir(exist_ok=True)
                    np.savez_compressed(diagnostic_dir / f"update_{update:06d}.npz", **{
                        key: buffer.data[key][:, :8].cpu().numpy()
                        for key in ("observations", "actions", "expert_actions", "expert_valid", "valid")})
                if verified:
                    save_checkpoint(checkpoint_dir / "verified_policy.pt", policy, trainer.optimizer, config, update, steps)
                    report["status"] = "policy_verified"
                    break
            else:
                report["status"] = "budget_exhausted"
        finally:
            # Keep incomplete attempts and physical state even on interruption.
            snapshot_environment(env, args.run_dir / "final_state.npz")
            write_json(args.run_dir / "unfinished_episodes.json", [
                dict(env_id=i, episode_id=int(env.episode_ids[i]), seed=config["seed"],
                     control_steps=max(int(env.steps[i]), 0), success=None)
                for i in range(env.num_envs)])


def main():
    args = parse_args("train")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.device = args.device or config["device"]
    config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    for arg, section, key in ((args.num_envs, "env", "num_envs"),
                              (args.fixed_scene, "env", "fixed_scene"),
                              (args.updates, "train", "updates"),
                              (args.curriculum_level, "curriculum", "start_level"),
                              (args.curriculum_progress_index, "curriculum", "start_progress_index"),
                              (args.bc_weight, "expert", "bc_weight")):
        if arg is not None:
            config[section][key] = arg
    if config["expert"]["bc_weight"] > 0 and args.dataset is None:
        raise ValueError("Pass --dataset for expert-assisted training, or --bc-weight 0 for PPO only")
    if config["train"]["rollout_steps"] % config["ppo"]["sequence_length"]:
        raise ValueError("rollout_steps must be divisible by sequence_length")
    scene_text = Path(config["scene_config"]).read_text(encoding="utf-8")
    scene_config = yaml.safe_load(scene_text)
    # Embed the exact scene definition for checkpoint evaluation.
    config["scene"] = scene_config
    config["dataset"] = str(args.dataset.resolve()) if args.dataset else None
    config["resume"] = str(args.resume.resolve()) if args.resume else None
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (args.run_dir / "scene_config.yaml").write_text(scene_text, encoding="utf-8")
    report = dict(status="started", completed_episodes=0, succeeded=0, steps=0)
    write_json(args.run_dir / "result.json", report)
    app, exit_code = None, 1
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(args).app
        install_interrupt_handlers()
        args.app = app
        train(args, config, scene_config, report)
        exit_code = 0
    except KeyboardInterrupt:
        report.update(status="interrupted")
        exit_code = 130
    except Exception:
        report.update(status="error")
        (args.run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
    finally:
        write_json(args.run_dir / "result.json", report)
        close_application(app, exit_code)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
