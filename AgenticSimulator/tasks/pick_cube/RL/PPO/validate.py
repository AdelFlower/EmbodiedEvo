"""Frozen batched evaluation; full policy by default, assisted only if requested."""

import traceback

from .runtime import close_application, install_interrupt_handlers, parse_args, write_json


def main():
    args = parse_args("validate")
    args.device = args.device or "cuda:0"
    args.seed = 43 if args.seed is None else args.seed
    args.run_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status="started", checkpoint=str(args.checkpoint.resolve()),
                  expert_assisted=args.curriculum_level < 8)
    write_json(args.run_dir / "result.json", report)
    app, exit_code = None, 1
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(args).app
        install_interrupt_handlers()
        import torch
        from .env import PickCubeEnv
        from .model import ActorCritic
        from .validation import evaluate_batch
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        config = checkpoint["config"]
        policy = ActorCritic(**config["model"]).to(args.device)
        policy.load_state_dict(checkpoint["model"])
        env = PickCubeEnv(
            config["scene"], dict(config["env"], num_envs=args.num_envs),
            config["reward"], config["ppo"]["gamma"], device=args.device, seed=args.seed,
            headless=args.headless, app=app,
            curriculum=config["curriculum"] if args.curriculum_level < 8 else None)
        report.update(evaluate_batch(env, policy, count=args.num_envs, level=args.curriculum_level,
                                     progress_index=args.curriculum_progress_index,
                                     seed=args.seed, run_dir=args.run_dir, tag="frozen_policy"))
        report["checkpoint_update"] = checkpoint["update"]
        print(report, flush=True)
        exit_code = 0
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        exit_code = 130
    except Exception:
        report["status"] = "error"
        (args.run_dir / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        write_json(args.run_dir / "result.json", report)
        close_application(app, exit_code)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
