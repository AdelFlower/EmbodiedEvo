"""Run the scripted expert through the exact PPO environment as a baseline."""

import traceback
from pathlib import Path

import yaml

from .runtime import close_application, parse_args, write_json, install_interrupt_handlers


def main():
    args = parse_args("train")
    config = yaml.safe_load(args.config.read_text())
    config["env"]["num_envs"] = args.num_envs or 16
    args.device = args.device or config["device"]
    seed = config["seed"] if args.seed is None else args.seed
    args.run_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status="started", diagnostic="expert_baseline", policy_success_verified=False)
    write_json(args.run_dir / "result.json", report)
    (args.run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    scene = yaml.safe_load(Path(config["scene_config"]).read_text())
    write_json(args.run_dir / "scene_config.json", scene)
    app, exit_code = None, 1
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(args).app
        install_interrupt_handlers()
        from .env import PickCubeEnv
        from .model import ActorCritic
        from .validation import evaluate_batch
        env = PickCubeEnv(scene, config["env"], config["reward"], config["ppo"]["gamma"],
                          device=args.device, seed=seed, headless=args.headless, app=app,
                          curriculum=config["curriculum"])
        policy = ActorCritic(**config["model"]).to(args.device)
        env.feedback_only = env.online_supervision = args.feedback_expert
        report["diagnostic"] = "feedback_expert_baseline" if args.feedback_expert else "expert_baseline"
        result = evaluate_batch(env, policy, count=env.num_envs, level=8 if args.feedback_expert else 0, seed=seed,
                                run_dir=args.run_dir, tag="expert", expert_only=True)
        report.update(status="complete", expert_baseline=result)
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
