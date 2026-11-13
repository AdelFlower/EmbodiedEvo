"""GUI/headless closed-loop evaluation using only the trained PPO policy."""

import time
import traceback

import numpy as np

from .runtime import (close_application, episode_rows, parse_args,
                      snapshot_environment, write_json, install_interrupt_handlers)


def save_trace(path, trace):
    np.savez_compressed(
        path, observations=np.asarray([r[0] for r in trace], dtype=np.float32).reshape(-1, 48),
        actions=np.asarray([r[1] for r in trace], dtype=np.float32).reshape(-1, 8),
        simulation_time=np.arange(len(trace), dtype=np.float64) / 50)


def evaluate(args, report, trace):
    import torch

    from .env import CONTROL_HZ, PickCubeEnv
    from .model import ActorCritic

    torch.cuda.set_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    config = checkpoint["config"]
    policy = ActorCritic(**config["model"]).to(args.device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    env_config = dict(config["env"], num_envs=1, fixed_scene=False)
    env = PickCubeEnv(config["scene"], env_config, config["reward"], config["ppo"]["gamma"],
                      device=args.device, seed=args.seed, headless=args.headless,
                      app=args.app, episode_id=args.episode_id)
    hidden = policy.initial_hidden(1)
    obs = env.observation()
    report.update(status="running", checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_update=checkpoint["update"], seed=args.seed, episode_id=args.episode_id,
                  deterministic=True, success=None, control_steps=0)
    write_json(args.run_dir / "config.json", dict(checkpoint_config=config,
               device=args.device, seed=args.seed, episode_id=args.episode_id,
               headless=args.headless, max_seconds=env_config["max_seconds"]))
    write_json(args.run_dir / "result.json", report)
    start = time.perf_counter()
    total_steps = 0
    try:
        while True:
            active = bool(env.active[0])
            if active:
                action, _, _, hidden = policy.act(obs, hidden, deterministic=True,
                    grip_decisions=policy.grip_decision_mask(env.learner_steps))
                trace.append((obs[0].cpu().numpy().copy(), action[0].cpu().numpy().copy()))
            else:
                action = torch.zeros(1, 8, device=args.device)
                action[:, 7] = -1
            obs, _, terminated, truncated, info = env.step(action)
            total_steps += 1
            done = terminated | truncated
            report["control_steps"] = len(trace)
            report["metrics"] = {k: float(v[0]) for k, v in info["metrics"].items()}
            if done[0]:
                report.update(episode_rows(info, done, args.seed)[0])
                report.update(status="complete", wall_seconds=time.perf_counter() - start)
                print(f"[ppo eval] success={report['success']} reason={report['reason']}", flush=True)
                break
            if active and len(trace) % CONTROL_HZ == 0:
                m = report["metrics"]
                print(f"[ppo eval] t={len(trace) / CONTROL_HZ:.1f}s "
                      f"rise={m['cube_rise']:.3f}m placement_xy={m['xy_distance']:.3f}m", flush=True)
                write_json(args.run_dir / "result.json", report)
                save_trace(args.run_dir / "rollout.npz", trace)
            if not args.headless and not args.no_real_time:
                delay = start + total_steps / CONTROL_HZ - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
    finally:
        snapshot_environment(env, args.run_dir / "final_state.npz")
    return env.sim


def main():
    args = parse_args("eval")
    args.device = args.device or "cuda:0"
    args.seed = 43 if args.seed is None else args.seed
    args.run_dir.mkdir(parents=True, exist_ok=False)
    report, trace = dict(status="started", success=None), []
    write_json(args.run_dir / "result.json", report)
    app, exit_code = None, 1
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(args).app
        install_interrupt_handlers()
        args.app = app
        sim = evaluate(args, report, trace)
        save_trace(args.run_dir / "rollout.npz", trace)
        write_json(args.run_dir / "result.json", report)
        exit_code = 0
        if not args.headless and not args.exit_on_done:
            print("[ppo eval] Results saved. Close the window or press Ctrl+C to exit.", flush=True)
            while app.is_running():
                sim.render()
                time.sleep(0.01)
    except KeyboardInterrupt:
        if report.get("status") != "complete":
            report.update(status="interrupted")
            exit_code = 130
    except Exception:
        report.update(status="error")
        (args.run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
    finally:
        save_trace(args.run_dir / "rollout.npz", trace)
        write_json(args.run_dir / "result.json", report)
        close_application(app, exit_code)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
