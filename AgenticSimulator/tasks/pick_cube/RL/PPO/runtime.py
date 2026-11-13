"""Small shared CLI/artifact helpers; no simulator imports at module import."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def parse_args(mode, argv=None):
    parser = argparse.ArgumentParser(description=f"Pick-cube recurrent PPO {mode}")
    if mode == "train":
        parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
        parser.add_argument("--dataset", type=Path)
        parser.add_argument("--updates", type=int)
        parser.add_argument("--num-envs", type=int)
        parser.add_argument("--bc-weight", type=float)
        parser.add_argument("--position-std", type=float, help="Override translation exploration std in meters, including resume")
        parser.add_argument("--grip-logit-scale", type=float, help="Rescale the grip head on startup; positive factors preserve deterministic choices")
        parser.add_argument("--reset-pose-head", action="store_true", help="Reset the six pose outputs when changing their parameterization")
        parser.add_argument("--feedback-expert", action="store_true", help="check_env only: execute the physical feedback training oracle")
        parser.add_argument("--fixed-scene", action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument("--resume", type=Path, help="Continue a checkpoint into a NEW run directory")
        parser.add_argument("--curriculum-level", type=int, help="Override initial handoff level (0..8; 8 = full policy)")
        parser.add_argument("--curriculum-progress-index", type=int, help="Index into handoff_fractions within a stage")
    else:
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--episode-id", type=int, default=0)
        parser.add_argument("--exit-on-done", action="store_true")
        parser.add_argument("--no-real-time", action="store_true")
        if mode == "validate":
            parser.add_argument("--num-envs", type=int, default=64)
            parser.add_argument("--curriculum-level", type=int, default=8)
            parser.add_argument("--curriculum-progress-index", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gui", action="store_true")
    # --help works even on a CPU host without Isaac Lab installed.
    import sys
    if {"--help", "-h"}.intersection(sys.argv[1:] if argv is None else argv):
        parser.print_help()
        print("Isaac Lab also supplies --headless, --device and other AppLauncher options.")
        parser.exit()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=mode != "eval", device=None)
    args = parser.parse_args(argv)
    if args.gui:
        args.headless = False
    path = args.run_dir.expanduser()
    if path.parent == Path("."):
        path = ROOT / ("runs" if mode == "train" else "evals") / path
    args.run_dir = path.resolve()
    try:
        prefix = args.run_dir.name[:12]
        if len(prefix) != 12 or not prefix.isdigit():
            raise ValueError
        datetime.strptime(prefix, "%Y%m%d%H%M")
    except ValueError:
        parser.error("--run-dir must start with a valid YYYYMMDDHHMM timestamp")
    return args


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def save_checkpoint(path, policy, optimizer, config, update, steps):
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save(dict(model=policy.state_dict(), optimizer=optimizer.state_dict(),
                    config=config, update=update, steps=steps), tmp)
    tmp.replace(path)


def append_jsonl(handle, rows):
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def episode_rows(info, done, seed):
    rows = []
    for index in done.nonzero().flatten().tolist():
        success = bool(info["success"][index])
        rows.append(dict(
            env_id=index, seed=seed, episode_id=int(info["episode_ids"][index]),
            success=success, seen_lift=bool(info["seen_lift"][index]),
            reason="success" if success else ("guide_failed" if info["guide_failed"][index] else
                   ("invalid_ik" if info["invalid_ik"][index] else "rollout_timeout")),
            curriculum_level=int(info["curriculum_levels"][index]),
            curriculum_progress_index=int(info["curriculum_progress_indices"][index]),
            handoff_fraction=float(info["handoff_fractions"][index]),
            learner_steps=int(info["learner_steps"][index]),
            expert_assisted=bool(info["curriculum_levels"][index] < 8),
            control_steps=int(info["steps"][index]), episode_return=float(info["episode_return"][index]),
            metrics={k: float(v[index]) for k, v in info["metrics"].items()}))
    return rows


def snapshot_environment(env, path):
    import numpy as np

    np.savez_compressed(path, observations=env.observation().cpu().numpy(),
                        episode_ids=env.episode_ids.cpu().numpy(), steps=env.steps.cpu().numpy(),
                        joint_positions=env.scene["robot"].data.joint_pos.cpu().numpy(),
                        cube_state=env.scene["cube"].data.root_state_w.cpu().numpy(),
                        plate_state=env.scene["plate"].data.root_state_w.cpu().numpy())


def close_application(app, exit_code):
    # Same process-exit policy as BC: artifacts are flushed before reaching here.
    # Kit shutdown has previously blocked on this installation.
    if app is not None:
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


def install_interrupt_handlers():
    # Kit's own Ctrl-C handler can exit before Python saves unfinished attempts.
    import signal
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
