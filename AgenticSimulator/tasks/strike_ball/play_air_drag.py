"""Replay the frozen strike policy with quadratic drag on the ball in still air."""

import argparse
import json
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher

from . import enter_run_directory


def replay(app, args):
    # Isaac/PhysX environment modules require a running Kit application.
    import torch
    from isaaclab.utils.math import quat_rotate_inverse

    from .env import StrikeBallEnv
    from .env_cfg import StrikeBallEnvCfg
    from .ppo_network import FrozenStrikeNetwork
    from .scene import BALL_RADIUS

    drag_factor = 0.5 * args.air_density * args.drag_coefficient * math.pi * BALL_RADIUS**2

    class AirDragStrikeBallEnv(StrikeBallEnv):
        def _apply_support(self):
            velocity_w = self.ball.data.root_lin_vel_w
            speed = velocity_w.norm(dim=-1, keepdim=True)
            drag_w = -drag_factor * speed * velocity_w
            force_w = torch.where(self.hit[:, None], drag_w, self.support[:, 0])
            # Isaac Lab's external wrench buffer uses body coordinates.
            force_b = quat_rotate_inverse(self.ball.data.root_quat_w, force_w)
            self.ball.set_external_force_and_torque(force_b[:, None, :], self.zero_torque)

    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if bundle.get("format") != "strike-ball-ppo-network-v1":
        raise ValueError("Expected a combined strike policy checkpoint")
    policy = FrozenStrikeNetwork(bundle, args.device)

    cfg = StrikeBallEnvCfg()
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
    env = AirDragStrikeBallEnv(cfg=cfg)
    try:
        observations, _ = env.reset(seed=args.seed)
        real_time = env.sim.has_gui()
        count = hits = passes = crossings = 0
        error_sum = 0.0
        with torch.inference_mode():
            while app.is_running() and (args.episodes == 0 or count < args.episodes):
                start = time.monotonic()
                observations, _, _, _, extras = env.step(policy(observations["policy"]))
                if "shot" in extras:
                    shot = extras["shot"]
                    n = len(shot["hit"])
                    if args.episodes:
                        n = min(n, args.episodes - count)
                    hits += int(shot["hit"][:n].sum())
                    passes += int(shot["passed"][:n].sum())
                    errors = shot["crossing_error"][:n]
                    measured = errors[errors.isfinite()]
                    crossings += len(measured)
                    error_sum += float(measured.sum())
                    count += n
                    print(f"Episodes {count}, paddle hits {hits}, passes {passes}", flush=True)
                if real_time:
                    time.sleep(max(0.0, env.step_dt - (time.monotonic() - start)))

        report = {
            "checkpoint": str(args.checkpoint),
            "seed": args.seed,
            "air_density_kg_m3": args.air_density,
            "drag_coefficient": args.drag_coefficient,
            "ball_radius_m": BALL_RADIUS,
            "episodes": count,
            "hits": hits,
            "passes": passes,
            "hit_rate": hits / count if count else None,
            "pass_rate": passes / count if count else None,
            "crossings": crossings,
            "mean_crossing_error_mm": 1000 * error_sum / crossings if crossings else None,
        }
        Path("evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.episodes and count < args.episodes:
            raise RuntimeError(f"Playback ended after {count} of {args.episodes} episodes")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).parent / "policy" / "network.pt")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10, help="Completed shots to score (0 plays indefinitely)")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--air-density", type=float, default=1.225, help="Still air density in kg/m^3")
    parser.add_argument("--drag-coefficient", type=float, default=0.47, help="Dimensionless ball drag coefficient")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.episodes < 0:
        parser.error("--num_envs must be positive and --episodes must be nonnegative")
    for value in (args.air_density, args.drag_coefficient):
        if not math.isfinite(value) or value < 0:
            parser.error("--air-density and --drag-coefficient must be finite and nonnegative")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    enter_run_directory(args.run_dir)
    app = AppLauncher(args).app
    try:
        replay(app, args)
    finally:
        app.close()


if __name__ == "__main__":
    main()
