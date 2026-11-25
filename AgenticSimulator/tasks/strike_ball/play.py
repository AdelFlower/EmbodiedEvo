"""Play the combined strike policy in Isaac, or score a fixed number of full episodes."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

from . import TASK_ID, enter_run_directory


class CameraPanel:
    """Show the live viewport pose in the same frame as the viewer configuration."""

    def __init__(self, origin):
        import omni.kit.app
        import omni.ui as ui

        self.origin = origin
        self.next_update = 0.
        self.window = ui.Window("Camera coordinates (m)", width=480, height=85)
        with self.window.frame:
            self.coordinates = ui.StringField(multiline=True, read_only=True).model
        updates = omni.kit.app.get_app().get_update_event_stream()
        self.subscription = updates.create_subscription_to_pop(self.update)

    def update(self, _event):
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState

        now = time.monotonic()
        if now < self.next_update or not self.window.visible:
            return
        self.next_update = now + 0.2
        viewport = get_active_viewport()
        text = "Waiting for a viewport camera with a focus point..."
        if viewport is not None and viewport.stage is not None:
            camera = ViewportCameraState(viewport=viewport)
            prim = viewport.stage.GetPrimAtPath(viewport.camera_path)
            if prim and prim.GetAttribute("omni:kit:centerOfInterest").Get() is not None:
                lines = []
                for name, point in (("CAMERA_EYE", camera.position_world), ("CAMERA_TARGET", camera.target_world)):
                    xyz = ", ".join(f"{v - offset:.5f}" for v, offset in zip(point, self.origin))
                    lines.append(f"{name} = ({xyz})")
                text = "\n".join(lines)
        if text != self.coordinates.as_string:
            self.coordinates.set_value(text)

    def close(self):
        self.subscription.unsubscribe()
        self.window.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--episodes", type=int, help="Episodes to score (0 plays indefinitely; --video defaults to 1)"
    )
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-camera", action="store_true", help="Show live camera coordinates in the GUI")
    parser.add_argument("--video", action="store_true", help="Write the scene camera RGB frames to replay.mp4")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.episodes is None:
        args.episodes = 1 if args.video else 0
    if args.video:
        args.enable_cameras = True
    if args.num_envs < 1 or args.episodes < 0:
        parser.error("--num_envs must be positive and --episodes must be nonnegative")
    checkpoint = args.checkpoint.expanduser().resolve()
    enter_run_directory(args.run_dir)
    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import torch

        from .env_cfg import StrikeBallEnvCfg
        from .ppo_network import FrozenStrikeNetwork
        from .scene import HOLE_CENTER, SAMPLE_Z

        cfg = StrikeBallEnvCfg()
        cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
        env = gym.make(TASK_ID, cfg=cfg, render_mode="rgb_array" if args.video else None)
        camera_panel = None
        video = None
        try:
            bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if bundle.get("format") != "strike-ball-ppo-network-v1":
                raise ValueError("Expected a combined strike policy checkpoint")
            policy = FrozenStrikeNetwork(bundle, args.device)
            observations, _ = env.reset(seed=args.seed)
            if args.show_camera:
                if not env.unwrapped.sim.has_gui():
                    raise ValueError("--show-camera requires a GUI viewport")
                origin = env.unwrapped.scene.env_origins[cfg.viewer.env_index].tolist()
                camera_panel = CameraPanel(origin)
            obs = observations["policy"]
            real_time = env.unwrapped.sim.has_gui() if args.real_time is None else args.real_time
            count = hits = passes = crossings = 0
            error_sum = 0.
            # Observation 26 is the height of the hole relative to the ball.
            start_height = HOLE_CENTER[2] - obs[:, 26]
            measured_min = float("inf")
            measured_max = float("-inf")
            height_edges = torch.linspace(*SAMPLE_Z, steps=5, device=obs.device)
            band_episodes = torch.zeros(4, device=obs.device, dtype=torch.long)
            band_passes = torch.zeros_like(band_episodes)
            if args.video:
                from imageio_ffmpeg import write_frames
                import isaaclab.sim as sim_utils

                # The default ground mesh has a printed grid even in RGB renders.
                floor_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18), roughness=1.)
                material_path = cfg.scene.ground.prim_path + "/VideoFloorMaterial"
                floor_material.func(material_path, floor_material)
                sim_utils.bind_visual_material(cfg.scene.ground.prim_path, material_path)

                # Attach the RGB annotator and warm the renderer at the reset pose.
                # Only scene pixels from the render product go to the encoder.
                env.render()
                for _ in range(32):
                    env.unwrapped.sim.render()
                    if env.render().any():
                        break
                else:
                    raise RuntimeError("The scene camera did not produce an RGB frame")
                video = write_frames("replay.mp4", cfg.viewer.resolution, fps=1 / env.unwrapped.step_dt,
                                     codec="libx264", macro_block_size=1, output_params=["-movflags", "+faststart"])
                video.send(None)
            with torch.inference_mode():
                while app.is_running() and (not args.episodes or count < args.episodes):
                    start = time.monotonic()
                    if video is not None:
                        # Capture before stepping so the final frame precedes the automatic reset.
                        video.send(env.render().tobytes())
                    observations, _, terminated, truncated, extras = env.step(policy(obs))
                    obs = observations["policy"]
                    if "shot" in extras:
                        shot = extras["shot"]
                        n = len(shot["hit"])
                        if args.episodes:
                            n = min(n, args.episodes - count)
                        finished = (terminated | truncated).nonzero().flatten()
                        heights = start_height[finished[:n]]
                        if args.episodes:
                            measured_min = min(measured_min, heights.min().item())
                            measured_max = max(measured_max, heights.max().item())
                            bands = torch.bucketize(heights, height_edges[1:-1])
                            band_episodes += torch.bincount(bands, minlength=4)
                            band_passes += torch.bincount(bands[shot["passed"][:n]], minlength=4)
                        start_height[finished] = HOLE_CENTER[2] - obs[finished, 26]
                        count += n
                        hits += int(shot["hit"][:n].sum())
                        passes += int(shot["passed"][:n].sum())
                        error = shot["crossing_error"][:n]
                        measured = error[error.isfinite()]
                        crossings += len(measured)
                        error_sum += float(measured.sum())
                        print(f"Episodes {count}, paddle hits {hits}, passes {passes}", flush=True)
                    if real_time:
                        interval = env.unwrapped.step_dt
                        time.sleep(max(0., interval - (time.monotonic() - start)))
            if args.episodes:
                report = {
                    "checkpoint": str(checkpoint), "seed": args.seed, "episodes": count,
                    "hits": hits, "passes": passes, "hit_rate": hits / count if count else None,
                    "pass_rate": passes / count if count else None,
                    "pass_given_hit": passes / hits if hits else None, "crossings": crossings,
                    "configured_height_m": list(SAMPLE_Z),
                    "observed_height_m": [measured_min, measured_max] if count else None,
                    "height_bands": [
                        {"min_m": lo, "max_m": hi, "episodes": total, "passes": successful}
                        for lo, hi, total, successful in zip(height_edges[:-1].tolist(), height_edges[1:].tolist(),
                                                            band_episodes.tolist(), band_passes.tolist())
                    ],
                    "mean_crossing_error_mm": 1000 * error_sum / crossings if crossings else None,
                }
                Path("evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
                if count < args.episodes:
                    raise RuntimeError(f"Playback ended after {count} of {args.episodes} episodes")
        finally:
            if video is not None:
                video.close()
            if camera_panel is not None:
                camera_panel.close()
            env.close()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        return 1
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
