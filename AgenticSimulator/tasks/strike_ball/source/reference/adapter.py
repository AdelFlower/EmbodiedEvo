"""把原生击球环境接入宿主的单回合回放和源侧训练。Isaac Sim 由宿主启动。"""

from __future__ import annotations

from argparse import Namespace
from contextlib import ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path

from AgenticSimulator.orchestrator.files import write_json


class StrikeBallAdapter:
    """env.yaml 用普通 YAML 字段覆盖 StrikeBallEnvCfg，agent.yaml 保存击球训练器的平铺参数。"""

    def __init__(self, config: dict):
        self.device = config["device"]
        self.checkpoint = Path(config["checkpoint"]).resolve()
        self.params_dir = Path(config.get("params_dir", self.checkpoint.parent / "params"))
        self.env_config_path = config.get("env_config_path")
        self.record_video = config.get("record_video", True)
        self.env = None
        self.policy = None

    def _env_config(self, num_envs):
        from .env_cfg import ASSET_DIR, StrikeBallEnvCfg

        if self.env is not None:
            raise RuntimeError("Adapter is already open")
        cfg = StrikeBallEnvCfg()
        # 宿主总会指定固定 P0 基准；单独使用适配器时也支持原生任务默认值。
        if self.env_config_path is not None:
            import yaml

            overrides = yaml.safe_load(Path(self.env_config_path).read_text(encoding="utf-8"))
            cfg.seed = overrides.pop("seed", cfg.seed)
            cfg.from_dict(overrides)
            for spawn in (cfg.scene.ball.spawn, cfg.scene.robot.spawn):
                if not Path(spawn.usd_path).is_absolute() and "://" not in spawn.usd_path:
                    spawn.usd_path = str(ASSET_DIR / spawn.usd_path)
        cfg.scene.num_envs = num_envs
        cfg.sim.device = self.device
        return cfg

    def _build_env(self, cfg, physics, *, record_video):
        from .env import StrikeBallEnv

        # 两侧录像沿用相同的基准视角。
        viewer = deepcopy(cfg.viewer)
        if physics is not None:
            physics.configure(cfg)
        cfg.viewer = viewer
        if record_video:
            from isaaclab.sensors import CameraCfg
            from isaaclab.sim import PinholeCameraCfg

            cfg.scene.wrist_camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/UR10_instanceable/wrist_3_link/WristCamera",
                update_period=0.0, width=640, height=480, data_types=["rgb"],
                spawn=PinholeCameraCfg(
                    focal_length=18.0, horizontal_aperture=24.0, clipping_range=(0.01, 10.0),
                ),
                # 镜头沿拍柄前移至腕关节前方，从击球侧斜看拍面中心，拍柄朝画面下方。
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.146, -0.17), rot=(0.0, 0.0, 0.30070580, 0.95371695), convention="ros",
                ),
            )
        env = StrikeBallEnv(cfg=cfg, render_mode="rgb_array" if record_video else None)
        self.env = env
        if physics is not None:
            apply_action = env.action_manager.apply_action

            def apply_with_physics():
                apply_action()
                # 下一步由原生环境把场景数据写入仿真，再推进一个物理子步。
                physics.before_step(env, env.physics_dt)

            env.action_manager.apply_action = apply_with_physics
        return env

    def open(self, physics=None):
        import torch
        from .ppo_network import FrozenStrikeNetwork

        cfg = self._env_config(num_envs=1)
        bundle = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        if bundle.get("format") != "strike-ball-ppo-network-v1":
            raise ValueError("Expected a combined strike policy checkpoint")
        policy = FrozenStrikeNetwork(bundle, self.device)
        self._build_env(cfg, physics, record_video=self.record_video)
        self.policy = policy

    def open_training(self, physics=None, *, num_envs: int, configure=None):
        """训练配方先配置奖励和算法，随后在同一份配置上应用本轮源物理。"""
        import yaml

        cfg = self._env_config(num_envs)
        # 击球训练器按属性读取参数，与独立训练脚本的 argparse 用法相同。
        params_file = self.params_dir / "agent.yaml"
        agent_cfg = Namespace(**yaml.safe_load(params_file.read_text(encoding="utf-8")))
        agent_cfg.device = self.device
        if configure is not None:
            configure(cfg, agent_cfg)
        env = self._build_env(cfg, physics, record_video=False)
        return env, agent_cfg

    def _state(self):
        env = self.env
        return {
            "ball_position_w": env.ball.data.root_pos_w[0].tolist(),
            "ball_velocity_w": env.ball.data.root_lin_vel_w[0].tolist(),
            "paddle_center_w": env.paddle_center()[0].tolist(),
            "joint_position": env.robot.data.joint_pos[0].tolist(),
            "hit": env.hit[0].item(),
            "passed": env.passed[0].item(),
        }

    def _video_frames(self):
        env = self.env
        env.sim.render()
        wrist = env.scene["wrist_camera"]
        wrist.update(0.0, force_recompute=True)
        return {
            "overview": env.render(recompute=True),
            "wrist": wrist.data.output["rgb"][0, ..., :3].cpu().numpy(),
        }

    def _open_videos(self, output, stack):
        from imageio_ffmpeg import write_frames

        env = self.env
        # 首次 render 建立 RGB 采集器。在初态等待首帧时只渲染，不推进物理。
        env.render()
        for _ in range(32):
            frames = self._video_frames()
            if all(frame.any() for frame in frames.values()):
                break
        else:
            missing = ", ".join(name for name, frame in frames.items() if not frame.any())
            raise RuntimeError(f"No initial RGB frame from camera(s): {missing}")
        videos = {}
        for name, frame in frames.items():
            height, width = frame.shape[:2]
            video = write_frames(
                str(output / f"{name}.mp4"), (width, height), fps=1 / env.step_dt,
                codec="libx264", macro_block_size=1, output_params=["-movflags", "+faststart"],
            )
            stack.callback(video.close)
            video.send(None)
            video.send(frame.tobytes())
            videos[name] = video
        return videos

    def rollout(self, episode: dict, output_dir: str | Path) -> dict:
        """用请求种子跑一个完整回合，输出重置前的轨迹、录像和原生穿孔结果。"""
        import torch

        if self.policy is None:
            raise RuntimeError("Call open() before rollout()")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "episode.json", episode)
        artifacts = {"episode": "episode.json", "trajectory": "trajectory.jsonl"}
        env = self.env
        observations, _ = env.reset(seed=episode["seed"])

        with ExitStack() as stack, torch.inference_mode():
            trajectory = stack.enter_context((output / "trajectory.jsonl").open("x", encoding="utf-8"))

            def record(sample):
                trajectory.write(json.dumps(sample, allow_nan=False) + "\n")
                trajectory.flush()

            record({"step": 0, "state": self._state()})
            videos = {}
            if self.record_video:
                videos = self._open_videos(output, stack)
                artifacts["videos"] = {name: f"{name}.mp4" for name in videos}

            compute_reward = env.reward_manager.compute

            def record_before_reset(dt):
                rewards = compute_reward(dt)
                # 此时动作已经执行，原生环境还没有清除最后一步的击球状态。
                record({
                    "step": step,
                    "observation": obs[0].tolist(),
                    "policy_action": action[0].tolist(),
                    "clipped_action": env.action_manager.action[0].tolist(),
                    "reward": rewards[0].item(),
                    "state": self._state(),
                })
                if videos:
                    for name, frame in self._video_frames().items():
                        videos[name].send(frame.tobytes())
                return rewards

            env.reward_manager.compute = record_before_reset
            stack.callback(setattr, env.reward_manager, "compute", compute_reward)
            for step in range(1, env.max_episode_length + 1):
                obs = observations["policy"]
                action = self.policy(obs)
                observations, _, terminated, truncated, extras = env.step(action)
                if (terminated | truncated)[0].item():
                    shot = extras["shot"]
                    break
            else:
                raise RuntimeError("Strike episode reached its horizon without a native termination")

        error = shot["crossing_error"][0].item()
        result = {
            "episode_id": episode["episode_id"],
            "seed": episode["seed"],
            "steps": step,
            "step_dt": env.step_dt,
            "metrics": {
                "hit": shot["hit"][0].item(),
                "passed": shot["passed"][0].item(),
                # 原生 +inf 表示本回合没有观测到穿越，JSON 使用 null。
                "crossing_error": None if error == math.inf else error,
                "contact_substep": shot["contact_substep"][0].item(),
            },
            "artifacts": artifacts,
        }
        write_json(output / "result.json", result)
        return result

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.env = None
            self.policy = None
