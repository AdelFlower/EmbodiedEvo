"""将已有 lift_cube 任务接入源/目标实验执行端。

调用方先启动 Isaac Sim 应用，再 open()、rollout()，最后在 finally 中 close()。
本模块不启动应用、不分配 GPU、不选择候选，也不向 Agent 发布宿主侧证据。

config 示例（路径由执行端解析）：
    {"checkpoint": "/path/to/model.pt", "params_dir": "/path/to/params",
     "device": "cuda:0", "record_video": True}
params_dir 默认是 checkpoint 同级的 params，必须包含原训练的 env.yaml 和
agent.yaml。录制视频需要调用方启动应用时启用 --enable_cameras，并安装
imageio/ffmpeg。输出 overview.mp4 和 wrist.mp4；相机仅录像，不进入 policy 观测。

physics=None 使用恢复的训练物理配置；传入物理实现时约定两个方法：
    configure(env_cfg): 在创建环境前修改物理配置。
    before_step(env, dt): 每个物理子步、动作应用后且写入仿真前调用。
纯参数实现的 before_step 可以为空。这里约定物理实现不持有跨回合隐状态；
不得修改 policy 观测/动作约定、控制周期、任务目标及评价规则。

rollout 接收 {"episode_id": 0, "seed": 42}，一次只运行一个环境、一个回合。
源/目标传入同一回合描述。seed 控制任务采样，实际初态另存入轨迹供对照。
output_dir 应是本次 --run-dir 下新的宿主证据子目录，不能直接作为公开目录。
"""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys


class LiftCubeAdapter:
    def __init__(self, config: dict):
        self.config = dict(config)
        self.checkpoint = Path(config["checkpoint"]).expanduser().resolve()
        self.params_dir = Path(
            config.get("params_dir", self.checkpoint.parent / "params")
        ).expanduser().resolve()
        self.env = None
        self.policy = None
        self.runner = None

    def open(self, physics=None) -> None:
        """恢复任务环境与固定 policy；异常交给执行端记录，不自动重试。"""
        if self.env is not None:
            raise RuntimeError("Adapter is already open")

        # 沿用现有 play.py 的包布局，只导入本任务目录中的实现。
        task_root = str(Path(__file__).resolve().parent / "IsaacLab-Arena")
        if task_root not in sys.path:
            sys.path.insert(0, task_root)

        self.env = self._build_env(physics)
        try:
            self._load_policy()
        except BaseException:
            self.close()
            raise

    def _build_env(self, physics):
        """复用训练环境和配置恢复方式，不另写场景或 rollout 动力学。"""
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_arena.env_cfg import LiftCubeEnvCfg
        from isaaclab_arena.play import load_config, restore_config

        env_cfg = LiftCubeEnvCfg()
        saved = load_config(self.params_dir / "env.yaml")
        if "usd_path" in saved["scene"]["table"]["spawn"]:
            from isaaclab.sim import UsdFileCfg

            env_cfg.scene.table.spawn = UsdFileCfg(
                usd_path=saved["scene"]["table"]["spawn"]["usd_path"]
            )
        restore_config(env_cfg, saved)
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = self.config["device"]
        if physics is not None:
            physics.configure(env_cfg)

        # 关闭目标/当前姿态及末端坐标系的调试箭头，避免进入相机画面。
        env_cfg.commands.object_pose.debug_vis = False
        env_cfg.scene.ee_frame.debug_vis = False

        # 源/目标共用固定录像视角，不继承 checkpoint 或候选的相机位置。
        env_cfg.viewer.eye = (0.0, -0.8, 1.8)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.75)
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.resolution = (640, 480)
        if self.config.get("record_video", True):
            from isaaclab.sensors import CameraCfg
            from isaaclab.sim import PinholeCameraCfg

            env_cfg.scene.wrist_camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
                update_period=0.0,
                width=640,
                height=480,
                data_types=["rgb"],
                spawn=PinholeCameraCfg(
                    focal_length=18.0,
                    horizontal_aperture=24.0,
                    clipping_range=(0.01, 10.0),
                ),
                # 手部局部 +Z 指向夹爪前方；相机偏置到 +X 侧，绕 Y 转 -30°
                # 朝向夹爪中心。此为初始安装值，实际取景尚未通过仿真确认。
                offset=CameraCfg.OffsetCfg(
                    pos=(0.06, 0.0, 0.02),
                    rot=(0.9659258263, 0.0, -0.2588190451, 0.0),
                    convention="ros",
                ),
            )

        env = ManagerBasedRLEnv(
            cfg=env_cfg,
            render_mode="rgb_array" if self.config.get("record_video", True) else None,
        )
        if physics is not None:
            apply_action = env.action_manager.apply_action

            def apply_with_physics():
                apply_action()
                # 原生 step 在此之后 write_data_to_sim，再推进一个物理子步。
                # 因此 set_external_force_and_torque 等写入会在本子步生效。
                physics.before_step(env, env.physics_dt)

            env.action_manager.apply_action = apply_with_physics
        return env

    def _load_policy(self) -> None:
        """沿用 RSL-RL 推理加载，保留训练时的动作裁剪和观测归一化。"""
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_arena.play import load_config, restore_config
        from isaaclab_arena.ppo_cfg import RLPolicyCfg
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg = RLPolicyCfg()
        restore_config(agent_cfg, load_config(self.params_dir / "agent.yaml"))
        agent_cfg.device = self.config["device"]
        self.env = RslRlVecEnvWrapper(self.env, clip_actions=agent_cfg.clip_actions)
        self.runner = OnPolicyRunner(
            self.env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device,
        )
        self.runner.load(str(self.checkpoint), load_optimizer=False)
        self.policy = self.runner.get_inference_policy(device=self.env.unwrapped.device)

    def _state(self) -> dict:
        """宿主侧原始状态；root_state 顺序为位置、WXYZ 姿态、线速度、角速度。"""
        env = self.env.unwrapped
        robot = env.scene["robot"]
        return {
            "robot_root_state": robot.data.root_state_w[0].tolist(),
            "joint_position": robot.data.joint_pos[0].tolist(),
            "joint_velocity": robot.data.joint_vel[0].tolist(),
            "cube_root_state": env.scene["dex_cube"].data.root_state_w[0].tolist(),
            "command": env.command_manager.get_command("object_pose")[0].tolist(),
        }

    def _video_frames(self) -> dict:
        """在同一仿真状态下渲染两路画面，不推进物理。"""
        env = self.env.unwrapped
        env.sim.render()
        wrist = env.scene["wrist_camera"]
        wrist.update(0.0, force_recompute=True)
        return {
            "overview": env.render(recompute=True),
            "wrist": wrist.data.output["rgb"][0, ..., :3].cpu().numpy().copy(),
        }

    def rollout(self, episode: dict, output_dir: str | Path) -> dict:
        """运行至原生终止条件成立；保留失败时已写出的轨迹和视频。"""
        import torch
        from isaaclab_arena.evaluation import attach_evaluation

        if self.policy is None:
            raise RuntimeError("Call open() before rollout()")

        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / "episode.json").write_text(
            json.dumps(episode, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        env = self.env.unwrapped
        env.seed(episode["seed"])
        observations, _ = self.env.reset()
        original_compute = env.reward_manager.compute
        metrics = attach_evaluation(env, episodes_per_env=1)
        evaluated_compute = env.reward_manager.compute
        steps = 0
        termination = []
        artifacts = {"episode": "episode.json", "trajectory": "trajectory.jsonl"}

        try:
            with ExitStack() as stack:
                trajectory = stack.enter_context(
                    (output / "trajectory.jsonl").open("x", encoding="utf-8")
                )

                def record(data):
                    trajectory.write(json.dumps(data, ensure_ascii=False) + "\n")
                    trajectory.flush()

                record({"step": 0, "state": self._state()})
                videos = {}
                if self.config.get("record_video", True):
                    import imageio.v2 as imageio

                    artifacts["videos"] = {}
                    for view in ("overview", "wrist"):
                        filename = f"{view}.mp4"
                        writer = imageio.get_writer(
                            str(output / filename), fps=1 / env.step_dt, codec="libx264",
                        )
                        stack.callback(writer.close)
                        videos[view] = writer
                        artifacts["videos"][view] = filename
                    # 首次调用建立 overview 的 RGB annotator，再统一渲染两路画面。
                    env.render(recompute=True)
                    for view, frame in self._video_frames().items():
                        videos[view].append_data(frame)

                def compute_and_record(dt):
                    nonlocal steps, termination
                    rewards = evaluated_compute(dt)
                    steps += 1
                    termination = [
                        name for name in env.termination_manager.active_terms
                        if env.termination_manager.get_term(name)[0].item()
                    ]
                    # reward.compute 位于自动 reset 之前，保留真正的终止状态。
                    record({
                        "step": steps,
                        "observation": observations[0].tolist(),
                        "policy_action": actions[0].tolist(),
                        "clipped_action": env.action_manager.action[0].tolist(),
                        "reward": rewards[0].item(),
                        "termination": termination,
                        "state": self._state(),
                    })
                    if videos:
                        for view, frame in self._video_frames().items():
                            videos[view].append_data(frame)
                    return rewards

                env.reward_manager.compute = compute_and_record
                with torch.inference_mode():
                    while not metrics.complete:
                        actions = self.policy(observations)
                        observations, _, _, _ = self.env.step(actions)
        finally:
            # 允许后续回合重新挂接评价，避免多次包装累积。
            env.reward_manager.compute = original_compute

        result = {
            "episode_id": episode["episode_id"],
            "seed": episode["seed"],
            "steps": steps,
            "step_dt": env.step_dt,
            "termination": termination,
            "metrics": metrics.summary(),
            "artifacts": artifacts,
        }
        (output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        return result

    def close(self) -> None:
        """关闭任务环境；应用退出和 GPU 槽位释放由调用方处理。"""
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.env = None
            self.policy = None
            self.runner = None
