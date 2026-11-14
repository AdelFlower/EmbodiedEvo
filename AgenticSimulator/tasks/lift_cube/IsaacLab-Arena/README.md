# Franka 抬方块 PPO 训练（Isaac Sim 4.5）

此任务使用本地 `isaac-sim` Conda 环境中的 Isaac Sim 4.5、Isaac Lab 和
RSL-RL 2.3.1，直接构建 `ManagerBasedRLEnv`，不依赖 Arena 通用框架。
不需要升级 Isaac Sim，也不需要修改已安装的 Isaac Lab 源码。

默认场景共用 `AgenticSimulator/scenes/plate_cube.py` 的资产构建函数，
读取 `AgenticSimulator/scenes/configs/plate_cube.yaml`：使用本地高 PD Franka、
本地方块，以及相同的质量、重力、阻尼、摩擦、缩放和相机配置。
PPO 只读取 YAML 中的 Cube 条目，不创建盘子；独立 `plate_cube.py` 场景仍保留盘子。
桌面为 1.2 × 0.8 m，高 0.75 m；机器人底座位于 `(-0.45, 0, 0.75)`。
方块在每个并行环境、每次重置时独立采样位置和 yaw。
任务为抬方块到采样目标。

## 启动

从 EmbodiedEvo 仓库根目录运行，先用少量环境验证：

```bash
conda activate isaac-sim
RUN_DIR="$PWD/runs/$(date +%Y%m%d%H%M)_lift_cube_smoke"
python AgenticSimulator/tasks/lift_cube/IsaacLab-Arena/train.py \
  --run-dir "$RUN_DIR" \
  --headless --device cuda:0 \
  --num_envs 16 --max_iterations 2
```

短跑正常后，使用新的运行目录启动完整训练：

```bash
RUN_DIR="$PWD/runs/$(date +%Y%m%d%H%M)_lift_cube_ppo"
python AgenticSimulator/tasks/lift_cube/IsaacLab-Arena/train.py \
  --run-dir "$RUN_DIR" \
  --headless --device cuda:0 \
  --num_envs 4096 --max_iterations 2000
```

每次使用新的 `YYYYMMDDHHMM` 前缀目录；已有目录不能覆盖，失败记录会保留。
启动脚本自动提供本任务的 Python 导入路径，无需额外 `pip install -e .`。
应完整激活 Conda 环境，以加载其已有的 Isaac Sim 动态库设置。

训练进程原始输出在 `$RUN_DIR/training.log`，可另开终端使用 `tail -f` 查看。
模型、TensorBoard 日志及参数在 `$RUN_DIR/logs/rsl_rl/generic_experiment/` 下。
仅在训练循环完成后写入 `$RUN_DIR/training_complete.json`；无完成记录视为失败。

可使用 `agent.algorithm.learning_rate=0.0001` 等 Hydra 参数调整 PPO 配置。
`--checkpoint /absolute/path/model_1999.pt` 从已有模型恢复到新的运行目录。
去掉 `--headless` 可打开图形窗口。

## 可视化已训练模型

`play.py` 只加载 checkpoint 执行策略，不继续训练。读取 checkpoint 同级
`params/env.yaml` 和 `params/agent.yaml`，默认显示一个环境并按真实时间运行。
从仓库根目录执行（将 `CHECKPOINT` 替换为实际文件）：

```bash
conda activate isaac-sim
CHECKPOINT="runs/<训练目录>/logs/rsl_rl/generic_experiment/<时间目录>/model_1600.pt"
python AgenticSimulator/tasks/lift_cube/IsaacLab-Arena/play.py \
  --checkpoint "$CHECKPOINT" \
  --run-dir "runs/$(date +%Y%m%d%H%M)_lift_cube_play" \
  --device cuda:0 --num_envs 1
```

不加 `--headless` 会打开 Isaac Sim 窗口；关闭窗口结束。
方块和机器人每回合自动重置，目标位置继续按训练配置采样。
`--steps 500` 限制播放步数，`--no-real-time` 取消真实时间限速。
无桌面环境时可以加 `--headless --steps 50` 验证加载和推理，但不会显示窗口。
日志位于播放运行目录中的 `play.log`，正常完成后写入 `playback_complete.json`。

## 文件

| 文件 | 用途 |
| --- | --- |
| `train.py` | 检查运行目录、保存原始日志并启动训练 |
| `play.py`、`isaaclab_arena/play.py` | 加载模型和训练配置，显示策略执行 |
| `isaaclab_arena/train.py` | 启动本地 Isaac Sim、配置任务并调用 RSL-RL |
| `isaaclab_arena/env_cfg.py` | 场景、观测、奖励、目标和终止配置 |
| `isaaclab_arena/robot_cfg.py` | Franka 动作、重置与末端传感器 |
| `isaaclab_arena/mdp.py` | 观测计算、任务奖励、成功检查与方块重置 |
| `isaaclab_arena/ppo_cfg.py` | PPO 网络与超参数 |
| `AgenticSimulator/scenes/plate_cube.py` | 场景与 PPO 共用的机器人、物体资产和物理配置构建 |

## 任务行为与本地适配

保留关节位置动作（scale 0.5）、50 Hz 控制（仿真 dt 1/200、decimation 4）、
5 秒回合。机器人使用 `plate_cube.py` 的高 PD 参数及默认关节姿态，
不再合成旧场景的机器人底座。训练时不添加额外的关节初始噪声。
奖励权重仍为 reaching 1、lifting 15、goal tracking 16/5、动作/关节速度惩罚 -0.0001。

方块根位置采样范围为世界/环境坐标 x 0–0.3、y -0.15–0.15、z 0.78 m，
yaw 0–360°，质量 0.35 kg，静态和动态摩擦系数均为 0.8。
目标按方块配置的名义初始位置加偏移生成，再转换到机器人基座坐标：
x 0.5–0.7、y -0.25–0.25、z 0.23–0.43 m，
对应环境坐标 z 0.98–1.18 m，即比方块初始根位置高 0.2–0.4 m。
抬升奖励阈值为桌面高度 + 0.04 = 0.79 m，掉落阈值为桌面高度 - 0.05 = 0.70 m。
训练默认关闭成功提前终止；成功检查为目标位置误差小于 0.05 m。

改变场景后请使用新的运行目录从头训练；旧模型尚未在新布局上训练。
`play.py` 继续读取 checkpoint 自带的训练配置，旧模型按旧布局播放。
新训练会在 `params/env.yaml` 的 `scene_config` 字段中保存当次 YAML 配置快照。

Isaac Sim 4.5 的原生姿态使用 WXYZ 四元数。物理配置与坐标变换使用该顺序，
策略中的目标和末端四元数观测显式转回原来的 XYZW。
RSL-RL 2.3 只接收单个 policy 张量，因此将原 `[policy, task_obs]` 按原顺序拼接，
保留全部 45 维输入，actor 与 critic 使用相同输入。
旧的 Isaac Lab 3 Warp 缓冲区及 RTX 兼容代码已移除。

## PPO 数值稳定性与验证

默认启用运行统计观测归一化，使用 `noise_std_type=log` 和固定学习率 `3e-4`。
旧的自适应学习率可能升到 `1e-2`，配置中的初始学习率不是其上限。
送入环境的动作裁剪到 `[-10, 10]`，在原 scale 0.5 和 offset 下仍覆盖 Franka
全部原生关节范围，同时限制 `last_action` 反馈及动作差平方惩罚的异常放大。
任务目标、奖励权重、5 秒动作预算、物体物理参数和成功阈值保持不变。

训练检查 actor/critic 输入输出及优化器更新前的梯度；发现 NaN/Inf 时停止，
在模型目录保留 `numerics_failure.json` 和 `numerics_failure.pt`，后者包括异常张量、
模型、优化器及 rollout。不会用零替换异常数据，也不会自动重试失败训练。
新配置应从头训练；旧 scalar 标准差 checkpoint 不能直接加载进 log 标准差网络。
`play.py` 仍使用 checkpoint 保存的配置，因此可以播放旧模型。

独立评估使用确定性策略和新的随机种子，每个环境运行相同数量的完整回合：

```bash
python AgenticSimulator/tasks/lift_cube/IsaacLab-Arena/play.py \
  --checkpoint "$CHECKPOINT" \
  --run-dir "runs/$(date +%Y%m%d%H%M)_lift_cube_eval" \
  --headless --device cuda:0 --num_envs 64 \
  --episodes-per-env 4 --seed 43 --no-real-time
```

`evaluation.json` 保留每个回合的结果，以及原生抬升阈值和目标距离阈值的命中率。
统计在物理步之后、自动重置之前进行，包括回合最后一步。训练期间
`Episode_Termination/success` 为零是因为关闭了成功提前终止，不能当作评估成功率。

物理抓取诊断可通过 `isaaclab_arena.check_grasp` 运行；这是 IK 控制器诊断，
不是 PPO 策略成绩。它使用原任务动作接口和回合长度，保存逐步轨迹及结果：

```bash
PYTHONPATH="$PWD/AgenticSimulator/tasks/lift_cube/IsaacLab-Arena:$PWD:${PYTHONPATH:-}" \
python -m isaaclab_arena.check_grasp \
  --run-dir "runs/$(date +%Y%m%d%H%M)_lift_cube_grasp_check" \
  --headless --device cuda:0 --num_envs 16
```

Arena 上游提交：`2c778fda27c54af687a7e765ee6e3db1294595f1`。
来源：https://github.com/isaac-sim/IsaacLab-Arena 。本地适配不再要求上游 Isaac Lab 3.0 固定提交。
