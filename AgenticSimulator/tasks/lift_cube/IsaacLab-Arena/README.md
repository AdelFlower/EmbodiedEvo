# Franka 抬方块 PPO 训练（Isaac Sim 4.5）

此任务使用本地 `isaac-sim` Conda 环境中的 Isaac Sim 4.5、Isaac Lab 和
RSL-RL 2.3.1，直接构建 `ManagerBasedRLEnv`，不依赖 Arena 通用框架。
不需要升级 Isaac Sim，也不需要修改已安装的 Isaac Lab 源码。

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
  --num_envs 1024 --max_iterations 2000
```

每次使用新的 `YYYYMMDDHHMM` 前缀目录；已有目录不能覆盖，失败记录会保留。
启动脚本自动提供本任务的 Python 导入路径，无需额外 `pip install -e .`。
应完整激活 Conda 环境，以加载其已有的 Isaac Sim 动态库设置。

训练进程原始输出在 `$RUN_DIR/training.log`，可另开终端使用 `tail -f` 查看。
模型、TensorBoard 日志及参数在 `$RUN_DIR/logs/rsl_rl/generic_experiment/` 下。
仅在训练循环完成后写入 `$RUN_DIR/training_complete.json`；无完成记录视为失败。

可使用 `agent.algorithm.learning_rate=0.001` 等 Hydra 参数调整 PPO 配置。
`--checkpoint /absolute/path/model_1999.pt` 从已有模型恢复到新的运行目录。
去掉 `--headless` 可打开图形窗口。

## 文件

| 文件 | 用途 |
| --- | --- |
| `train.py` | 检查运行目录、保存原始日志并启动训练 |
| `isaaclab_arena/train.py` | 启动本地 Isaac Sim、配置任务并调用 RSL-RL |
| `isaaclab_arena/env_cfg.py` | 场景、观测、奖励、目标和终止配置 |
| `isaaclab_arena/robot_cfg.py` | Franka 动作、重置与末端传感器 |
| `isaaclab_arena/mdp.py` | 观测计算、任务奖励、成功检查与方块重置 |
| `isaaclab_arena/ppo_cfg.py` | PPO 网络与超参数 |
| `isaaclab_arena/stand.py` | Franka 底座 USD 合成和对齐 |

## 任务行为与本地适配

保留关节位置动作（scale 0.5）、Franka ready pose 与底座、DexCube 尺寸、
50 Hz 控制（仿真 dt 1/200、decimation 4）、5 秒回合与 PPO 超参数。
奖励为 reaching 1、lifting 15、goal tracking 16/5，以及动作/关节速度惩罚 -0.0001。
目标范围为 x 0.4–0.6、y -0.25–0.25、z 0.255–0.455 m。
抬升奖励沿用原有绝对高度阈值 0.04 m，掉落阈值 -0.05 m。
训练默认关闭成功提前终止；成功检查为目标位置误差小于 0.05 m。

Isaac Sim 4.5 的原生姿态使用 WXYZ 四元数。物理配置与坐标变换使用该顺序，
策略中的目标和末端四元数观测显式转回原来的 XYZW。
RSL-RL 2.3 只接收单个 policy 张量，因此将原 `[policy, task_obs]` 按原顺序拼接，
保留全部 45 维输入，actor 与 critic 使用相同输入，不做观测归一化。
旧的 Isaac Lab 3 Warp 缓冲区及 RTX 兼容代码已移除。

Arena 上游提交：`2c778fda27c54af687a7e765ee6e3db1294595f1`。
来源：https://github.com/isaac-sim/IsaacLab-Arena 。本地适配不再要求上游 Isaac Lab 3.0 固定提交。
