# strike_ball 源环境参考

用 UR10 机械臂的球拍击中静止的球，使球向下完整穿过板上的圆孔。每次回放运行一个 3 秒回合。物理子步长为 0.005 秒，每 4 个子步执行一次策略动作，即每秒 50 次动作。

## 如何判断一次击球

源侧回放结果的 `metrics` 来自环境在回合结束、自动重置前保存的 `extras["shot"]`。

| 源侧字段 | 含义 |
| --- | --- |
| `hit` | 球拍在本回合至少有一次有效触球。 |
| `contact_substep` | 首次有效触球的物理子步编号。整回合未击中时为 0。 |
| `crossing_error` | 击中后，球心首次向下穿过板中平面时，与孔中心的水平距离，单位为米。没有观测到该穿越时为 `null`。 |
| `passed` | 原生成功标志。球被击中，向下穿过中平面时留有容纳球半径的间隙，随后球体整体越过板的下表面。 |

例如，`hit=true`、`passed=false` 表示已经触球，但本回合未完成穿孔。源侧可用 `artifacts.trajectory` 对照录像：`step=0` 记录初态，后续每行的 `observation` 和 `policy_action` 对应执行动作前的输入与动作，`state` 对应执行动作后的状态。球与球拍位置字段的后缀 `_w` 表示世界坐标系。

源侧和目标侧回放均提供两路同步录像。查看 `artifacts.videos.overview` 观察球的飞行和穿孔过程，查看 `artifacts.videos.wrist` 从随腕部运动的视角观察球与拍面的相对位置。两路录像都从初态开始，每个控制步结束时各记录一帧，末帧在自动重置前采集。目标侧还公开 `steps` 和 `step_dt`。

## 修改源物理

本目录在工作区中只读。编辑工作区的 `source/physics.py`，把候选运行时需要的辅助文件放在 `source/` 中。宿主会单独归档源候选和 `training/` 配方。

- `configure(env_cfg)` 在创建环境**之前**调用。每次从固定 P0 环境基准开始；训练时先执行训练配方的配置函数，再应用源物理。修改物体属性时，可以沿 `env_cfg.py` 中的物体定义找到 `scene.py` 中实际创建它的函数，核对最终写入的值。
- `before_step(env, dt)` 在**每个物理子步**调用。此时原生环境已经应用关节动作，接下来会将场景数据写入仿真并推进 `dt` 秒。源环境在有效触球前托住球，触球后解除支撑。实现逐步作用时，需考虑原生环境写入外力的时机。

## 从回放到续训

1. 用 P0 做一次双侧基准回放，例如 `python scripts/env.py rollout --environments source target --policy-id default --seed 43`。同一请求的两侧使用相同种子。
2. 修改 `source/physics.py` 后，用当前策略和相同种子运行 `python scripts/env.py rollout --environments source --seed 43`。记录响应中的 `results.source.candidate_id`，它标识本次实际运行的源代码副本。
3. 运行 `python scripts/env.py train --candidate-id <本次返回的编号>`。宿主在该候选上从最新策略续训。训练成功后，用 `python scripts/env.py rollout --environments target --seed 43` 在同一初始采样条件下观察新策略，再决定是否进入下一轮。

击球任务的 `network.pt` 同时保存基础挥拍网络和动作修正网络。续训时恢复完整组合策略，固定基础挥拍和先前阶段的修正参数，用 PPO 更新当前阶段的动作分支。训练环境按 32 个一组采样，并行环境数使用 32 的正整数倍。P0 已累计 160 次更新；从 P0 新增 1000 次后，本轮计数为 1000，累计计数为 1160。阶段切换看累计计数。首次从 P0 续训会初始化其未保存的价值网络和优化器状态，之后的训练权重会保存这些状态。

## 修改续训奖励

在工作区 `training/train.py` 的 `configure(env_cfg, agent_cfg)` 中设置 `env_cfg.rewards`。默认配置为空，PPO 使用原有的击球评分和预测模型更新。配置了奖励项后，训练器会为每个并行环境分别累加整拍中每次 `env.step()` 返回的奖励，以这个总分替换默认评分。此时 PPO 只用真实仿真回合更新。训练日志的 `mean_shot_reward` 是这一批回合总分的平均值，`passes` 仍由原生穿孔判定统计。

例如，只奖励回合结束时已完整穿孔的球，可以在训练配方中这样写：

```python
from isaaclab.managers import RewardTermCfg


def passed_at_deadline(env):
    final_step = env.episode_length_buf >= env.max_episode_length
    return (final_step & env.passed).float()


def configure(env_cfg, agent_cfg):
    env_cfg.seed = agent_cfg.seed
    env_cfg.rewards = {"pass_at_deadline": RewardTermCfg(func=passed_at_deadline, weight=1.0)}
```

Isaac Lab 在每个控制步计算奖励时已应用权重和步长，训练器直接累加返回值。

## 文件索引

| 参考文件 | 可查内容 |
| --- | --- |
| [env_cfg.py](env_cfg.py)、[mdp.py](mdp.py) | 场景配置、初态采样、35 项策略观测的排列、六关节动作的缩放和限位。 |
| [scene.py](scene.py)、[env.py](env.py) | 球与球拍的创建、接触检测、物理子步更新和完整穿孔的几何判定。 |
| [adapter.py](adapter.py) | 物理接口的调用顺序、单回合执行、轨迹字段、腕部相机的安装位置及双视角录像。 |
| [ppo_network.py](ppo_network.py)、[ppo_outcome.py](ppo_outcome.py)、[ppo_shot.py](ppo_shot.py) | 基础挥拍与修正量如何组成动作、预测分支如何拟合、PPO 如何评分及恢复训练。 |
| [policy/params/env.yaml](policy/params/env.yaml)、[policy/params/agent.yaml](policy/params/agent.yaml) | P0 环境的时间设置，以及初始续训配方的种子、阶段和采样参数。 |
