# lift_cube 源环境参考资料

本目录提供源环境的配置与实现副本。启动会话时，宿主将这些资料放入 Agent
工作区的 `reference/` 并只读挂载；可修改的源物理代码位于工作区的 `source/`。

## 文件职责

下表路径均相对于本目录。

| 文件 | 职责 |
| --- | --- |
| `policy/params/env.yaml` | 固定 policy 对应的训练环境配置，包含场景、执行器、观测、动作、重置、奖励、终止条件和仿真参数 |
| `policy/params/agent.yaml` | 保存的 policy 与训练器配置，包括网络结构、观测归一化和动作裁剪 |
| `isaaclab_arena/env_cfg.py` | lift_cube 环境的默认配置，将场景、观测、动作和任务规则组合起来 |
| `isaaclab_arena/robot_cfg.py` | 机器人配置接入、关节动作定义、基础观测项和末端坐标系配置 |
| `isaaclab_arena/mdp.py` | 观测、坐标变换、重置、奖励和成功判定的具体计算 |
| `isaaclab_arena/evaluation.py` | 回合指标的记录与汇总，在自动重置前记录任务结果 |
| `isaaclab_arena/play.py` | YAML 读取、配置恢复，以及独立播放固定 policy 的流程；adapter 复用其中的配置恢复函数 |
| `isaaclab_arena/ppo_cfg.py` | policy 与训练器的默认配置，加载时由保存的 `agent.yaml` 覆盖 |
| `scenes/cube.py` | 场景配置读取、资产生成、物理材质应用、机器人配置及独立预览场景构建 |
| `scenes/configs/cube.yaml` | 场景构建使用的初始描述，包含方块参数、初始位置范围和预览视角 |
| `adapter.py` | 环境与固定 policy 的加载、物理钩子调用、单回合执行、轨迹记录和双视角录像 |

## 配置加载与生效顺序

源 rollout 的环境创建过程如下：

1. 创建 `LiftCubeEnvCfg`，其中场景默认配置使用 `scenes/configs/cube.yaml`。
2. 读取固定 policy 对应的 `params/env.yaml`，恢复保存的训练环境配置。
3. adapter 设置单环境运行和本次使用的设备。
4. 调用当前候选的 `source/physics.py` 中的 `configure(env_cfg)`，修改物理配置。
5. adapter 关闭调试坐标箭头，设置固定录像视角和相机。
6. 使用修改后的配置创建仿真环境。

因此，`cube.yaml` 和训练 `env.yaml` 中的值分别描述场景默认值和恢复配置，
当前源环境还会应用候选的修改。创建时的物理参数以传入环境的最终配置为准。

policy 加载时，先创建 `ppo_cfg.py` 中的默认配置，再恢复保存的 `agent.yaml`，
并加载宿主指定的固定权重。推理沿用恢复的动作裁剪与观测归一化设置。

运行期间，每个物理子步在应用动作后调用候选的 `before_step(env, dt)`，
随后将场景数据写入仿真并推进物理。这里的 `dt` 是物理子步时长，单位为秒，
与一次 policy 控制步的时长不同。候选可以通过这个钩子施加额外物理作用。

## 使用边界

本目录用于阅读，不是候选运行时的代码包。副本保留原实现中的模块名和资源
路径；宿主执行实验时使用实际任务实现，并加载当前源候选。

物理修改及其辅助模块放在 `source/`。候选归档只保存该目录，`reference/`
不会随候选复制到运行目录，因此候选代码不应依赖工作区的参考目录进行导入。

分析脚本、笔记和临时文件放在 `scratch/`。实验命令、反馈格式、预算、物理
接口约定及候选提交方式见工作区根目录的 `tools.md`，任务要求见 `instruction.md`。
