# Pick-cube BC 采集与训练

采集的 `expert.py` 基于 `batch_controller.py`，保留其专家阶段和成功判据，
并在 BC 目录内拆分决策与跟踪。原始控制器和运行入口保持不变。物理与低层跟踪为
200 Hz，专家每 20 ms 输出一次动作（50 Hz），闭合和搬运力度固定为每指
10 N。每次动作相对于调用时的实际 `panda_hand` 位姿生成固定目标，四个
物理子步跟踪同一目标，不重复累加增量。

## 采集

在同一个 Isaac Lab 环境安装额外存储依赖（不更换该环境的 torch）：

```bash
python -m pip install -r AgenticSimulator/tasks/pick_cube/RL/BC/requirements.txt
```

先用一个环境采集一条轨迹并打开窗口：

```bash
python -m AgenticSimulator.tasks.pick_cube.RL.BC.collect_bc_data \
  --run-dir "$(date +%Y%m%d%H%M)_bc_smoke" \
  --num-envs 1 --num-episodes 1 --device cuda:0 --gui
```

正式采集示例（默认无窗口，不启用相机传感器）：

```bash
python -m AgenticSimulator.tasks.pick_cube.RL.BC.collect_bc_data \
  --run-dir "$(date +%Y%m%d%H%M)_bc_collect" \
  --num-envs 64 --num-episodes 1024 --seed 42 --device cuda:0
```

`--run-dir` 必须显式指定，并以有效的 `YYYYMMDDHHMM` 开头。只提供目录名时，
自动放入本目录的 `data/`；也可以提供该目录下的完整路径。已有运行目录拒绝
覆盖。`--num-episodes` 是尝试数，包含成功与失败，不是成功轨迹配额。

每批所有环境结束后才重置。重置恢复关节与物体状态、速度、夹爪增益及控制器
内部状态。最后一批只计入实际请求的 episode。物体随机数由 seed 和全局
episode 编号派生，采样不依赖并行环境数；这不保证 GPU 物理轨迹逐位一致。

## 存储与失败记录

```text
data/YYYYMMDDHHMM_bc_collect/
  config.json
  scene_config.yaml
  result.json
  manifest.jsonl
  shards/
    shard_00000.h5
    shard_00001.h5
```

默认每 128 条 episode 一个分片，可用 `--episodes-per-shard` 调整。文件中
`episodes/00000000/` 等组独立保存：

- `observations`: float32 `(T, 48)`，schema 定义的原始物理观测。
- `actions`: float32 `(T, 8)`，m/rad 增量、开合指令及归一化力度。
- `simulation_time`: 50 Hz 采样时间，单位 s，不包含预热。
- `expert_state`: 诊断用专家阶段，不作为模型输入。
- 属性：初始位姿、预热后的初始状态、原始结果及 `complete`/`success`。

闭合 10 N 的力度标签为 0；打开时记录 -1，执行时忽略力度。观测包含的
`previous_action` 是上一次 50 Hz 动作。四元数单位化，并将最大绝对值分量
的符号固定为正；BC 推理应使用同一个 `observation()`，保持输入约定一致。

缓冲数据每个仿真秒写入，episode 结束时立即写入。成功与失败都保留。
关闭窗口或可捕获异常会尽量保存当前缓冲；尚未终止的轨迹留在原分片中，
`complete=False`，不伪造失败结果。强制杀进程可能丢失尚未写入的数据。
`manifest.jsonl` 记录开始和结束事件，`result.json` 区分成功、失败、中断和
尚未开始的数量。程序异常另外保留 `error.txt`，不自动重试或认定为基础设施故障。

## 训练

`--dataset` 指向一次采集的运行目录，而不是多次运行的共同 `data/` 父目录：

```bash
python -m AgenticSimulator.tasks.pick_cube.RL.BC.train \
  --dataset AgenticSimulator/tasks/pick_cube/RL/BC/data/你的采集运行目录 \
  --run-dir "AgenticSimulator/tasks/pick_cube/RL/BC/runs/$(date +%Y%m%d%H%M)_gru_train" \
  --device cuda:0
```

仍按完整 episode 划分训练/验证，允许同一分片内包含两组 episode。
默认只使用成功且完整的轨迹；失败和中断轨迹记录在 `split.json` 的排除项中。
观测统计量只由训练集计算。原有 NPZ 格式也可以读取。

当前模型为 `GRUBCPolicy`：48 维观测编码后输入单向、单层 GRU，再输出
8 维动作。每条完整轨迹从零隐藏状态开始，轨迹内部保持时间顺序；补齐帧
不参与损失或指标计算。`batch_size: 16` 表示 16 条轨迹；`hidden_dim: 256`
为编码/输出层宽度，`recurrent_dim: 128` 为 GRU 记忆维度。

训练默认读取 `config.yaml` 顶层参数，评估读取同一文件的 `eval:` 区块；
两个入口都可用 `--config` 指定另一份文件，命令行参数优先。
训练必须显式提供 `--dataset` 和 `--run-dir`；评估必须显式提供
`--checkpoint` 和 `--run-dir`。这些运行路径不从 YAML 自动填充。

## 模型可视化

使用训练 checkpoint 在新采样的单场景中闭环执行；默认打开 GUI，并在结束后
暂停物理、保持画面。控制动作来自模型，专家阶段和目标路点不参与控制：

```bash
python -m AgenticSimulator.tasks.pick_cube.RL.BC.eval_bc \
  --checkpoint AgenticSimulator/tasks/pick_cube/RL/BC/runs/你的GRU训练运行目录/checkpoints/best.pt \
  --run-dir "$(date +%Y%m%d%H%M)_bc_eval" \
  --device cuda:0 --seed 43
```

只提供运行目录名时，保存到 `BC/evals/<run-dir>/`。也可指定 BC 内的完整路径；
目录名必须包含有效的时间前缀，已有目录拒绝覆盖。`--scene-config` 默认仍为
`AgenticSimulator/scenes/configs/plate_cube.yaml`。`--seed` 和 `--episode-id`
决定物体初始位姿，可用于复现采样。

评估要求 checkpoint 中 `model_type: gru`，旧 MLP checkpoint 会明确报错，
需要使用新训练生成的 GRU 权重。模型结构及归一化从 checkpoint 恢复，
不会被当前 YAML 顶层训练参数覆盖。预热结束后隐藏状态初始化为零，
随后每个 50 Hz 控制步更新一次，四个物理子步不额外更新记忆。
每次独立评估都重新初始化，不继承其他场景的隐藏状态。

模型使用 checkpoint 自带的均值/标准差和位姿缩放。动作频率为 50 Hz，
200 Hz 物理步跟踪固定目标，力度按任务约定固定为每指 10 N：执行前将
闭合力度标签置为 0、打开力度标签置为 -1，下一观测保存实际执行的动作。

原生几何阈值保持 XY < 3 cm、底部中心垂直差 < 1 cm、连续稳定 > 0.5 s。
BC 没有专家阶段，因此评估器用实际观察到的抓取抬升（至少 6 cm，抓取距离
不超过 5 cm）与释放代替 LIFT/VERIFY 阶段门控。这里不执行专家逐阶段超时；
BC 使用 `--max-seconds` 指定的全局 rollout 仿真时间上限，默认 30 s。
该独立 BC rollout 不是 workspace-session-v1 的 Test 运行。

运行目录保留 checkpoint 路径、场景配置快照、初始/最终物理状态、
`result.json` 和 `rollout.npz`。轨迹记录原始观测、模型预测动作、实际执行动作
及仿真时间。异常保留当前轨迹和 `error.txt`，不会调用专家接管或自动重试。

GUI 默认按实时速度播放；`--no-real-time` 取消实时限速。任务结束后结果先保存，
视口继续处理鼠标事件但不推进物理。关闭窗口或 Ctrl+C 退出。若需保存后自动
退出，添加 `--exit-on-done`；`--headless` 也会自动退出。落盘后直接结束仿真进程，
避开已观察到的 Kit 关闭阻塞。
配置设为自动退出后，可用 `--no-exit-on-done` 恢复 GUI 保持画面；
`--real-time` 可覆盖 `eval.no_real_time: true`，`--gui` 可覆盖无窗口配置。

## CPU 检查

在具有 torch、numpy、pyyaml 和 h5py 的 CPU 环境中运行：

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s AgenticSimulator/tasks/pick_cube/RL/BC/tests -v
```

CPU 检查覆盖动作目标保持、坐标变换、夹爪控制、分片读写、按 episode 划分、
中断保存、整批调度及合成数据训练/加载。它们不验证实际 GPU 仿真的抓放成功率。
