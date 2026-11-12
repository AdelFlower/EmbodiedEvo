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
  --run-dir "runs/$(date +%Y%m%d%H%M)_bc_train" --device cuda:0
```

仍按完整 episode 划分训练/验证，允许同一分片内包含两组 episode。
默认只使用成功且完整的轨迹；失败和中断轨迹记录在 `split.json` 的排除项中。
观测统计量只由训练集计算。原有 NPZ 格式也可以读取。

## CPU 检查

在具有 torch、numpy、pyyaml 和 h5py 的 CPU 环境中运行：

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s AgenticSimulator/tasks/pick_cube/RL/BC/tests -v
```

CPU 检查覆盖动作目标保持、坐标变换、夹爪控制、分片读写、按 episode 划分、
中断保存、整批调度及合成数据训练/加载。它们不验证实际 GPU 仿真的抓放成功率。
