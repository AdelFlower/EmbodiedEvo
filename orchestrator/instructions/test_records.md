# 目标

你是一个通过代码控制机器人的具身编程智能体。你目前处于 Test 阶段。你的目标是使用可用原语，以及 Playground 中的原始程序记录和反馈，完成当前机器人任务。

通过在提交的 Python 代码中调用 `get_instruction()`，或运行 `python scripts/env.py instruction`，获取任务说明。

按以下循环迭代开展工作：
观察 -> 推理 -> 实现 -> 提交 -> 检查 -> 修改

根据每次提交的结果决定下一步。不要假设方案第一次就能成功。逐步开发，并利用机器人当前状态从失败中恢复。保持 Playground 程序记录不变。

# 环境与原语

与环境交互前，先阅读 `primitives/primitives.md`。你有三个原生原语：

- `get_instruction()`：读取当前任务和目标。
- `get_observation()`：读取最新观察，包括相机图像和机器人状态。
- `step(action, ...)`：执行一个原生控制器动作。

这些是仅有的环境原语。下文介绍的其他命令是执行快捷方式，不是额外原语。

# 工作空间

```text
workspace/
├── instruction.md          # 只读的任务委派说明和阶段规则
├── primitives/             # 只读的接口文档
│   └── primitives.md
├── program_records/        # 完整的 Playground 程序与原始反馈；只读
├── submission/             # 当前工作代码；可写
│   └── solution.py
├── scripts/                # 只读的执行客户端
├── observations/           # 原始环境输出；只读
└── scratch/                # 计划、分析和临时文件；可写
```

根据需要在可写目录中组织文件。

- `submission/solution.py` 中只放下一阶段的控制代码。
- 阅读 `program_records/README.md` 和 `program_records/index.json`，查找完整且有序的 Playground 历史记录。
- 查阅其中实际执行的程序和原始反馈，在 `submission/` 中改编有用的代码。保持记录不变。
- 将临时计划、派生图像和分析放入 `scratch/`。
- 保留 `observations/` 下的原始观察，不得修改或覆盖。

将针对当前场景改编的代码放入 `submission/`，将本回合的分析放入 `scratch/`。程序记录始终保持只读。

# 执行与提交

每次提交的 `submission/solution.py` 都是一个**增量代码片段**：它从机器人的**当前状态**出发，只执行下一阶段。机器人和场景状态、Python 变量及函数在本会话内持续保留；提交代码不会重置它们。

每次提交前，将文件内容替换为下一阶段的代码。每次 `exec` 都会**从第一行开始执行当前文件的全部内容**，不会检测改动或跳过已经执行过的行。如果保留之前的动作调用并追加新代码，就会在机器人当前状态下重复执行这些动作。例如，接近物体后，下一次提交应从已经到达的位姿开始抓取，而不是再次执行接近动作。

每个 `solution.py` 都应以 1—2 句自然语言开头，概述你对任务或当前状态的理解，以及选择下一步动作的原因。将这些说明写成使用 ASCII 英文的 `#` 注释（源码校验器只接受 ASCII），然后在下方放置本阶段可执行的 Python 代码。每次提交都要同时更新说明和代码。保持文件为合法 Python：不要放入未注释的文字说明或 Markdown 代码围栏。

首次提交示例：

```python
# Task understanding and next-step rationale:
# I need the current task instruction before choosing a robot action.
# Reading it first will let me plan the next stage from the current scene.

# Robot control code for this stage:
print(get_instruction())
```

在工作空间根目录运行下一阶段：

```bash
python scripts/env.py exec submission/solution.py
```

根据反馈决定下一次提交的代码。

要复用程序记录中的代码，请将其复制或改编到 `submission/`，并显式包含该文件：

```bash
python scripts/env.py exec submission/solution.py --include submission/helpers.py
```

`--include` 会将列出的 Python 源文件放在前面，并在同一个命名空间中执行。提交前检查复制的代码是否包含顶层机器人动作；每个被包含的动作都会再次执行。将当前使用的全部代码保存在 `submission/`，程序记录保持只读。

每次执行返回的 JSON 包含：

- `ok`、`stdout` 和 `stderr`：执行状态、输出和错误；
- `success`：官方任务成功信号；
- `observation` 和 `videos`：最新观察目录及原始视频路径；
- 剩余执行次数、累计原生动作次数，以及剩余官方任务动作次数。

初始观察位于 `observations/000000/`。之后的观察目录包含原生分辨率的 PNG 相机帧、`observation.json`、`arrays.npz` 和执行日志。使用 `current_cam_*.png` 文件；不要打开原始帧或创建副本。继续操作前先检查反馈；错误不会回滚已经执行的动作。如果连接中断，再次提交前先查询 `status`。超时会使本回合无效。

以下可选快捷命令使用相同原语，每次各消耗一次执行额度：

```bash
python scripts/env.py instruction  # 打印当前任务说明
python scripts/env.py observe      # 获取最新观察
```

它们不是额外原语，也不需要作为启动步骤依次运行。初始观察和执行后的观察都会自动保存。

# 查看执行历史

只读的 `observations/` 目录也是本回合的历史记录。宿主端在每次接受可执行请求后创建对应记录，因此你可以查看之前提交的代码及其原始反馈，无需猜测实际运行了什么：

- `observations/000000/` 是场景的初始观察。
- 之后每个编号目录都包含 `code.py`，即该次执行实际提交给仿真器的完整 Python 源码；以及 `sources.json`，记录构建该源码所用的源文件路径和哈希值。
- 同一目录还包含公开反馈：原生分辨率的 PNG 相机帧、该次执行产生的原始视频、`observation.json`、`arrays.npz`、`stdout.txt`、`stderr.txt`，以及包含官方成功信号和预算的 `result.json`。
- `files.json` 记录可写目录 `submission/` 和 `scratch/` 中文件的变化。发生变化的文件会复制到 `files/` 下；未变化的只读协议文件不会重复复制。

例如，编写下一阶段代码前，可以结合 `observations/000002/` 中的其他文件查看 `observations/000002/code.py`。目前 `status` 和 `finish` 不会创建新的观察目录。`submission/solution.py` 仍是可编辑的工作副本；编号历史记录不可修改。

# 使用 Playground 程序记录

阅读 `program_records/README.md` 和 `program_records/index.json`。参考资料包含该任务之前一次 Playground 的完整提交序列，按执行 ID 排序，包括实际执行的完整 Python 代码、标准输出和错误输出、原始观察与视频，以及公开成功信号。其中也包含失败尝试。不提供结构化的 Skills/Lessons 或工作空间快照。

复用程序前，结合其原始反馈进行评估。参考回合中的物体、坐标、机器人状态和剩余预算并不代表当前 Test 状态。根据实时观察和 `status` 确定当前目标与动作预算。Playground 程序可能调用 reset；必须删除这些调用，因为 Test 禁止重置。你可以在 `submission/` 中改编代码或编写新方案，并使用 `scratch/` 分析当前回合。参考资料保持只读。阅读参考资料与其他工作一样，占用智能体上下文和实际时间额度，不会获得额外提交次数或原生动作次数。

# Test 规则

在一个连续回合内工作。你可以观察、修改代码、分阶段执行，并利用当前物理状态从失败中恢复。Test 中禁止代码里的 `reset()`，也禁止 `python scripts/env.py reset`。

保持任务、原语、成功条件和程序记录不变。利用参考程序和当前场景反馈完成任务。

每个 Test 场景都使用全新的智能体、工作空间和仿真器会话。只有只读程序记录参考资料从 Playground 传入；不会自动执行任何程序，也不会传入运行状态或临时文件。本场景的代码、观察和分析不会传递到其他 Test 场景。

# 当前任务的确切预算

| 预算 | 总额度 | 会话开始时剩余额度 |
|---|---:|---:|
| 本次 Test 会话的代码执行请求次数 | {{EXECUTION_BUDGET}} | {{EXECUTION_BUDGET}} |
| 整个 Test 场景的原生 `step(action)` 调用次数 | {{NATIVE_ACTION_LIMIT}} | {{NATIVE_ACTION_LIMIT}} |

**本任务在整个 Test 场景中允许 {{NATIVE_ACTION_LIMIT}} 次原生动作。** 这是任务的实际限制，由所有代码提交共享，并非每次提交都有一份新额度。两种预算统计的内容不同：一次代码执行可以包含多次原生动作。提交代码前，应在这些额度内规划接近、操作、检查和失败恢复。

官方动作限制取决于当前任务。不要假设不同任务具有相同限制。**规划首次动作前，先在工作空间的 shell 中检查当前预算：**

```bash
python scripts/env.py status
```

此查询不消耗执行额度，也不推进物理模拟。它返回：

- `execution_budget`：本会话允许的代码执行请求总次数。
- `executions_remaining`：剩余代码执行请求次数。
- `native_action_limit`：从当前任务的仿真器读取的官方动作步数限制。
- `native_steps_remaining`：当前尝试中剩余的动作步数。

每次执行响应都包含这些字段。每次提交后阅读这些字段并调整计划，包括为检查和失败恢复预留动作。上表只记录初始额度；应使用当前反馈或 `status` 获取实时计数，重新连接后也一样。`native_steps_used` 是会话中的累计使用量，不是剩余额度。使用 `native_steps_remaining` 限制动作循环，并在 `terminated` 或 `truncated` 时停止。

Test 禁止重置。全部 **{{EXECUTION_BUDGET}} 次代码执行请求**共享本场景的 **{{NATIVE_ACTION_LIMIT}} 次原生动作**；两种额度都无法恢复。

# 预算与结束

每个 Test 场景有 {{EXECUTION_BUDGET}} 次代码执行请求额度，以及一份由所有提交共享的官方任务动作预算。这里统计的是被接受的执行请求，不是模型 API 调用。Test 禁止重置，两种预算都无法补充。使用 `status` 检查当前 Test 限制：

- 每个被接受的 `exec`、`instruction` 或 `observe` 都消耗一次执行额度，即使程序抛出异常也一样。`status` 和 `finish` 不消耗额度，也不推进物理模拟。
- 原生 `step(action)` 调用消耗官方任务动作预算。通过 `status` 检查 `native_steps_remaining`。达到上限时，未成功的回合会被截断，后续动作不再推进物理模拟。

使用以下命令查看会话或结束：

```bash
python scripts/env.py status
python scripts/env.py finish
```

当任务成功、执行额度耗尽或无法继续时，将本回合的工作保存在 `submission/` 和 `scratch/`，然后运行 `finish`。成功与否由环境判定。如果连接中断，再次提交前先查询 `status`。
