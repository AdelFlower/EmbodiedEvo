# 实验工具、源物理与训练接口

通过 `python scripts/env.py` 向宿主请求 rollout 或训练、读取会话状态，以及提交
最终候选和策略。客户端只发送请求；策略加载、仿真、训练和反馈发布由宿主完成。
所有命令示例均在工作区根目录执行。

每次 rollout 都在新建的仿真进程中初始化并运行一个回合。上次实验中的物理
状态、Python 变量和模块状态不会保留到下次实验；工作区代码和已发布的反馈会保留。
每轮训练也在新进程中创建环境，通过最新 checkpoint 继承学习状态，不延续上一轮
环境的物理状态。校准候选与策略登记在同一会话中保留，不提供跨会话恢复。

## 工作区

```text
workspace/
├── instruction.md       # 任务目标与执行要求，只读
├── tools.md             # 本工具手册，只读
├── source/              # 当前源物理候选，可修改
├── training/            # 训练配方，可沿用或修改，每轮训练单独归档
├── reference/           # 源环境的固定参考资料，只读，不进入候选归档
├── scripts/
│   └── env.py           # 实验客户端，只读
├── observations/        # 宿主发布的实验反馈，只读
└── scratch/             # 分析脚本、笔记和临时文件，可修改
```

修改源物理实现时编辑 `source/`，修改奖励、课程和训练代码时编辑 `training/`。
这两个目录分别整体归档为候选。将结果分析和临时文件放在 `scratch/`，不要把
视频、实验结果或无关临时文件放入代码候选中。

`reference/` 提供当前任务的环境配置、场景构建、policy 配置、adapter 和评价
实现，供查阅源环境的工作方式。这里的文件是参考副本，不作为候选代码执行。
参考配置中的值可能被源物理候选覆盖，应结合当前候选理解实际源物理。

容器中的本地 Python 可用于分析已公开的反馈。仿真和训练通过客户端提交给宿主，
不在容器内直接启动 Isaac Sim 或训练程序。

## status：查询状态

```bash
python scripts/env.py status
```

不运行实验，也不消耗源或目标实验预算。返回 JSON，主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `ok` | 请求是否正常处理 |
| `closed`、`reason` | 会话是否已关闭及当前状态原因 |
| `phase` | 兼容字段。首轮策略成功登记前为 `calibration`，登记后为 `training` |
| `calibrated_candidate` | `latest_policy_candidate` 的兼容字段，两者取值相同 |
| `latest_policy_id` | 最新成功登记的策略编号，初始为 `default`（P0） |
| `latest_policy_candidate` | 训练出最新策略的源物理候选编号，首轮策略成功登记前为 `null` |
| `training_rounds_completed` | 已成功训练并登记策略的轮数，不含 P0 |
| `training_iterations_per_request` | 每轮新增训练量，当前固定为 1000 iterations |
| `training_num_envs_default` | 默认并行环境数量，当前为 4096 |
| `requests_used` | 已登记的执行请求数，不是某一侧的实验次数 |
| `experiments_used` | source、target 各自已使用的次数 |
| `experiments_remaining` | 各侧剩余次数；`null` 表示未设置上限 |
| `latest_request` | 最近登记的执行请求编号 |
| `active_request` | 当前记录为执行中的请求编号；没有则为 `null` |
| `final_candidate` | 已选择的最终候选编号；未选择则为 `null` |
| `final_policy_id` | 最终提交的策略编号；尚未提交或无候选提交时为 `null` |

开始工作及决定下一次实验前，使用状态中的实时计数安排预算。
看 `training_rounds_completed` 了解已成功登记的轮数，看 `closed` 判断会话是否结束。
源侧探测返回的候选编号属于本次 rollout。只有新策略成功登记后，状态中的策略与训练候选才会一起更新。

## rollout：使用指定策略观察表现

使用最新策略运行源环境：

```bash
python scripts/env.py rollout --environments source
```

运行固定目标环境：

```bash
python scripts/env.py rollout --environments target
```

开始时可用 P0 在一次请求中分别运行两侧：

```bash
python scripts/env.py rollout --environments source target --policy-id default --episode-id 0
```

- `--environments` 必填，可指定 source、target 或两侧，每侧最多出现一次。
- `--policy-id` 默认为 `latest`，使用最新成功登记的策略；还未生成新策略时使用 P0。
  `default` 始终表示 P0；也可指定训练返回的策略编号，例如 `p000001`。
  参数接受策略标识，不接受模型文件路径。
- `--episode-id` 默认为 `0`，选择宿主配置中的回合条目，不是实验请求编号。
- `--seed` 可选，取值为 `0` 到 `2147483647` 的整数。指定时使用该种子；
  省略时，宿主为本次请求随机生成一个种子，并在响应顶层的 `seed` 中返回。

初始策略和回合条目由宿主任务配置决定，新策略在本会话训练成功后登记。
同一策略编号可用于源侧或目标侧 rollout；历史策略可用于对照，续训始终使用最新策略。

`--policy-id` 选择策略权重及其观测归一化状态，策略配置从其 `agent.yaml` 加载。
两侧都从固定 P0 的 `env.yaml` 恢复基础环境。源侧应用请求时工作区 `source/` 的
物理实现，目标侧应用宿主固定的目标物理。训练奖励和课程由 `training/` 在训练时配置。
例如，A 训出 P1 后，当前 `source/` 仍是 A，就可以在 A 上观察 P1。把 `source/` 改成 B
后，同一个源侧 rollout 命令观察的是 P1 在 B 上的表现。

两侧按指定顺序执行，各运行一个完整回合，使用所选的同一 policy 和回合描述。
同一次请求的两侧共用一个种子。`episode_id` 选择预设回合配置，`seed` 控制
任务定义的随机采样；即使回合编号不变，省略 `--seed` 的新请求也会重新生成种子。
种子不改变固定目标的物理参数，也不扩大任务规定的随机采样范围。

需要复查已有条件时，使用之前响应中的种子。例如，已有目标反馈的 `seed` 为
`123`，修改源代码后可运行：

```bash
python scripts/env.py rollout --environments source --policy-id default --episode-id 0 --seed 123
```

这里的 policy 和回合编号也应与已有目标反馈一致。分开请求源、目标时，若两次
都省略 `--seed`，宿主会分别生成种子，不能当作同一初始条件的成对实验。
相同种子用于尽量复现采样条件，不保证不同物理实现产生相同轨迹，也不保证
仿真逐步完全确定；仍应结合实际反馈比较。

每次请求源侧 rollout 时，宿主保存当前整个 `source/` 并在这份副本上运行。
从 `results.source.candidate_id` 读取本次编号。本轮显式训练时传入该编号，
即可用刚观察的物理版本续训。

`reference/` 不随候选归档或复制到候选运行目录。目标单侧请求不保存源候选。
一次只提交一个请求，等待返回后再修改 `source/` 或 `training/`，不在后台并发操作。

### 返回结果

客户端将响应 JSON 打印到终端。正常处理的实验响应包含：

| 字段 | 含义 |
| --- | --- |
| `ok` | 请求中的实验是否均正常执行；不代表任务成功或物理校准完成 |
| `request_id` | 本次执行请求编号 |
| `op` | 本次操作，rollout 响应为 `rollout` |
| `seed` | 本次请求确定的随机种子，两侧共用；复查时通过 `--seed` 指定 |
| `observation_dir` | 本次公开反馈目录，相对于工作区 |
| `results.source` | 源侧结果，请求包含 source 时存在 |
| `results.target` | 目标侧结果，请求包含 target 时存在 |

查看各侧自己的 `ok` 和 `error`。一侧失败时，另一侧可能已有可用结果。
请求被拒绝或执行器发生异常时，响应可能只包含错误信息，不保证存在上述全部字段。

源侧结果包含 adapter 返回的任务数据，执行层另提供候选编号 `candidate_id`、
实际使用的策略编号 `policy_id`、当前最新策略 `latest_policy_id`、进程退出码
`returncode`、反馈文件 `artifacts` 和日志 `logs`。这些字段取决于
实验已完成的阶段，例如候选准备失败时可能没有候选编号。出现可公开的源侧
异常时，还可能包含 `diagnostic`。任务数据的字段及指标含义由具体 adapter 定义。

目标侧返回执行状态、实际使用的策略编号，以及宿主任务配置允许公开的字段和文件。
公开范围不由 Agent 的请求决定，也不保证与源侧相同。仅依据已公开的反馈分析目标，
不尝试读取目标私有文件。

### 读取反馈文件

每次完成处理的执行请求会保存总响应到 `observations/<request_id>/result.json`。
以响应中的 `observation_dir`、`artifacts` 和 `logs` 为准定位文件。
这些返回路径已经相对于工作区，不要再次拼接 `observation_dir`。

`artifacts` 是可嵌套的字典，叶子值为公开文件路径。任务可以在其中提供回合
描述、轨迹、视频或其他反馈；具体键名、文件格式和记录内容由 adapter 定义。
`logs` 提供已公开日志的文件路径。

轨迹的字段、单位、坐标系和采样时机，以及录像的数量、视角和设置，应结合
当前任务的参考资料与 adapter 实现理解。本手册不约定所有任务共用一套状态字段
或相机名称。失败、未启用记录或目标权限限制都可能导致某些文件未提供。

## train：从最新 checkpoint 续训一轮

修改源物理并用当前策略观察后，将此次 `results.source.candidate_id` 传给 `train`。
例如，若这次返回的是 `c000002`，执行：

```bash
python scripts/env.py train --candidate-id c000002
```

首次训练必须显式传入编号。已有成功登记的续训策略后，省略 `--candidate-id` 会沿用
`status` 中的 `latest_policy_candidate`。例如，A 训出 P1 后又探测了 B，显式传 B
是在 B 上从 P1 续训，省略编号则是在 A 上从 P1 续训。

可以同时指定并行环境数量，例如 `python scripts/env.py train --candidate-id c000002 --num-envs 4096`。

- `--num-envs`：可选正整数；每次省略均使用默认值 4096，不继承上次命令的参数。
- 每轮固定新增 1000 iterations，每 100 iterations 保存 checkpoint；不提供
  `--iterations` 或保存间隔参数。
- 训练在本轮候选的源环境中运行，始终从最新成功登记的 checkpoint 继续；不提供
  `--policy-id`、目标环境训练、rollout 的回合编号或种子参数。

每轮使用提交时的 `training/` 快照。可以直接沿用现有配方，也可以先修改再提交。
宿主按本轮物理候选创建源环境，调用训练入口，登记本轮最后的 checkpoint。策略依次为
`default`（P0）→ `p000001`（P1）→ `p000002`（P2）→ …。
一次 `train` 完成一轮训练。随后可用 `python scripts/env.py rollout --environments target`
在固定目标环境中观察新策略，也可按需要在当前源环境或两侧探测。
中间 checkpoint 保留在宿主，不单独分配可选策略编号。

### 训练反馈

训练响应的顶层包含 `ok`、`request_id`、`observation_dir`、`op="train"`、
`candidate_id`、`iterations`、`num_envs` 和 `results.source`，不含 rollout 的 `seed`。
以下字段位于 `results.source`，准备失败或执行中断时可能只有部分字段：

| 字段 | 含义 |
| --- | --- |
| `ok`、`error`、`diagnostic` | 执行状态和可公开的错误信息 |
| `candidate_id` | 训练本轮策略所用的已归档源物理版本 |
| `training_candidate_id` | 本轮训练代码快照编号，与物理候选分开归档 |
| `parent_policy_id` | 本轮开始时加载的策略 |
| `policy_id` | 本轮成功登记的新策略编号 |
| `latest_policy_id` | 当前最新成功登记的策略 |
| `iterations`、`num_envs` | 初始训练入口报告的本轮训练量和环境数量 |
| `returncode` | 训练子进程退出码 |
| `artifacts.params` | 已公开的 `env.yaml` 和 `agent.yaml` 配置文件路径 |
| `artifacts.tensorboard` | 已公开的原始 TensorBoard 事件文件路径 |
| `logs` | 已公开的标准输出、错误日志或准备错误文件路径 |

模型文件和宿主路径不作为客户端可选参数；使用返回的策略编号进行 rollout。
配置和训练日志按实际产生的内容发布，训练反馈不包含自动评价结果或训练录像。
训练或策略登记失败时，保留此前成功登记的策略和候选配对。若策略已登记但反馈发布失败，最新策略
可能已经推进；结合返回的编号和 `status` 判断进度，不因 `ok=false` 自动重训。

## 源物理模块

源物理入口由宿主任务配置指定，位于 `source/` 内。入口文件名和可修改的
环境对象由当前任务决定；初始实现提供以下两个函数：

```python
def configure(env_cfg):
    # 在创建环境前修改源物理配置。
    pass


def before_step(env, dt):
    # 每个物理子步调用，dt 的单位为秒。
    pass
```

`configure(env_cfg)` 在任务 adapter 准备好基础环境配置后、创建环境前调用。
质量、摩擦等创建时参数在这里设置；具体对象和字段由任务的配置与场景实现
决定。修改会在下一次源侧 rollout 创建环境时生效。训练加载传给 `train` 的候选编号对应的已归档代码。

每次创建环境时，宿主从固定 P0 环境基准开始，调用一次物理 `configure`；不会
把上一轮已保存的校准结果作为下一轮基准再叠加一次。训练时，训练配方的
`configure` 先于物理 `configure` 执行。

`before_step(env, dt)` 在每个物理子步中、动作应用之后调用，之后宿主会将场景
数据写入仿真并推进物理。可通过已有 Isaac Sim / Isaac Lab 接口补充空气阻力等
逐步物理作用。纯参数修改可以保持此函数为空；实现不依赖跨回合隐状态。

可以在 `source/` 内组织辅助模块，并通过模块名导入。候选所需的辅助代码应放在
该目录内，不依赖工作区的 `reference/` 路径进行导入。使用已有模拟器接口，
不修改或重新编译引擎。
各轮遵循相同的策略观测与动作约定、控制周期、任务目标、评价规则及录像设置。

## 训练模块

当前任务入口为 `training/train.py`，提供以下两个函数，修改时保留其调用约定：

```python
def configure(env_cfg, agent_cfg) -> None:
    # 原地调整训练环境配置和算法配置。
    ...


def train(env, agent_cfg, checkpoint, output_dir, iterations) -> dict:
    # 加载 checkpoint，续训并保存本轮最后状态，返回结果字典。
    ...
```

每轮训练按以下顺序准备环境和配方：

1. 从固定 P0 的 `env.yaml` 恢复基础环境配置 `env_cfg`，从最新策略的 `agent.yaml`
   恢复算法配置 `agent_cfg`，并设置本轮并行环境数量。
2. 调用当前训练配方的 `configure(env_cfg, agent_cfg)`，按配方调整奖励、课程、
   学习率等。
3. 宿主将 `agent_cfg.save_interval` 设为 100，应用本轮物理候选的配置，创建训练环境。
4. 将环境和配置交给 `train`，加载最新 checkpoint 继续学习。

初始配方沿用 P0 基准中的奖励和课程，以及最新策略对应的 PPO 参数，可以不修改
直接使用。要继续沿用自己调整过的奖励或课程，保留 `training/` 中对应的配置代码；
每轮会在固定基准上重新执行它，不从上一轮保存的 `env.yaml` 自动继承这些改动。

`train` 接收已创建的原生环境 `env`、算法配置 `agent_cfg`、最新策略文件
`checkpoint`、本轮新的输出目录 `output_dir` 和本轮新增迭代数 `iterations=1000`。
初始实现复用 RSL-RL PPO，恢复模型、观测归一化和优化器状态，使用本轮配置的
学习率。训练入口负责环境包装、学习、保存和日志写入；仿真应用和原生环境的
创建、关闭由宿主管理，不在入口中重复启动。

初始实现保存最终模型 `model.pt`、配置 `params/env.yaml`、`params/agent.yaml`、
TensorBoard 日志和 `training_complete.json`。其中 `env.yaml` 记录本轮实际训练配置，
用于查看和分析，不作为后续 rollout 或续训的环境基准；`agent.yaml` 和 checkpoint
用于加载该策略及继续训练。返回字典包含：

| 字段 | 含义 |
| --- | --- |
| `checkpoint` | 本轮最后模型文件的路径，用于登记新策略 |
| `params_dir` | 配置目录路径，必须包含 `env.yaml` 和 `agent.yaml` |
| `parent_checkpoint` | 本轮加载的模型路径 |
| `log_dir` | 本轮日志目录 |
| `iterations`、`num_envs` | 本轮训练量和环境数量 |

模型和配置必须实际保存在提供的 `output_dir` 内，初始实现返回绝对路径；这些
路径用于宿主登记，不直接作为公开响应字段。保留返回约定及配置格式，使后续
训练和 rollout 能加载结果。按传入的 `iterations` 训练，保持每 100 iterations
保存，并在本轮结束时保存最后状态。

训练辅助模块放在 `training/` 内，可通过模块名导入，不依赖工作区的 `reference/`
路径。训练配方可以修改奖励、课程和训练代码，使用本轮候选的物理实现，保持任务目标、
成功判定、最终评价规则、观测与动作约定及控制周期不变。

## finish：提交最终候选和最新策略

已有成功登记的续训策略时，提交最新策略及训练出它的源候选：

```bash
python scripts/env.py finish
```

例如，B 训出了 P2，之后即使又做了源侧探测，默认仍提交 P2 和 B。
显式传入 `--candidate-id` 时，编号必须与 `latest_policy_candidate` 一致。
`finish` 只提交已有记录并结束会话。

如果首轮策略尚未成功登记，`finish` 可直接结束会话，最终候选与策略均为空。
此时也可指定已经归档的源候选与 P0 一起提交，例如 `python scripts/env.py finish --candidate-id c000001`
（用实际返回的编号替换 `c000001`）。

响应通过 `final_candidate` 和 `final_policy_id` 给出最终提交；未选择候选时两者
为 `null`。成功结束后停止提交请求，宿主会关闭会话并清理资源。

## 预算、时限和失败

源、目标的实验次数上限由宿主任务配置分别指定；剩余次数以 `status` 为准。
一次双侧 rollout 分别消耗两侧预算，一轮训练只消耗一次 source 预算，与并行
环境数量和迭代次数无关。计数发生在对应仿真子进程成功启动后，之后的
初始化失败、代码异常或超时仍计入；尚未启动的一侧不计入。
`status` 和 `finish` 不消耗实验预算。

rollout 的单次请求时限由宿主的 `session.execution_timeout_sec` 指定，当前为
1200 秒，包含准备、等待 GPU、所请求各侧的执行及反馈发布。双侧 rollout 共用
这段时间。

训练不设单次请求时限，等待 GPU、训练执行和反馈发布均不受上述 1200 秒限制。
正常情况下完成本轮 1000 iterations 后保存并返回；执行报错仍按失败返回。

会话总时限由 `session.timeout_sec` 单独控制，覆盖 Agent 启动后的整段工作时间，
包括等待实验的时间。当前未设置；若设置，也会限制训练所在的会话。

客户端在响应 `ok=true` 时返回退出码 0；响应失败或通信异常时返回非零退出码。
通信中断不代表实验未执行，也不撤销已经消耗的预算。不要自动重发请求；先尝试
查询 `status` 并检查已发布的反馈，再决定下一步。服务串行处理请求，执行期间
提交的状态查询也可能等待当前请求完成。

保留失败记录，根据已有证据分析原因。超时、进程崩溃或缺少报告本身不能证明是
基础设施故障，不应通过重复执行并丢弃失败结果来替换原始记录。
