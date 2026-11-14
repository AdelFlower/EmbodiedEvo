# 模拟器会话客户端

请在工作区根目录运行命令。复制文件或创建标记文件都不会提交代码。

```bash
python scripts/env.py status
python scripts/env.py instruction
python scripts/env.py observe
python scripts/env.py exec submission/solution.py
python scripts/env.py exec submission/solution.py --include skills/helper.py
python scripts/env.py reset     # 仅限 Playground
python scripts/env.py finish
```

`exec` 会一次性读取列出的文件，按照命令行顺序将 `--include` 指定的文件放在入口文件前面，
然后把合并后的 Python 源码发送给宿主机的 `env.step(code)`。辅助文件和入口文件在同一个
持久化命名空间中执行；这不是 Python 模块导入。编写可复用函数时，应避免在文件顶层执行
非预期的机器人动作。只能提交 `submission/` 或 `skills/` 下的 `.py` 文件。
宿主机会将实际提交的原始内容归档到只读的 `observations/<execution>/code.py`
历史记录中。提交后对文件的修改会在下一次调用时生效。

机器人状态、变量、函数和数值数组会在多次执行之间保留。提交的代码只支持导入
`numpy`（其中的数值计算子集）和 `math`。源码必须符合环境的 8 MiB ASCII 代码空间限制。
提交的代码不能读取工作区或宿主机文件；请使用工作区工具分析本地文件和图像，再提交由此
得到的数值计算逻辑。`--include` 是显式加载辅助源码的方式。工作区中的 Python
运行在另一个进程中，其状态与模拟器会话相互独立。

`instruction`、`observe` 和 `reset` 是便捷请求，分别通过同一个 `env.step(code)`
边界执行 `print(get_instruction())`、`get_observation()` 和 `reset()`。
它们不是额外的机器人原语。每次被接受的执行都会消耗一个执行名额，包括这些便捷请求，
以及执行时报错的程序。对于 RoboDojo，原生的 `step(action, ...)` 还会消耗基准任务的
原生动作预算。`native_steps_used` 统计已消耗的动作数，`native_steps_remaining`
报告剩余动作额度。对于 RoboDojo，Playground 重置会恢复官方的原生动作额度，但会保留
会话累计的执行请求次数和已用时间。其 `native_steps_used` 在多次重置之间仍然累计。
其他基准保留各自现有的动作预算规则。Test 禁止重置。`status` 和 `finish` 不消耗执行名额，
也不推进物理模拟。无效的通信请求和 Test 中的 `reset` 命令会在执行前被拒绝。
如果 Test 程序在代码中调用 `reset()`，则会报错，并消耗一个执行名额。

JSON 回复包含 `ok`、`stdout`、`stderr`、官方的 `success`、预算计数，以及相对于
工作区的路径。每次执行的回复和 `status` 都包含 `execution_budget`（允许的请求总数）、
`executions_remaining`、`native_action_limit`（从模拟器读取的当前任务动作上限），
以及 `native_steps_remaining`（当前尝试剩余的动作数）。这些预算彼此不同；请根据当前值
规划操作，不要假设不同任务的动作上限相同。Playground 重置会恢复 RoboDojo 的剩余动作
额度，但不会改变这两种总额度，也不会返还已使用的执行名额。
`observation` 指向 `observations/` 下的一个文件夹，其中包含原生分辨率的 PNG 相机图像、
`observation.json` 和 `arrays.npz`。JSON 中带有 `npz_key` 的条目引用该 NPZ 文件中的
数组；请使用 `numpy.load(path, allow_pickle=False)` 加载。每次执行还会在该文件夹中
保存 `result.json`、日志和原始相机视频。相同的原生分辨率 PNG 图像也会保存在宿主机运行
目录的 `raw_observations/` 中，用于回放和审计。
初始观察保存在 `observations/000000/`。
即使执行名额已经用完，`status` 仍会报告最近一次已完成的观察。
JSON 中的 stdout/stderr 字段各自最多包含最后 256 KiB 内容；观察目录会保留完整输出。
宿主机侧的私有外层奖励和流水账不会返回给客户端。

模拟只会在调用原生控制接口时推进。AI 思考和编辑代码会消耗实际时间，但不会推进物理步。
重置会恢复同一个场景；已消耗的执行名额、累计原生动作使用量和实际时间不会清零，
Python 变量也会继续保留。
Test 禁止重置，包括通过别名或辅助函数间接调用重置。

调用是同步且串行的。不要并发提交。错误可能发生在物理动作已经执行之后；系统不会回滚。
重试前请先检查结果。连接失败时，不要自动重新提交：代码可能已经执行。请先查询 `status`。
程序执行超时会使当前 episode 以无效状态结束，且不会启动替代场景。`finish` 会关闭会话；
调用前请保存好可复用的工作成果。Agent 结束时，其模拟器会话也会结束。
