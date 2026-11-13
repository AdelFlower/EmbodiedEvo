# Pick-cube PPO with expert roll-in curriculum

独立 Isaac Lab 在线训练，复用 BC 场景、48 维观测协议、IK 和动作执行器。
它不接入 workspace-session-v1 Playground/Test，不在官方 Test 中学习或重置。

网络输出 6 维位姿残差、开合和夹持力。力度按 `F=10+10*a7` 映射为每指
0–20 N，打开时闭合力为零。专家用 10 N，网络的力度仍由自己的 Beta 分布输出。
位姿由高斯分布输出，开合为 Bernoulli。策略频率 50 Hz，物理频率 200 Hz；
每次锁存目标并跟踪四个子步，沿用原来的关节限速、限位和成功判据。
开合命令默认每 10 个策略步选择一次（5 Hz），其间保持实际上一条开合命令。
位姿及夹持力仍每步更新；PPO 只在真正选择开合的帧计入 Bernoulli 概率，
保持期间不重复计分。训练和独立评估使用同一频率。
PPO 打开夹爪时，每根手指从自己的实际位置以 0.05 m/s 向外移动，避免从零开始
的共同位置目标先挤压、居中方块。闭合仍执行网络输出的力。训练专家搬运时补偿
实际抓取偏移，以方块中心对准盘子；网络执行不调用这个专家补偿。

Actor 使用 GRU，critic 使用独立 MLP。输入包括物理观测、实际上一条动作，
以及从物理观测计算的夹爪到方块、方块到盘子相对位置。专家阶段不输入网络。
默认 `model.absolute_targets: true`：位姿头预测基座系中的目标位置和目标姿态，
在网络内部根据当前手掌位姿换算成原执行器所需的残差，再采样动作。
固定的目标输出会保持固定姿态，避免恒定残差不断累积漂移；目标仍完全由网络决定。
采样分布和 PPO 概率都定义在实际残差上，IK、限速及夹持力接口保持原协议。
从残差头切换到目标头的实验应显式传 `--reset-pose-head`；常规同结构续训不用此参数。

## 训练方法

从原始初态由专家执行前缀，到接管位置后完全由网络控制。课程依次为：
释放、下降放置、搬运、抬升、夹取、下降接近、接近、抬高避障、全程策略。
接管时保留实际物理状态和控制器状态，不瞬移物体或伪造接触。
前缀和预热帧不参与 PPO 损失；接管时清空策略记忆。保留少量上一课程样本。
每个阶段内部还按专家已完成比例 98%、95%、90%、85%、70%、50%、25%、0% 逐步提前接管，
每个接管点都需要通过冻结评估。闭合阶段按 1.5 秒闭合过程的时间比例接管。

默认 512 个随机初态环境，rollout 64 步，每轮 8 个 minibatch、4 个 epoch。
独立 critic 避免价值回归直接改变 actor 特征；KL 调整学习率并记录触发停止的 KL。
积累足够训练尝试后，定期冻结权重在独立场景检查，达到 80% 才提前接管。
最终从原始初态、关闭专家，在 seed 43、44 各评估 64 次，均达到 90% 才写入
`verified_policy.pt`。这是验收门槛，代码实现不意味着已经获得该成功率。

训练奖励包含势函数进展和按秒累计的物理状态奖励，鼓励持续保持接近、对齐、
抬升及向盘子搬运。完成奖励为 100，高于停留在最高分状态的折扣持续收益。
进入原有放置容差后，放置奖励不再鼓励继续压低或推动方块，接近奖励退出，
改为鼓励夹爪离开，让物体稳定。成功仍要求实际释放及原有稳定时间。
这些训练奖励不改变原任务成功判定。

在能够运行 BC 的 Isaac Lab Python 环境，从仓库根目录执行：

```bash
python -m AgenticSimulator.tasks.pick_cube.RL.PPO.check_env \
  --num-envs 16 --run-dir "$(date +%Y%m%d%H%M)_expert_baseline" --headless

python -m AgenticSimulator.tasks.pick_cube.RL.PPO.train \
  --run-dir "$(date +%Y%m%d%H%M)_ppo_curriculum" --headless
```

`check_env` 只检查专家通过 PPO 执行接口的成功率，不能当作 RL 策略结果。
`--num-envs` 覆盖并行数；`--updates` 是更新次数上限。
平移探索初始标准差为 1 cm，旋转为 0.03 rad，随训练学习；续训默认恢复已学到的
标准差，只有显式传 `--position-std 0.01` 才将平移探索重新设为 1 cm。
`--curriculum-level 8` 从全程策略课程开始，不保留上一档辅助样本；
`retention_fraction` 只用于中间课程。独立评估也关闭这种保留和专家控制。

续训使用新目录，保留之前所有失败记录：

```bash
python -m AgenticSimulator.tasks.pick_cube.RL.PPO.train \
  --resume AgenticSimulator/tasks/pick_cube/RL/PPO/runs/实际运行目录/checkpoints/latest.pt \
  --updates 2000 --run-dir "$(date +%Y%m%d%H%M)_ppo_resume" --headless
```

续训恢复网络、优化器、更新编号、课程级别和阶段内接管点；物理环境及课程窗口重新开始。
`--updates 2000` 表示训练到第 2000 轮。`--curriculum-level` 可显式覆盖接管级别。
默认不加载失败的 BC 权重，也不启用离线 BC 损失。需要辅助监督时传
`--dataset 路径 --bc-weight 数值`；数据统计仅由训练集计算，保存于权重。
续训保持 checkpoint 的观测归一化，不因新增专家数据改变现有网络输入尺度。

`ppo.online_expert_weight` 默认 0.3：训练专家继续观察网络执行后的物理状态，
给出独立动作标签，在同一段网络历史上计算辅助监督损失。标签不替换网络动作，
也不作为 PPO 采样动作或网络输入。设为 0 可关闭。
默认 `online_expert_epochs: 4` 在每轮 PPO 之后单独做 4 轮监督更新，再用更新后的
策略采集下一轮数据，避免 PPO 的旧策略 KL 限制阻止动作拟合。日志分别记录
PPO 步数和 `online_expert_steps/loss`。更新后用最近观测历史刷新 GRU 状态。
在线反馈专家跟踪方块当前位置，按实际指间距、手指运动及夹持状态给出接近、
闭合、抬升、搬运和释放标签。搬运时保留当前偏航角并纠正倾斜；靠近盘子中心
且高度足够低才指导释放。标签持续到真实任务结束，不依赖原专家假定动作已经
执行的阶段计时。原始专家前缀保持原有流程。这些反馈标签仅用于训练。
释放前只要实际命令仍闭合，位姿标签继续指向下降目标；实际打开后才保持位置，
避免位姿头提前停止而开合头尚未松开的不一致状态。
可用 `PPO.check_env --feedback-expert` 单独诊断反馈专家；其结果明确标为专家基线，
不能作为网络策略成绩。
部署和全程独立评估不创建专家。续训参数 `--grip-logit-scale 0.3` 可将开合
输出头缩小以增加随机探索，正系数保留该时刻确定性开合选择；默认不缩放。

## 独立可视化评估

```bash
python -m AgenticSimulator.tasks.pick_cube.RL.PPO.eval \
  --checkpoint AgenticSimulator/tasks/pick_cube/RL/PPO/runs/实际运行目录/checkpoints/latest.pt \
  --run-dir "$(date +%Y%m%d%H%M)_ppo_eval_seed43" --seed 43 --episode-id 0
```

评估入口始终关闭专家，从原始初态运行确定性策略。默认 GUI 并保持结束画面；
`--exit-on-done` 结束即退出；`--headless` 无窗口。保存动作轨迹和实际任务结果。
成功要求实际抬升、释放、放置几何及稳定时间，与 BC 评估一致；不是官方 Test 分数。

批量评估使用 `PPO.validate`，参数为 `--checkpoint 路径 --num-envs 64
--seed 43 --run-dir 新目录 --headless`。默认全程关闭专家；仅诊断课程时显式加
`--curriculum-level 0..7`，结果标记为辅助评估。记录前 8 个环境的动作和观测轨迹，
其中 `policy_controlled` 标出真正由网络执行的帧。

## 结果

当前工作区已训练出的权重：
`runs/202611131545_ppo_release_mode_consistency/checkpoints/verified_policy.pt`（update 1294）。
从原始初态、确定性网络控制的冻结评估：seed 10042、43、44 各 64/64。
另启不创建专家的 `PPO.validate`，seed 46 为 64/64；记录位于
`evals/202611131550_verified_independent_seed46/`。
单环境 GUI 的 seed 45、episode 0 也实际完成任务，记录位于
`evals/202611131549_verified_gui_40/`。这些是本独立 Isaac Lab 任务的有限样本结果；
权重与实验数据保留在本机忽略目录中，不随 Git 提交。

训练使用 PPO 和在线物理反馈专家的独立监督更新；推理只使用网络与原位姿执行器，
开合和 0–20 N/指的夹持力由网络输出，专家的 10 N 只是训练标签。

本机一键可视化（默认 seed 45，成功后保留窗口）：

```bash
bash AgenticSimulator/tasks/pick_cube/RL/PPO/runs/202611131545_ppo_release_mode_consistency/view_policy.sh
```

该本机脚本使用 Isaac Sim 自带 Python，并复用已有 Python 依赖。
可追加 `--exit-on-done` 在任务结束后关闭窗口，或 `--seed 数字 --episode-id 数字`。
完整交付记录见该运行目录下的 `delivery.json`。

- `metrics.jsonl`：PPO 损失、实际参与优化帧数、学习率、停止 KL、课程窗口。
  `success_rate` 包含辅助训练；全程网络结果单独为 `full_policy_success_rate`。
- `episodes.jsonl`：所有已结束尝试，标记接管级别、专家辅助、策略步数和失败原因。
- `validations/`：冻结策略评估，区分辅助验证与全程策略，每个环境只计一次尝试。
- `checkpoints/`：初始、最新、定期、课程通过权重；只有最终验收通过才保存
  `verified_policy.pt`，`result.json` 才标记 `policy_success_verified: true`。
- `interrupted_episodes.jsonl`：评估边界截断的训练尝试；`unfinished_episodes.json`
  和 `final_state.npz` 保留退出时未完成尝试。未完成不计为成功。
- 失败运行不覆盖；每次必须提供以 `YYYYMMDDHHMM` 开头的新 `--run-dir`。

方法参考：[PPO](https://arxiv.org/abs/1707.06347)、
[JSRL](https://proceedings.mlr.press/v202/uchendu23a.html)。课程、奖励和执行接口是本任务实现。
