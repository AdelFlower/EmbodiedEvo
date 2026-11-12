"""Pick-cube BC 的观测与动作协议。

除动作中的力度分量及其在 previous_action 中的副本外，均使用物理量；
其余归一化由数据预处理模块负责。

坐标约定：
- 位置、姿态、线速度统一表达在机器人底座坐标系。
- 末端参考点为 panda_hand。
- 位姿排列为 [x, y, z, qw, qx, qy, qz]。
- 四元数必须为单位四元数。
- 位置单位 m，关节角和旋转角单位 rad，速度单位 /s。

动作约定：
- 平移增量相对于策略调用时的实际末端位置。
- 旋转增量使用底座坐标系中的旋转向量。
- 目标旋转为 R_target = Exp(delta_rot) @ R_current。
- 每次策略调用生成一次目标，在下次调用前跟踪该目标。
- grip_command：0 表示打开，1 表示闭合夹持。
- grip_force_normalized：归一化力度，范围 [-1, 1]。
- 每根手指的闭合驱动力 F = 10 + 10 * grip_force_normalized，单位 N。
- 对应驱动力范围为 [0, 20] N；打开时忽略力度分量。
- previous_action 同样保存归一化力度，不保存牛顿值。

本协议不包含专家阶段、目标路点和成功标志。
"""

SCHEMA_VERSION = "pick-cube-bc-v2"

OBSERVATION_FRAME = "robot_base"
END_EFFECTOR_BODY = "panda_hand"
QUATERNION_ORDER = "wxyz"

GRIP_OPEN = 0
GRIP_CLOSE = 1

GRIP_FORCE_MIN_N = 0.0
GRIP_FORCE_MAX_N = 20.0
GRIP_FORCE_SCALE_N = (GRIP_FORCE_MAX_N - GRIP_FORCE_MIN_N) / 2.0
GRIP_FORCE_OFFSET_N = (GRIP_FORCE_MAX_N + GRIP_FORCE_MIN_N) / 2.0


def grip_force_to_newtons(normalized_force):
    """将 [-1, 1] 力度转换为每根手指的驱动力；执行器负责输入限幅。"""
    return GRIP_FORCE_OFFSET_N + GRIP_FORCE_SCALE_N * normalized_force


def normalize_grip_force(force_n):
    """将 [0, 20] N 的每指驱动力转换为 [-1, 1] 训练标签，不做裁剪。"""
    return (force_n - GRIP_FORCE_OFFSET_N) / GRIP_FORCE_SCALE_N


# 顺序属于数据协议，采集、训练和推理必须保持一致。
OBS_FIELDS = (
    ("arm_joint_pos", 7),       # 手臂关节角
    ("arm_joint_vel", 7),       # 手臂关节速度
    ("finger_joint_pos", 2),    # 两根手指的位置，单位 m
    ("ee_pose", 7),             # 当前 panda_hand 位姿
    ("cube_pose", 7),           # 方块实时位姿
    ("cube_linear_vel", 3),     # 方块线速度
    ("plate_pose", 7),          # 盘子实时位姿
    ("previous_action", 8),     # 上一步交给执行器的动作
)

ACTION_FIELDS = (
    ("delta_pos", 3),           # 末端平移增量，单位 m
    ("delta_rot", 3),           # 末端旋转向量，单位 rad
    ("grip_command", 1),        # 打开 0 / 闭合 1
    ("grip_force_normalized", 1),  # [-1, 1]，执行时转换为每指 [0, 20] N
)


def _build_slices(
    fields: tuple[tuple[str, int], ...],
) -> tuple[dict[str, slice], int]:
    """根据字段顺序生成索引范围和总维度。"""
    slices = {}
    offset = 0

    for name, size in fields:
        if name in slices:
            raise ValueError(f"重复字段：{name}")
        if size <= 0:
            raise ValueError(f"字段维度必须大于零：{name}")

        slices[name] = slice(offset, offset + size)
        offset += size

    return slices, offset


OBS_SLICES, OBS_DIM = _build_slices(OBS_FIELDS)
ACTION_SLICES, ACTION_DIM = _build_slices(ACTION_FIELDS)

assert OBS_DIM == 48
assert ACTION_DIM == 8


# 连续动作共 7 维，开合指令单独作为二分类标签。
CONTINUOUS_ACTION_INDICES = (0, 1, 2, 3, 4, 5, 7)
GRIP_COMMAND_INDEX = ACTION_SLICES["grip_command"].start
GRIP_FORCE_INDEX = ACTION_SLICES["grip_force_normalized"].start


# Episode 初始时没有上一动作：
# 零位姿增量、打开指令、归一化力度 -1（对应 0 N）；打开时力度无效。
INITIAL_PREVIOUS_ACTION = (
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    float(GRIP_OPEN),
    -1.0,
)
