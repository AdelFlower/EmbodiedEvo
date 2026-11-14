"""lift_cube 的初始源物理版本，复制到 Agent 工作区后作为候选修改起点。"""


def configure(env_cfg):
    """初始方块质量为 0.35 kg，静摩擦和动摩擦系数均为 0.8。"""
    cube = env_cfg.scene.dex_cube.spawn
    cube.custom_mass = 0.35
    cube.custom_material.static_friction = 0.8
    cube.custom_material.dynamic_friction = 0.8


def before_step(env, dt):
    """每个物理子步调用；初始版本无额外物理作用。"""
    pass
