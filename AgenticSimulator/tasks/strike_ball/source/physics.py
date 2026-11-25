"""击球任务的初始源物理，供 Agent 在工作区中修改。"""


def configure(env_cfg):
    """创建环境前可在这里修改物体属性。"""
    pass


def before_step(env, dt):
    """每个物理子步可在这里施加额外的力，dt 的单位为秒。"""
    pass
