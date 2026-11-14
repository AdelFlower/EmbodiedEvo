"""第一组实验的固定目标物理参数，由宿主加载，不属于源候选。"""


def configure(env_cfg):
    """设置方块质量为 0.6 kg，静摩擦和动摩擦系数均为 0.1。"""
    cube = env_cfg.scene.dex_cube.spawn
    cube.custom_mass = 0.6
    cube.custom_material.static_friction = 0.1
    cube.custom_material.dynamic_friction = 0.1

    # 同步内存中的场景描述，使记录与实际生成参数一致；不改原训练 YAML。
    for item in env_cfg.scene_config["objects"]:
        if item["name"] == "Cube":
            item["physics"]["mass"] = 0.6
            item["material"]["static_friction"] = 0.1
            item["material"]["dynamic_friction"] = 0.1


def before_step(env, dt):
    """本组仅修改创建时的参数，无额外的每步物理作用。"""
    pass
