from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SceneBundle:
    entities: dict[str, Any]

    # sim.reset() 后执行场景自己的初始化。
    initialize: Callable[[], None]

    # 仅用于预览，例如保持机械臂姿态。
    preview_step: Callable[[float], None] | None = None

    def write_data_to_sim(self):
        for entity in self.entities.values():
            entity.write_data_to_sim()

    def update(self, dt):
        for entity in self.entities.values():
            entity.update(dt)