"""Trajectory prior using the same one-shot detections as the actor."""

from integrations.deployable.observation import detection_positions
from integrations.friction_rl.network import ScriptController


class DetectionScriptController(ScriptController):
    """Reuse clean corrective labels and DART noise with substituted object poses."""

    def __init__(self, observer, params, force, sigma, rho, rng):
        super().__init__(params, force, sigma, rho, rng)
        self.observer = observer

    def __call__(self, t, obs, state, decoder):
        cube, plate = detection_positions(self.observer.detection)
        detected_state = {**state, "sample": {**state["sample"], "cube_position": cube},
                          "plate_position": plate}
        return super().__call__(t, obs, detected_state, decoder)
