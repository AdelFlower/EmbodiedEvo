"""Cube-to-plate adapter: CPU policy/evidence contract and lazy M3a batch runtime."""

from .contract import describe
from .layouts import make_layouts, training_layout
from .policy import PolicyAdapter
from .target import load_target_spec

__all__ = ["describe", "make_layouts", "training_layout", "PolicyAdapter", "load_target_spec", "create_runtime"]


def create_runtime(environment_ref, num_envs, render):
    """Caller launches SimulationApp first; M3a supports render=False only."""
    from simulators.cube_to_plate_batch import BatchRuntime
    return BatchRuntime(environment_ref, num_envs, render)
