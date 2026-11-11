"""Task-independent boundaries; adapters report execution failures as data."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class PolicyBundle:
    id: str
    checkpoint_ref: Any
    complete_training_state: bool


@dataclass(frozen=True)
class PreparedSource:
    ref: Any
    manifest: dict


@dataclass(frozen=True)
class CapabilityUnavailable:
    gap: str


@dataclass
class StepBatch:
    actor: Any
    critic: Any
    public: dict
    source_truth: dict
    done: Any
    execution_error: list[str | None]
    outcome: list[dict | None]
    host_scoring: list[dict]


class Runtime(Protocol):
    def reset(self, layouts: list) -> StepBatch: ...
    def step(self, actions: Any) -> StepBatch: ...
    def close(self) -> None: ...


class TaskAdapter(Protocol):
    def describe(self) -> dict: ...
    def make_layouts(self, stream: str, seed: int, count: int) -> list: ...
    def prepare_source(self, bundle_dir: Path) -> PreparedSource | CapabilityUnavailable: ...
    def create_runtime(self, environment_ref: Any, num_envs: int, render: bool) -> Runtime: ...


class Policy(Protocol):
    def act(self, actor_batch: Any, deterministic: bool) -> Any: ...


class PolicyAdapter(Protocol):
    def load(self, bundle: PolicyBundle) -> Policy: ...


@dataclass(frozen=True)
class TrainSpec:
    experiment_seed: int
    task_number: int
    round_id: str
    environment_ref: Any
    entry_policy: PolicyBundle
    reward_snapshot: Path
    iterations: int
    warmup_iterations: int
    num_envs: int
    minibatch: int
    source_evaluation: dict
    training_layout_stream: dict
    profile: str
    device: str
    episodes_per_env: int


@dataclass(frozen=True)
class TrainResult:
    status: Literal["complete", "failed"]
    iterations_completed: int
    checkpoint_ref: Any
    partial_checkpoint_refs: list
    curve_ref: str | None
    trace_ref: str | None
    evaluation_status: Literal["complete", "failed_mid_task", "final_incomplete"]
    rollouts_completed: int
    rollouts_partial: int
    transitions: int
    evaluation_episodes: int
    elapsed_s: float
    gpu_s: float
    optimizer_state: Literal["restored", "created"]
    error: str | None


class TrainHandle(Protocol):
    def progress(self) -> dict: ...
    def poll(self) -> TrainResult | None: ...
    def stop(self) -> None: ...


class PPOBackend(Protocol):
    def start(self, spec: TrainSpec, directory: Path) -> TrainHandle: ...


class RewardRunner(Protocol):
    def precheck(self, reward_path: Path, trace_refs: list) -> dict: ...


@dataclass
class EpisodeRecord:
    layout_id: int
    steps: int
    public: list[dict]
    source_truth: list[dict]
    outcome: dict
    host_scoring: list[dict]
    status: Literal["complete", "execution_error", "cancelled"]


class EndState(str, Enum):
    RECOVERED = "recovered"
    SUBMITTED_NOT_RECOVERED = "submitted_not_recovered"
    AGENT_EXITED = "agent_exited"
    HOST_CANCELLED = "host_cancelled"
