"""Validate the host's fixed protocol once, before constructing an experiment."""

from copy import deepcopy
from dataclasses import dataclass


def exact(value, keys: str) -> bool:
    return isinstance(value, dict) and set(value) == set(keys.split())


def text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def fixed(value, expected) -> bool:
    return type(value) is type(expected) and value == expected


@dataclass(frozen=True)
class HostConfig:
    values: dict

    @property
    def training(self) -> dict:
        return self.values["training"]


def load_host_config(value: dict) -> HostConfig:
    if not exact(value, "task_adapter initial_policy source target layouts training judgment evaluation agent runtime sandbox"):
        raise ValueError("host configuration: exact §3 keys required")
    if not text(value["task_adapter"]):
        raise ValueError("task_adapter must be a nonempty string")
    policy = value["initial_policy"]
    policy_keys = {"adapter", "artifact", "calibration"}
    if (not isinstance(policy, dict) or set(policy) not in (policy_keys, policy_keys | {"training_checkpoint"})
            or not all(text(v) for v in policy.values())):
        raise ValueError("initial_policy: adapter, artifact, calibration and optional training_checkpoint strings required")
    for name, keys in (("source", "bundle"), ("target", "private_spec")):
        if not exact(value[name], keys) or not all(text(v) for v in value[name].values()):
            raise ValueError(f"invalid {name} descriptor")
    if not exact(value["layouts"], "judgment_stream final_stream source_eval_stream training_stream") or not all(
            text(v) for v in value["layouts"].values()):
        raise ValueError("four layout stream names required")
    if len(set(value["layouts"].values())) != 4:
        raise ValueError("four distinct layout stream names required")
    training = value["training"]
    constants = {"iterations_per_job": 100, "num_envs": 512, "episodes_per_env": 1,
                 "minibatch": 8192, "device": "cuda", "warmup_iterations": 2}
    if (not exact(training, "profile iterations_per_job num_envs episodes_per_env minibatch device warmup_iterations source_evaluation")
            or not text(training["profile"]) or not all(fixed(training[k], v) for k, v in constants.items())):
        raise ValueError("training must use the fixed §3 protocol")
    evaluation = {"include_initial": True, "every_iterations": 10, "num_envs": 512, "deterministic": True}
    if not exact(training["source_evaluation"], " ".join(evaluation)) or not all(
            fixed(training["source_evaluation"][k], v) for k, v in evaluation.items()):
        raise ValueError("source evaluation must include iterations 0, 10, …, 100 with 512 mean-action episodes")
    for name, settings in (("judgment", {"layouts": 4, "required_strict_successes": 4}),
                           ("evaluation", {"layouts": 30, "pass_strict": 27})):
        if not exact(value[name], " ".join(settings)) or not all(fixed(value[name][k], v) for k, v in settings.items()):
            raise ValueError(f"invalid fixed {name} settings")
    agent = value["agent"]
    if not exact(agent, "preset model reasoning_effort relaunch") or not all(
            text(agent[k]) for k in ("preset", "model", "reasoning_effort")):
        raise ValueError("invalid agent descriptor")
    relaunch = {"turn_continues": 3, "transport_failures": 5, "transport_wait_s": 300}
    if not exact(agent["relaunch"], " ".join(relaunch)) or not all(
            fixed(agent["relaunch"][k], v) for k, v in relaunch.items()):
        raise ValueError("invalid relaunch protocol")
    if not all(isinstance(value[k], dict) for k in ("runtime", "sandbox")):
        raise ValueError("runtime and sandbox must be host descriptor objects")
    return HostConfig(deepcopy(value))
