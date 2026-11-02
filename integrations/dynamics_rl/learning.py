"""Optional CPU learner for the host-owned state-policy experiment.

Torch and Stable Baselines3 are imported only when a learner is constructed.
The simulator worker and the ordinary CPU test environment do not need them.
All observations passed here have already gone through the shared encoder.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

POLICY_FORMAT = "dynamics-rl-actor-v1"
# In the v1 encoder these are orientations, velocities, redundant robot/gripper
# state, and previous actions. Their expert-only correlations caused unstable
# feedback and copying of the previous gripper command during development.
ROBUST_INACTIVE_COLUMNS = tuple(np.r_[3:13, 16:32, 35:46, 49:56].tolist())


def demonstration_boundaries(observations, actions):
    """Detect action discontinuities from data, without teacher stages or timings."""
    observations, actions = np.asarray(observations), np.asarray(actions)
    if observations.shape[1:] != (57,) or actions.shape[1:] != (7,):
        raise ValueError("Robust BC requires the 57-state/7-action v1 contract")
    starts = np.r_[True, np.diff(observations[:, -1]) > 0]
    episodes = np.cumsum(starts)
    delta = actions - observations[:, 49:56]
    acceleration = np.linalg.norm(delta - np.roll(delta, 1, axis=0), axis=1)
    acceleration[starts] = 0
    threshold = max(1e-4, float(np.quantile(acceleration, 0.95)))
    switches = np.flatnonzero((acceleration >= threshold) | (np.abs(delta[:, 6]) > 1))
    window = set()
    for index in switches:
        window.update(i for i in range(max(0, index - 3), min(len(actions), index + 4))
                      if episodes[i] == episodes[index])
    return np.asarray(sorted(window) if window else np.arange(len(actions)), dtype=np.int64)


def _actor_gradient_mask(model):
    """Recreate the hook per update block, including after loading a full checkpoint."""
    if model.bc_mode != "robust":
        return None
    import torch

    first = model.actor.latent_pi[0]
    mask = torch.ones_like(first.weight)
    mask[:, ROBUST_INACTIVE_COLUMNS] = 0
    return first.weight.register_hook(lambda gradient: gradient * mask)


def _bc_objective(model, observations, targets, *, refinement=True, rotation_weight=None):
    import torch

    if model.bc_mode == "classic":
        prediction = model.actor(observations, deterministic=True)
        objective = torch.nn.functional.mse_loss(prediction, targets)
        return objective, objective
    mean, _, _ = model.actor.get_action_dist_params(observations)
    rotation = 0.3 if refinement else 1.0
    if rotation_weight is not None:
        rotation = rotation_weight
    weight = [1, 1, 1, rotation, rotation, rotation]
    pose = (mean[:, :6] - torch.atanh(targets[:, :6].clamp(-0.999, 0.999))).square()
    pose = (pose * torch.tensor(weight, device=model.device)).mean()
    grip = (mean[:, 6] - 3 * targets[:, 6]).square().mean()
    action_error = (torch.tanh(mean) - targets).square().mean()
    return pose + 0.02 * grip, action_error


def _robust_batch(model, batch_size, generator, *, refinement):
    """Synthetic terminal holds supervise only the actor, never critic replay."""
    import torch

    hold_count = batch_size // 4 if refinement else 0
    boundary_count = batch_size // 4 if refinement else batch_size // 2
    ordinary_count = batch_size - hold_count - boundary_count
    indices = np.r_[
        generator.integers(len(model.bc_observations), size=ordinary_count),
        generator.choice(model.bc_boundary_indices, size=boundary_count),
    ]
    observations = model.bc_observations[indices].copy()
    targets = model.bc_actions[indices].copy()
    if hold_count:
        choices = generator.integers(len(model.bc_terminal_observations), size=hold_count)
        holds = model.bc_terminal_observations[choices].copy()
        holds[:, -1] *= generator.uniform(0, 1, size=hold_count)
        observations = np.concatenate((observations, holds))
        targets = np.concatenate((targets, model.bc_terminal_actions[choices]))
    observations[:, ROBUST_INACTIVE_COLUMNS] = 0
    if refinement:
        observations[:, 32:35] += generator.normal(0, 0.003, size=(batch_size, 3))
    return (torch.as_tensor(observations, device=model.device),
            torch.as_tensor(targets, device=model.device))


def _dataset(data, *, observation_shape=None, action_shape=None):
    required = ("obs", "actions", "next_obs", "rewards", "terminated", "truncated")
    result = {key: np.asarray(data[key], dtype=np.float32).copy() for key in required}
    count = len(result["obs"])
    if count == 0 or any(len(value) != count for value in result.values()):
        raise ValueError("A demonstration dataset needs equally sized, nonempty arrays")
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Demonstrations must contain finite values")
    if result["obs"].ndim != 2 or result["actions"].ndim != 2:
        raise ValueError("Demonstration observations and actions must be matrices")
    if result["next_obs"].shape != result["obs"].shape:
        raise ValueError("Current and next observation shapes must agree")
    if observation_shape is not None and result["obs"].shape[1:] != observation_shape:
        raise ValueError("Demonstration observation shape differs from the environment")
    if action_shape is not None and result["actions"].shape[1:] != action_shape:
        raise ValueError("Demonstration action shape differs from the environment")
    if np.any(np.abs(result["actions"]) > 1.000001):
        raise ValueError("Demonstration actions must be normalized to [-1, 1]")
    for name in ("rewards", "terminated", "truncated"):
        result[name] = result[name].reshape(count)
    for name in ("terminated", "truncated"):
        if not np.isin(result[name], (0, 1)).all():
            raise ValueError(f"{name} must contain boolean episode-end flags")
    return result


@functools.lru_cache(maxsize=1)
def _learning_types():
    """Construct optional subclasses lazily, with stable names for SB3/pickle."""
    try:
        import torch
        from stable_baselines3 import SAC
        from stable_baselines3.common.buffers import ReplayBuffer
        from stable_baselines3.common.type_aliases import ReplayBufferSamples
    except ImportError as exc:
        raise ImportError(
            "RL learning needs the optional .venv-rl environment (Torch and Stable Baselines3)"
        ) from exc

    class DemonstrationReplayBuffer(ReplayBuffer):
        """An ordinary online ring buffer plus a permanent demonstration partition."""

        def __init__(self, *args, demonstration_data=None, demo_fraction=0.25, **kwargs):
            if not 0 < demo_fraction < 1:
                raise ValueError("demo_fraction must be strictly between zero and one")
            # The 500-step limit belongs to the task, so it also ends bootstrapping.
            kwargs["handle_timeout_termination"] = False
            super().__init__(*args, **kwargs)
            self.demo_fraction = float(demo_fraction)
            self.demonstrations = _dataset(
                demonstration_data,
                observation_shape=self.obs_shape,
                action_shape=(self.action_dim,),
            )
            self.last_demo_samples = 0
            self.last_online_samples = 0

        @property
        def demonstration_count(self):
            return len(self.demonstrations["obs"])

        def _sample_demonstrations(self, count, env=None):
            data = self.demonstrations
            indices = np.random.randint(self.demonstration_count, size=count)
            ended = np.maximum(data["terminated"][indices], data["truncated"][indices])
            return ReplayBufferSamples(
                observations=self.to_torch(self._normalize_obs(data["obs"][indices], env)),
                actions=self.to_torch(data["actions"][indices]),
                next_observations=self.to_torch(
                    self._normalize_obs(data["next_obs"][indices], env)
                ),
                dones=self.to_torch(ended[:, None]),
                rewards=self.to_torch(
                    self._normalize_reward(data["rewards"][indices, None], env)
                ),
            )

        def sample(self, batch_size, env=None):
            if batch_size < 1:
                raise ValueError("batch_size must be positive")
            count = max(1, round(batch_size * self.demo_fraction))
            if self.size() == 0:
                count = batch_size
            demonstrations = self._sample_demonstrations(count, env)
            self.last_demo_samples = count
            self.last_online_samples = batch_size - count
            if count == batch_size:
                return demonstrations
            online = super().sample(batch_size - count, env)
            fields = {}
            for name in ("observations", "actions", "next_observations", "dones", "rewards"):
                fields[name] = torch.cat((getattr(demonstrations, name), getattr(online, name)))
            return ReplayBufferSamples(**fields)

    class SACFromDemonstrations(SAC):
        """SAC's own optimizer update augmented with successful-demo BC gradients.

        A temporary optimizer hook adds the auxiliary gradient immediately before
        each actor update. The SAC targets, entropy objective, critic optimizers,
        and target-network updates remain owned by Stable Baselines3.
        """

        def __init__(self, *args, successful_data=None, bc_weight=10.0,
                     bc_final_weight=1.0, bc_decay_steps=50_000, bc_mode="classic",
                     initial_log_std=-4.0, actor_learning_rate=None,
                     actor_log_std_max=None, bc_online_rotation_weight=0.3, **kwargs):
            if actor_log_std_max is not None and (
                not np.isfinite(actor_log_std_max) or not -20 <= actor_log_std_max <= 2
            ):
                raise ValueError("actor_log_std_max must be None or finite in [-20, 2]")
            self.actor_log_std_max = actor_log_std_max
            self._actor_log_std_hook = None
            super().__init__(*args, **kwargs)
            if bc_mode not in {"classic", "robust"}:
                raise ValueError("bc_mode must be classic or robust")
            if not np.isfinite(initial_log_std) or not -20 <= initial_log_std <= 2:
                raise ValueError("initial_log_std must be in [-20, 2]")
            self.bc_mode = bc_mode
            self.initial_log_std = float(initial_log_std)
            if actor_learning_rate is not None and (
                not np.isfinite(actor_learning_rate) or actor_learning_rate <= 0
            ):
                raise ValueError("actor_learning_rate must be None or finite and positive")
            self.actor_learning_rate = actor_learning_rate
            if not np.isfinite(bc_online_rotation_weight) or bc_online_rotation_weight <= 0:
                raise ValueError("bc_online_rotation_weight must be finite and positive")
            self.bc_online_rotation_weight = float(bc_online_rotation_weight)
            self.bc_weight = float(bc_weight)
            self.bc_final_weight = float(bc_final_weight)
            self.bc_decay_steps = int(bc_decay_steps)
            if (not np.isfinite([self.bc_weight, self.bc_final_weight]).all()
                    or min(self.bc_weight, self.bc_final_weight) < 0
                    or self.bc_decay_steps < 1):
                raise ValueError("BC weights must be finite/nonnegative and decay steps positive")
            self.bc_observations = None
            self.bc_actions = None
            self.bc_terminal_observations = self.bc_terminal_actions = None
            self.bc_boundary_indices = None
            self.bc_generator = np.random.default_rng(self.seed)
            self.last_train_metrics = {}
            if successful_data is not None:
                self.set_successful_demonstrations(successful_data)

        def _setup_model(self):
            super()._setup_model()
            # SB3 restores saved attributes before calling _setup_model on load.
            # Recreate this bound there; module forward hooks are not state_dict data.
            if self._actor_log_std_hook is not None:
                self._actor_log_std_hook.remove()
                self._actor_log_std_hook = None
            maximum = self.actor_log_std_max
            if maximum is not None:
                self._actor_log_std_hook = self.actor.log_std.register_forward_hook(
                    lambda _module, _inputs, output: output.clamp(max=maximum),
                )

        def _excluded_save_params(self):
            return [*super()._excluded_save_params(), "_actor_log_std_hook"]

        def set_successful_demonstrations(self, data):
            checked = _dataset(
                data, observation_shape=self.observation_space.shape,
                action_shape=self.action_space.shape,
            )
            if "episode_success" in data and not np.asarray(data["episode_success"]).all():
                raise ValueError("Failed trajectories must not supervise the actor")
            self.bc_observations = checked["obs"]
            self.bc_actions = checked["actions"]
            if self.bc_mode == "robust":
                self.bc_boundary_indices = demonstration_boundaries(
                    self.bc_observations, self.bc_actions,
                )
                terminal = np.flatnonzero(
                    checked["terminated"].astype(bool) | checked["truncated"].astype(bool)
                )
                if len(terminal) == 0:
                    raise ValueError("Robust BC needs terminal observations of successful episodes")
                self.bc_terminal_observations = checked["next_obs"][terminal].copy()
                self.bc_terminal_actions = checked["actions"][terminal].copy()

        def current_bc_weight(self):
            progress = min(1.0, max(0.0, self.num_timesteps / self.bc_decay_steps))
            return self.bc_weight + progress * (self.bc_final_weight - self.bc_weight)

        def _update_learning_rate(self, optimizers):
            super()._update_learning_rate(optimizers)
            # SAC otherwise resets every optimizer to its global learning rate on
            # every train() call, defeating an actor-only adjustment made elsewhere.
            rate = getattr(self, "actor_learning_rate", None)
            if rate is not None:
                for group in self.actor.optimizer.param_groups:
                    group["lr"] = rate
            self.logger.record(
                "train/actor_learning_rate", self.actor.optimizer.param_groups[0]["lr"],
            )
            self.logger.record(
                "train/critic_learning_rate", self.critic.optimizer.param_groups[0]["lr"],
            )

        def train(self, gradient_steps, batch_size=64):
            if self.bc_observations is None or self.bc_actions is None:
                raise RuntimeError("SAC training requires the successful demonstration partition")
            losses, objectives = [], []
            weight = self.current_bc_weight()

            def auxiliary_gradient(optimizer, args, kwargs):
                if self.bc_mode == "robust":
                    observations, targets = _robust_batch(
                        self, batch_size, self.bc_generator, refinement=True,
                    )
                else:
                    indices = np.random.randint(len(self.bc_observations), size=batch_size)
                    observations = torch.as_tensor(
                        self.bc_observations[indices], device=self.device,
                    )
                    targets = torch.as_tensor(self.bc_actions[indices], device=self.device)
                loss, action_error = _bc_objective(
                    self, observations, targets, rotation_weight=self.bc_online_rotation_weight,
                )
                losses.append(float(action_error.detach()))
                objectives.append(float(loss.detach()))
                # SAC has already populated actor gradients. Add, do not replace them.
                (weight * loss).backward()

            hook = self.actor.optimizer.register_step_pre_hook(auxiliary_gradient)
            gradient_mask = _actor_gradient_mask(self)
            try:
                super().train(gradient_steps, batch_size)
            finally:
                hook.remove()
                if gradient_mask is not None:
                    gradient_mask.remove()
            if losses:
                self.logger.record("train/bc_loss", float(np.mean(losses)))
                self.logger.record("train/bc_objective", float(np.mean(objectives)))
                self.logger.record("train/bc_weight", weight)
                self.logger.record(
                    "train/bc_online_rotation_weight", self.bc_online_rotation_weight,
                )
            self.last_train_metrics = {
                key.removeprefix("train/"): float(value)
                for key, value in self.logger.name_to_value.items()
                if key.startswith("train/") and isinstance(value, (int, float, np.number))
            }

    # Replay buffers are pickled by SB3. Stable module names avoid closures there.
    for cls in (DemonstrationReplayBuffer, SACFromDemonstrations):
        cls.__module__ = __name__
        cls.__qualname__ = cls.__name__
        globals()[cls.__name__] = cls
    return DemonstrationReplayBuffer, SACFromDemonstrations


def __getattr__(name):
    if name in {"DemonstrationReplayBuffer", "SACFromDemonstrations"}:
        _learning_types()
        return globals()[name]
    raise AttributeError(name)


def create_model(env, demonstration_data, successful_data, *, seed=0,
                 bc_mode="classic", initial_log_std=-4.0, actor_learning_rate=None,
                 actor_log_std_max=None, bc_online_rotation_weight=0.3, **overrides):
    """Create a CPU SAC learner; demonstrations do not count as online timesteps."""
    import torch

    replay_class, model_class = _learning_types()
    if not (np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)):
        raise ValueError("The policy requires a normalized [-1, 1] action space")
    settings = {
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "learning_starts": 0,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
        "ent_coef": 0.001,
        "policy_kwargs": {"net_arch": [256, 256]},
        "device": "cpu",
        "verbose": 0,
    }
    settings.update(overrides)
    if settings["learning_starts"] != 0:
        raise ValueError("Random-action warmup would discard the pretrained actor's behavior")
    if settings.get("use_sde", False):
        raise ValueError("This learner uses a squashed Gaussian actor, not gSDE")
    settings["replay_buffer_class"] = replay_class
    settings["replay_buffer_kwargs"] = {
        "demonstration_data": demonstration_data,
        "demo_fraction": 0.25,
        "handle_timeout_termination": False,
    }
    model = model_class(
        "MlpPolicy", env, seed=seed, successful_data=successful_data,
        bc_mode=bc_mode, initial_log_std=initial_log_std,
        actor_learning_rate=actor_learning_rate, actor_log_std_max=actor_log_std_max,
        bc_online_rotation_weight=bc_online_rotation_weight, **settings,
    )
    # SB3's log_std_init applies to gSDE only; initialize the ordinary head directly.
    with torch.no_grad():
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(model.initial_log_std)
        if bc_mode == "robust":
            first = model.actor.latent_pi[0]
            first.weight[:, ROBUST_INACTIVE_COLUMNS] = 0
            clock = model.bc_observations[:, -1]
            # Generic time basis across the observed data range, not task waypoints.
            knots = torch.linspace(float(clock.min()) - 0.03, float(clock.max()) + 0.01,
                                   first.out_features, device=model.device)
            first.weight[:, -1] = torch.where(
                torch.arange(first.out_features, device=model.device) % 2 == 0, 200.0, -200.0,
            )
            first.bias.copy_(-knots * first.weight[:, -1])
            model.actor.mu.weight.mul_(0.01)
            model.actor.mu.bias.zero_()
    return model


def bc_pretrain(model, observations, actions, *, updates=20_000, batch_size=256,
                callback=None, report_every=100, seed=0):
    """Fit the deterministic actor mean; callback(metrics) may return False to stop.

    The caller must pass successful demonstrations only. Callbacks run before
    training and every ``report_every`` updates, allowing a wall-time deadline.
    """
    import torch

    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if (observations.ndim != 2 or actions.ndim != 2 or len(observations) == 0
            or len(observations) != len(actions)
            or observations.shape[1:] != model.observation_space.shape
            or actions.shape[1:] != model.action_space.shape
            or not np.isfinite(observations).all() or not np.isfinite(actions).all()
            or np.any(np.abs(actions) > 1.000001)):
        raise ValueError("BC requires matching finite observation/action matrices")
    if updates < 1 or batch_size < 1 or report_every < 1:
        raise ValueError("BC update, batch, and reporting counts must be positive")
    if model.bc_mode == "robust" and (
        not np.array_equal(observations, model.bc_observations)
        or not np.array_equal(actions, model.bc_actions)
    ):
        raise ValueError("Robust BC must use the model's successful demonstration partition")
    model.policy.set_training_mode(True)
    generator = np.random.default_rng(seed)
    observation_tensor = torch.as_tensor(observations, device=model.device)
    action_tensor = torch.as_tensor(actions, device=model.device)
    metrics = {"updates": 0, "bc_loss": None}
    if callback is not None and callback(dict(metrics)) is False:
        return metrics
    recent, objectives = [], []
    gradient_mask = _actor_gradient_mask(model)
    try:
        for update in range(1, updates + 1):
            refinement = model.bc_mode == "robust" and update > 20_000
            if model.bc_mode == "robust":
                if update == 20_001:
                    model.actor.optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-4)
                if refinement:
                    fraction = (update - 20_000) / max(1, updates - 20_000)
                    for group in model.actor.optimizer.param_groups:
                        group["lr"] = 1e-5 + 9e-5 * (1 - fraction)
                batch_observations, targets = _robust_batch(
                    model, batch_size, generator, refinement=refinement,
                )
            else:
                indices = generator.integers(len(observations), size=batch_size)
                batch_observations, targets = observation_tensor[indices], action_tensor[indices]
            loss, action_error = _bc_objective(
                model, batch_observations, targets, refinement=refinement,
            )
            model.actor.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.actor.optimizer.step()
            recent.append(float(action_error.detach()))
            objectives.append(float(loss.detach()))
            if update % report_every == 0 or update == updates:
                metrics = {
                    "updates": update, "bc_loss": float(np.mean(recent)),
                    "bc_objective": float(np.mean(objectives)), "bc_mode": model.bc_mode,
                    "stage": "refinement" if refinement else "initialization",
                }
                recent.clear()
                objectives.clear()
                if callback is not None and callback(dict(metrics)) is False:
                    break
    finally:
        if gradient_mask is not None:
            gradient_mask.remove()
    # Retain small exploration variance even if pretraining optimizer state is reused.
    with torch.no_grad():
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(model.initial_log_std)
    return metrics


def load_model(path, *, env=None):
    """Restore a full training checkpoint; replay is loaded separately via SB3."""
    _, model_class = _learning_types()
    return model_class.load(path, env=env, device="cpu")


def export_policy(model, path, metadata=None):
    """Write only the deterministic actor and its inference contract, no teacher."""
    import torch

    payload = {
        "format": POLICY_FORMAT,
        "observation_dim": int(np.prod(model.observation_space.shape)),
        "action_dim": int(np.prod(model.action_space.shape)),
        "hidden_sizes": list(model.actor.net_arch),
        "latent_state": {
            key: value.detach().cpu().clone()
            for key, value in model.actor.latent_pi.state_dict().items()
        },
        "mean_state": {
            key: value.detach().cpu().clone()
            for key, value in model.actor.mu.state_dict().items()
        },
        "online_steps": int(model.num_timesteps),
        "rl_updates": int(model._n_updates),
        "metadata": {
            **dict(metadata or {}),
            "learner": {
                "bc_mode": model.bc_mode, "initial_log_std": model.initial_log_std,
                "actor_learning_rate": model.actor_learning_rate,
                "actor_log_std_max": model.actor_log_std_max,
                "bc_weight": model.bc_weight, "bc_final_weight": model.bc_final_weight,
                "bc_decay_steps": model.bc_decay_steps,
                "bc_online_rotation_weight": model.bc_online_rotation_weight,
                "masked_actor_columns": (
                    list(ROBUST_INACTIVE_COLUMNS) if model.bc_mode == "robust" else []
                ),
                "robust_recipe": (
                    "clock_hinges_auto_boundaries_logit_loss_terminal_hold_v1"
                    if model.bc_mode == "robust" else None
                ),
                "synthetic_hold_supervision": (
                    "actor_only_not_replay" if model.bc_mode == "robust" else None
                ),
            },
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


class LoadedPolicy:
    """Small CPU actor loader; needs Torch but neither SB3 nor a teacher archive."""

    def __init__(self, payload):
        import torch

        if payload["format"] != POLICY_FORMAT:
            raise ValueError("Unknown dynamics RL policy artifact format")
        self.metadata = payload["metadata"]
        self.online_steps = payload["online_steps"]
        self.rl_updates = payload["rl_updates"]
        self.observation_dim = payload["observation_dim"]
        self.action_dim = payload["action_dim"]
        layers = []
        width = self.observation_dim
        for hidden in payload["hidden_sizes"]:
            layers.extend((torch.nn.Linear(width, hidden), torch.nn.ReLU()))
            width = hidden
        self.latent = torch.nn.Sequential(*layers)
        self.mean = torch.nn.Linear(width, self.action_dim)
        self.latent.load_state_dict(payload["latent_state"])
        self.mean.load_state_dict(payload["mean_state"])
        self.latent.eval()
        self.mean.eval()

    def predict(self, observations, deterministic=True):
        import torch

        if not deterministic:
            raise ValueError("The exported actor has deterministic inference only")
        observations = np.asarray(observations, dtype=np.float32)
        single = observations.ndim == 1
        if single:
            observations = observations[None]
        if (observations.ndim != 2 or observations.shape[1] != self.observation_dim
                or not np.isfinite(observations).all()):
            raise ValueError("Policy input has the wrong shape or nonfinite values")
        with torch.inference_mode():
            features = self.latent(torch.from_numpy(observations))
            result = torch.tanh(self.mean(features)).numpy()
        return (result[0] if single else result), None


def load_policy(path):
    import torch

    return LoadedPolicy(torch.load(path, map_location="cpu", weights_only=True))
