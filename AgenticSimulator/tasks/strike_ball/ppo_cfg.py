"""RSL-RL PPO configuration for the UR10 strike task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class StrikeBallPpoCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "strike_ball"
    action_repeat = 1
    num_steps_per_env = 48
    max_iterations = 4000
    save_interval = 200
    empirical_normalization = True
    clip_actions = 10.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0, noise_std_type="log", actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64], activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.0,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=3e-4, schedule="fixed",
        gamma=0.995, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    )
