"""Fail visibly on non-finite PPO values without repairing or hiding samples."""

from pathlib import Path
import json

import torch


def require_finite(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name}")


def install_numerics_guard(runner, log_dir):
    """Check network boundaries and gradients before Adam can corrupt parameters."""
    directory = Path(log_dir)

    def check(name, value):
        try:
            require_finite(name, value)
        except FloatingPointError:
            # Preserve the actual failed rollout and parameters for diagnosis.
            torch.save({
                "tensor_name": name, "tensor": value.detach().cpu(),
                "model_state_dict": runner.alg.policy.state_dict(),
                "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                "rollout": {k: v for k, v in vars(runner.alg.storage).items() if torch.is_tensor(v)},
            }, directory / "numerics_failure.pt")
            (directory / "numerics_failure.json").write_text(json.dumps({
                "tensor": name, "shape": list(value.shape),
                "last_completed_iteration": runner.current_learning_iteration,
                "nonfinite_count": (~torch.isfinite(value)).sum().item(),
            }, indent=2) + "\n")
            raise

    for name in ("actor", "critic"):
        module = getattr(runner.alg.policy, name)
        module.register_forward_pre_hook(lambda module, args, name=name: check(name + " input", args[0]))
        module.register_forward_hook(lambda module, args, out, name=name: check(name + " output", out))

    def before_step(optimizer, args, kwargs):
        gradients = [p.grad for p in runner.alg.policy.parameters() if p.grad is not None]
        # A single device-to-host check for all gradient tensors.
        finite = torch.stack([torch.isfinite(g).all() for g in gradients])
        if not finite.all():
            for name, parameter in runner.alg.policy.named_parameters():
                if parameter.grad is not None:
                    check(name + " gradient", parameter.grad)

    runner.alg.optimizer.register_step_pre_hook(before_step)
