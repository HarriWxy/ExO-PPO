"""Direct-ratio ExO-PPO with a recent-policy one-step flow actor.

The package is intentionally separate from the original single-file experiments in
``Mujoco/``.  Use ``python -m flow.train`` for TensorFlow or
``python -m flow.torch_train`` for PyTorch.
"""

__all__ = [
    "buffer",
    "models",
    "objectives",
    "torch_models",
    "torch_objectives",
    "torch_train",
]
