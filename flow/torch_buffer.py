"""Torch-native rollout, replay, normalization, and GAE utilities.

The standalone Gymnasium entry point uses the NumPy helpers in :mod:`flow.buffer`.
SRB already keeps thousands of vectorized environments on one Torch device, so
moving every transition through NumPy would add a device synchronization and two
copies per step.  This module provides the same data contract without leaving the
policy device.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass
class TorchRollout:
    """A flattened rollout containing the exact variables used by ExO ratios."""

    actor_observations: torch.Tensor
    critic_observations: torch.Tensor
    pre_tanh_actions: torch.Tensor
    flow_init: torch.Tensor
    flow_start: torch.Tensor
    behavior_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __post_init__(self) -> None:
        sample_count = self.actor_observations.shape[0]
        for name, value in vars(self).items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.shape[0] != sample_count:
                raise ValueError(
                    f"{name} has {value.shape[0]} samples; expected {sample_count}"
                )

    def __len__(self) -> int:
        return int(self.actor_observations.shape[0])

    def detached(self) -> TorchRollout:
        """Return a graph-free rollout suitable for a bounded replay window."""

        return TorchRollout(
            **{name: value.detach() for name, value in vars(self).items()}
        )


def flatten_torch_rollout(
    actor_observations: torch.Tensor,
    critic_observations: torch.Tensor,
    pre_tanh_actions: torch.Tensor,
    flow_init: torch.Tensor,
    flow_start: torch.Tensor,
    behavior_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
) -> TorchRollout:
    """Flatten leading ``[time, environments]`` dimensions in stable order."""

    if actor_observations.ndim < 3:
        raise ValueError("actor_observations must have shape [T, N, ...]")
    time_steps, num_envs = actor_observations.shape[:2]
    count = time_steps * num_envs
    values = {
        "actor_observations": actor_observations,
        "critic_observations": critic_observations,
        "pre_tanh_actions": pre_tanh_actions,
        "flow_init": flow_init,
        "flow_start": flow_start,
        "behavior_log_prob": behavior_log_prob,
        "advantages": advantages,
        "returns": returns,
    }
    for name, value in values.items():
        if value.shape[:2] != (time_steps, num_envs):
            raise ValueError(
                f"{name} has leading shape {tuple(value.shape[:2])}; "
                f"expected {(time_steps, num_envs)}"
            )
    return TorchRollout(
        **{
            name: value.reshape(count, *value.shape[2:])
            for name, value in values.items()
        }
    )


class TorchReplayWindow:
    """A bounded, device-resident window of complete policy rollouts."""

    _FIELDS = tuple(TorchRollout.__dataclass_fields__)

    def __init__(self, max_rollouts: int) -> None:
        if max_rollouts < 1:
            raise ValueError("max_rollouts must be positive")
        self._rollouts: deque[TorchRollout] = deque(maxlen=max_rollouts)

    @property
    def max_rollouts(self) -> int:
        return int(self._rollouts.maxlen or 0)

    @property
    def rollout_count(self) -> int:
        return len(self._rollouts)

    def __len__(self) -> int:
        return sum(len(rollout) for rollout in self._rollouts)

    def append(self, rollout: TorchRollout) -> None:
        self._rollouts.append(rollout.detached())

    def tensors(self) -> dict[str, torch.Tensor]:
        if not self._rollouts:
            raise RuntimeError("cannot concatenate an empty replay window")
        return {
            field: torch.cat(
                [getattr(rollout, field) for rollout in self._rollouts], dim=0
            )
            for field in self._FIELDS
        }


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    *,
    bootstrap_truncated: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute timeout-aware vectorized GAE for ``[time, environments]`` data.

    GAE recursion always stops at either kind of episode boundary.  A truncated
    transition may still bootstrap from its terminal observation when the task
    models an infinite horizon, which is the convention used by Isaac Lab.
    """

    shape = rewards.shape
    if rewards.ndim != 2:
        raise ValueError("GAE tensors must have shape [time, environments]")
    for name, value in {
        "terminated": terminated,
        "truncated": truncated,
        "values": values,
        "next_values": next_values,
    }.items():
        if value.shape != shape:
            raise ValueError(f"{name} has shape {tuple(value.shape)}; expected {shape}")
    if not 0.0 < gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in their valid ranges")

    terminated = terminated.to(dtype=torch.bool)
    truncated = truncated.to(dtype=torch.bool)
    episode_done = terminated | truncated
    bootstrap_mask = ~terminated
    if not bootstrap_truncated:
        bootstrap_mask &= ~truncated

    deltas = (
        rewards + gamma * next_values * bootstrap_mask.to(dtype=rewards.dtype) - values
    )
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rewards[0])
    for time_index in range(rewards.shape[0] - 1, -1, -1):
        continuation = (~episode_done[time_index]).to(dtype=rewards.dtype)
        last_advantage = (
            deltas[time_index] + gamma * gae_lambda * continuation * last_advantage
        )
        advantages[time_index] = last_advantage
    return advantages, advantages + values


class TorchRunningMeanStd:
    """Numerically stable running moments stored on a Torch device."""

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cpu",
        epsilon: float = 1e-4,
    ) -> None:
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = torch.as_tensor(epsilon, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = torch.as_tensor(values, dtype=torch.float64, device=self.mean.device)
        if values.numel() == 0:
            return
        if values.shape[1:] != self.mean.shape:
            raise ValueError(
                f"values have sample shape {tuple(values.shape[1:])}; "
                f"expected {tuple(self.mean.shape)}"
            )
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = torch.as_tensor(
            values.shape[0], dtype=torch.float64, device=self.mean.device
        )

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        old_m2 = self.var * self.count
        batch_m2 = batch_var * batch_count
        new_m2 = (
            old_m2 + batch_m2 + delta.square() * self.count * batch_count / total_count
        )
        self.mean.copy_(new_mean)
        self.var.copy_(new_m2 / total_count)
        self.count.copy_(total_count)

    def normalize(self, values: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        if clip <= 0.0:
            raise ValueError("clip must be positive")
        values = torch.as_tensor(values, dtype=torch.float32, device=self.mean.device)
        mean = self.mean.to(dtype=torch.float32)
        variance = self.var.to(dtype=torch.float32)
        normalized = (values - mean) * torch.rsqrt(variance + 1e-8)
        return normalized.clamp(-clip, clip)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "mean": self.mean.detach().clone(),
            "var": self.var.detach().clone(),
            "count": self.count.detach().clone(),
        }

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        for name in ("mean", "var", "count"):
            if name not in state_dict:
                raise KeyError(f"normalizer state is missing {name!r}")
        mean = torch.as_tensor(
            state_dict["mean"], dtype=torch.float64, device=self.mean.device
        )
        var = torch.as_tensor(
            state_dict["var"], dtype=torch.float64, device=self.var.device
        )
        count = torch.as_tensor(
            state_dict["count"], dtype=torch.float64, device=self.count.device
        )
        if mean.shape != self.mean.shape or var.shape != self.var.shape:
            raise ValueError("normalizer state shape does not match this normalizer")
        if count.numel() != 1:
            raise ValueError("normalizer count must be scalar")
        self.mean.copy_(mean)
        self.var.copy_(var)
        self.count.copy_(count.reshape(()))
