"""Rollout replay and normalization utilities for the flow experiment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass
class Rollout:
    """A flattened rollout with all variables required by direct ratios."""

    observations: np.ndarray
    pre_tanh_actions: np.ndarray
    flow_init: np.ndarray
    flow_start: np.ndarray
    behavior_log_prob: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray

    def __len__(self) -> int:
        return int(self.observations.shape[0])


def generalized_advantage_estimate(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute vectorized GAE for arrays shaped ``[time, environments]``."""

    if rewards.shape != dones.shape or rewards.shape != values.shape:
        raise ValueError("rewards, dones and values must have identical shapes")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    next_value = np.asarray(next_value, dtype=np.float32)

    for time_index in range(rewards.shape[0] - 1, -1, -1):
        not_done = 1.0 - dones[time_index].astype(np.float32)
        delta = rewards[time_index] + gamma * next_value * not_done - values[time_index]
        last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
        advantages[time_index] = last_advantage
        next_value = values[time_index]
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def flatten_rollout(
    observations: np.ndarray,
    pre_tanh_actions: np.ndarray,
    flow_init: np.ndarray,
    flow_start: np.ndarray,
    behavior_log_prob: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
) -> Rollout:
    """Flatten the leading time/environment dimensions in a stable order."""

    time_steps, num_envs = observations.shape[:2]
    count = time_steps * num_envs
    return Rollout(
        observations=observations.reshape(count, *observations.shape[2:]),
        pre_tanh_actions=pre_tanh_actions.reshape(count, *pre_tanh_actions.shape[2:]),
        flow_init=flow_init.reshape(count, *flow_init.shape[2:]),
        flow_start=flow_start.reshape(count, 1),
        behavior_log_prob=behavior_log_prob.reshape(count),
        advantages=advantages.reshape(count),
        returns=returns.reshape(count),
    )


class ReplayWindow:
    """A bounded window of complete policy rollouts."""

    _FIELDS = (
        "observations",
        "pre_tanh_actions",
        "flow_init",
        "flow_start",
        "behavior_log_prob",
        "advantages",
        "returns",
    )

    def __init__(self, max_rollouts: int) -> None:
        if max_rollouts < 1:
            raise ValueError("max_rollouts must be positive")
        self._rollouts: Deque[Rollout] = deque(maxlen=max_rollouts)

    @property
    def max_rollouts(self) -> int:
        return int(self._rollouts.maxlen or 0)

    def __len__(self) -> int:
        return sum(len(rollout) for rollout in self._rollouts)

    @property
    def rollout_count(self) -> int:
        return len(self._rollouts)

    def append(self, rollout: Rollout) -> None:
        self._rollouts.append(rollout)

    def arrays(self) -> dict[str, np.ndarray]:
        if not self._rollouts:
            raise RuntimeError("cannot concatenate an empty replay window")
        return {
            field: np.concatenate(
                [getattr(rollout, field) for rollout in self._rollouts], axis=0
            )
            for field in self._FIELDS
        }


class RunningMeanStd:
    """Numerically stable running moments for vector observations."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        batch_mean = np.mean(values, axis=0)
        batch_var = np.var(values, axis=0)
        batch_count = values.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        old_m2 = self.var * self.count
        batch_m2 = batch_var * batch_count
        new_m2 = (
            old_m2
            + batch_m2
            + np.square(delta) * self.count * batch_count / total_count
        )
        self.mean = new_mean
        self.var = new_m2 / total_count
        self.count = float(total_count)

    def normalize(self, values: np.ndarray, clip: float = 10.0) -> np.ndarray:
        normalized = (np.asarray(values) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip, clip).astype(np.float32)
