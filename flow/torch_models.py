"""PyTorch one-step flow policy for direct-ratio ExO-PPO.

The policy keeps the flow latent in the sampled record so that the residual
Gaussian likelihood is tractable.  The environment only sees the transformed
action; importance ratios are evaluated in this augmented ``(z, x)`` space.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn


LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass
class PolicySample:
    """A sampled action together with the variables needed for replay ratios."""

    pre_tanh_action: torch.Tensor
    flow_init: torch.Tensor
    flow_start: torch.Tensor
    log_prob: torch.Tensor
    mean: torch.Tensor


def _batch_column(
    value: torch.Tensor | float,
    batch_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Convert a scalar, ``[B]`` or ``[B, 1]`` time tensor to ``[B, 1]``."""

    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.numel() == 1:
        return tensor.reshape(1, 1).expand(batch_size, 1)
    if tensor.numel() != batch_size:
        raise ValueError(
            f"time tensor has {tensor.numel()} elements; expected {batch_size}"
        )
    return tensor.reshape(batch_size, 1)


class IntervalFlowPolicy(nn.Module):
    """Observation-conditioned interval-average velocity policy.

    ``velocity(observation, state, t, r)`` predicts the average velocity over
    ``[t, r]``.  A single evaluation at ``r=1`` therefore gives the one-step
    terminal mean ``state + (1-t) * velocity``.  A learned null context is
    exposed through ``condition_mask`` for OFP self-guidance.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        initial_log_std: float = -1.0,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        self.initial_log_std = float(initial_log_std)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        context_layers: list[nn.Module] = []
        input_dim = self.obs_dim
        for size in self.hidden_sizes:
            context_layers.append(nn.Sequential(nn.Linear(input_dim, size), nn.SiLU()))
            input_dim = size
        self.context_layers = nn.ModuleList(context_layers)
        self.null_context = nn.Parameter(torch.zeros(self.hidden_sizes[-1]))

        # t, r, r-t and sinusoidal features for t and r.
        time_feature_dim = 7
        flow_layers: list[nn.Module] = []
        input_dim = self.action_dim + time_feature_dim + self.hidden_sizes[-1]
        for size in self.hidden_sizes:
            flow_layers.append(nn.Sequential(nn.Linear(input_dim, size), nn.SiLU()))
            input_dim = size
        self.flow_layers = nn.ModuleList(flow_layers)
        self.velocity_head = nn.Linear(input_dim, self.action_dim)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), self.initial_log_std)
        )

    def _context(
        self,
        observation: torch.Tensor,
        condition_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        context = observation.to(dtype=torch.float32)
        for layer in self.context_layers:
            context = layer(context)
        batch_size = context.shape[0]
        if condition_mask is None:
            mask = torch.ones(
                (batch_size, 1), dtype=context.dtype, device=context.device
            )
        else:
            mask = torch.as_tensor(
                condition_mask, dtype=context.dtype, device=context.device
            ).reshape(-1, 1)
            if mask.shape[0] == 1 and batch_size != 1:
                mask = mask.expand(batch_size, 1)
            if mask.shape[0] != batch_size:
                raise ValueError(
                    f"condition_mask has batch {mask.shape[0]}; expected {batch_size}"
                )
        null_context = self.null_context.to(context).reshape(1, -1)
        return mask * context + (1.0 - mask) * null_context

    def velocity(
        self,
        observation: torch.Tensor,
        state: torch.Tensor,
        start_time: torch.Tensor | float,
        end_time: torch.Tensor | float,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict ``u_theta(state, start_time, end_time | observation)``."""

        observation = torch.as_tensor(observation, dtype=torch.float32)
        state = torch.as_tensor(state, dtype=torch.float32, device=observation.device)
        batch_size = state.shape[0]
        start = _batch_column(
            start_time,
            batch_size,
            dtype=state.dtype,
            device=state.device,
        )
        end = _batch_column(
            end_time,
            batch_size,
            dtype=state.dtype,
            device=state.device,
        )
        interval = end - start
        time_features = torch.cat(
            [
                start,
                end,
                interval,
                torch.sin(math.pi * start),
                torch.cos(math.pi * start),
                torch.sin(math.pi * end),
                torch.cos(math.pi * end),
            ],
            dim=-1,
        )
        context = self._context(observation, condition_mask)
        hidden = torch.cat([state, time_features, context], dim=-1)
        for layer in self.flow_layers:
            hidden = layer(hidden)
        return self.velocity_head(hidden)

    def interval_mean(
        self,
        observation: torch.Tensor,
        flow_init: torch.Tensor,
        flow_start: torch.Tensor | float,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate the terminal mean with one interval-flow evaluation."""

        flow_init = torch.as_tensor(flow_init, dtype=torch.float32)
        batch_size = flow_init.shape[0]
        start = _batch_column(
            flow_start,
            batch_size,
            dtype=flow_init.dtype,
            device=flow_init.device,
        )
        end = torch.ones_like(start)
        average_velocity = self.velocity(
            observation,
            flow_init,
            start,
            end,
            condition_mask=condition_mask,
        )
        return flow_init + (1.0 - start) * average_velocity

    def one_step_mean(
        self,
        observation: torch.Tensor,
        noise: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate ``epsilon + u(epsilon, 0, 1 | observation)``."""

        noise = torch.as_tensor(noise, dtype=torch.float32)
        batch_size = noise.shape[0]
        zeros = torch.zeros((batch_size, 1), dtype=noise.dtype, device=noise.device)
        return self.interval_mean(
            observation,
            noise,
            zeros,
            condition_mask=condition_mask,
        )

    def effective_log_std(self) -> torch.Tensor:
        return self.log_std.clamp(self.min_log_std, self.max_log_std)

    def std(self) -> torch.Tensor:
        return self.effective_log_std().exp()

    def conditional_log_prob(
        self,
        observation: torch.Tensor,
        pre_tanh_action: torch.Tensor,
        flow_init: torch.Tensor,
        flow_start: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate ``log p(pre_tanh_action | observation, flow_init)``."""

        mean = self.interval_mean(observation, flow_init, flow_start)
        return self.log_prob_from_mean(pre_tanh_action, mean)

    def log_prob_from_mean(
        self,
        pre_tanh_action: torch.Tensor,
        mean: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the residual Gaussian without another flow network call."""

        pre_tanh_action = torch.as_tensor(
            pre_tanh_action, dtype=torch.float32, device=mean.device
        )
        mean = torch.as_tensor(mean, dtype=torch.float32, device=mean.device)
        log_std = self.effective_log_std().to(mean)
        normalized = (pre_tanh_action - mean) * torch.exp(-log_std)
        per_dimension = -0.5 * (normalized.square() + 2.0 * log_std + LOG_TWO_PI)
        return per_dimension.sum(dim=-1)

    def conditional_entropy(self) -> torch.Tensor:
        """Entropy of the residual Gaussian conditional on the flow latent."""

        return (self.effective_log_std() + 0.5 * (1.0 + LOG_TWO_PI)).sum()

    def sample(
        self,
        observation: torch.Tensor,
        flow_init: torch.Tensor | None = None,
        flow_start: torch.Tensor | None = None,
        residual_noise: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> PolicySample:
        """Sample once and return every value needed to replay its ratio."""

        observation = torch.as_tensor(observation, dtype=torch.float32)
        batch_size = observation.shape[0]
        device = observation.device
        if flow_init is None:
            flow_init = torch.randn(
                (batch_size, self.action_dim), dtype=torch.float32, device=device
            )
        else:
            flow_init = torch.as_tensor(flow_init, dtype=torch.float32, device=device)
        if flow_start is None:
            flow_start = torch.zeros(
                (batch_size, 1), dtype=torch.float32, device=device
            )
        else:
            flow_start = _batch_column(
                flow_start,
                batch_size,
                dtype=torch.float32,
                device=device,
            )

        mean = self.interval_mean(observation, flow_init, flow_start)
        if deterministic:
            pre_tanh_action = mean
        else:
            if residual_noise is None:
                residual_noise = torch.randn_like(mean)
            else:
                residual_noise = torch.as_tensor(
                    residual_noise, dtype=mean.dtype, device=mean.device
                )
            pre_tanh_action = mean + self.std().to(mean)[None, :] * residual_noise
        log_prob = self.log_prob_from_mean(pre_tanh_action, mean)
        return PolicySample(
            pre_tanh_action=pre_tanh_action,
            flow_init=flow_init,
            flow_start=flow_start,
            log_prob=log_prob,
            mean=mean,
        )


class ValueNetwork(nn.Module):
    """Small state-value network for flat continuous-control observations."""

    def __init__(self, hidden_sizes: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")
        layers: list[nn.Module] = []
        input_dim: int | None = None
        # The first Linear is materialized lazily so the value network can be
        # constructed from the same hidden-size config before obs_dim is known.
        for size in hidden_sizes:
            if input_dim is None:
                layers.append(nn.LazyLinear(int(size)))
            else:
                layers.append(nn.Linear(input_dim, int(size)))
            layers.append(nn.SiLU())
            input_dim = int(size)
        self.hidden_layers = nn.Sequential(*layers)
        self.value_head = nn.Linear(int(input_dim), 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = torch.as_tensor(observation, dtype=torch.float32)
        return self.value_head(self.hidden_layers(hidden)).squeeze(-1)


def build_policy_copy(
    source: IntervalFlowPolicy,
    trainable: bool = False,
) -> IntervalFlowPolicy:
    """Create a policy with copied weights on the source's device."""

    try:
        device = next(source.parameters()).device
    except StopIteration:  # pragma: no cover - a policy always has parameters
        device = torch.device("cpu")
    copied = IntervalFlowPolicy(
        obs_dim=source.obs_dim,
        action_dim=source.action_dim,
        hidden_sizes=source.hidden_sizes,
        initial_log_std=source.initial_log_std,
        min_log_std=source.min_log_std,
        max_log_std=source.max_log_std,
    ).to(device)
    copied.load_state_dict(source.state_dict())
    copied.train(mode=trainable)
    for parameter in copied.parameters():
        parameter.requires_grad_(trainable)
    return copied


def update_ema(
    target: IntervalFlowPolicy,
    source: IntervalFlowPolicy,
    decay: torch.Tensor | float,
) -> None:
    """Update policy parameters and buffers with an exponential moving average."""

    decay_value = float(torch.as_tensor(decay).detach().cpu())
    if not 0.0 <= decay_value <= 1.0:
        raise ValueError("EMA decay must be in [0, 1]")
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.mul_(decay_value).add_(
                source_parameter, alpha=1.0 - decay_value
            )
        for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
            target_buffer.copy_(
                decay_value * target_buffer + (1.0 - decay_value) * source_buffer
            )
