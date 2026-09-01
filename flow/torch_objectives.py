"""PyTorch direct-ratio ExO and recent-policy OFP objectives."""

from __future__ import annotations

from typing import Mapping

import torch

from .torch_models import IntervalFlowPolicy


def smooth_exo_ratio(
    ratio: torch.Tensor,
    center: torch.Tensor,
    clip_radius: float,
    beta: float,
) -> torch.Tensor:
    """Apply ExO's smooth exponential extension around a recent-policy center."""

    ratio = torch.as_tensor(ratio, dtype=torch.float32)
    center = torch.as_tensor(center, dtype=ratio.dtype, device=ratio.device).detach()
    if clip_radius <= 0.0 or beta <= 0.0:
        raise ValueError("clip_radius and beta must be positive")
    radius = torch.as_tensor(clip_radius, dtype=ratio.dtype, device=ratio.device)
    beta_tensor = torch.as_tensor(beta, dtype=ratio.dtype, device=ratio.device)
    upper = center + radius
    lower = torch.maximum(center - radius, torch.zeros_like(center))

    # Clamp the unused branch's exponent to <= 0.  torch.where evaluates both
    # branches, so this keeps extreme replay ratios finite before selection.
    safe_upper_ratio = torch.maximum(ratio, upper)
    safe_lower_ratio = torch.minimum(ratio, lower)
    upper_tail = (
        upper
        + 1.0 / beta_tensor
        - torch.exp(beta_tensor * (upper - safe_upper_ratio)) / beta_tensor
    )
    lower_tail = (
        lower
        - 1.0 / beta_tensor
        + torch.exp(beta_tensor * (safe_lower_ratio - lower)) / beta_tensor
    )
    transformed = torch.where(ratio > upper, upper_tail, ratio)
    transformed = torch.where(ratio < lower, lower_tail, transformed)
    return transformed.clamp_min(0.0)


def direct_ratio_exo_loss(
    policy: IntervalFlowPolicy,
    recent_policy: IntervalFlowPolicy,
    observation: torch.Tensor,
    pre_tanh_action: torch.Tensor,
    flow_init: torch.Tensor,
    flow_start: torch.Tensor,
    behavior_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    clip_radius: float,
    beta: float,
    entropy_coefficient: float = 0.0,
    max_log_ratio: float = 20.0,
) -> Mapping[str, torch.Tensor]:
    """Compute ExO-PPO from an exact conditional augmented-policy ratio."""

    if max_log_ratio <= 0.0:
        raise ValueError("max_log_ratio must be positive")
    current_log_prob = policy.conditional_log_prob(
        observation, pre_tanh_action, flow_init, flow_start
    )
    with torch.no_grad():
        recent_log_prob = recent_policy.conditional_log_prob(
            observation, pre_tanh_action, flow_init, flow_start
        )
    behavior_log_prob = torch.as_tensor(
        behavior_log_prob, dtype=torch.float32, device=current_log_prob.device
    )
    advantage = torch.as_tensor(
        advantage, dtype=torch.float32, device=current_log_prob.device
    )
    log_ratio = (current_log_prob - behavior_log_prob).clamp(
        -max_log_ratio, max_log_ratio
    )
    recent_log_ratio = (recent_log_prob - behavior_log_prob).clamp(
        -max_log_ratio, max_log_ratio
    )
    ratio = log_ratio.exp()
    recent_ratio = recent_log_ratio.exp().detach()
    exo_ratio = smooth_exo_ratio(ratio, recent_ratio, clip_radius, beta)

    direct_surrogate = ratio * advantage
    exo_surrogate = exo_ratio * advantage
    surrogate = torch.minimum(direct_surrogate, exo_surrogate)
    policy_gradient_loss = -surrogate.mean()
    entropy = policy.conditional_entropy()
    entropy_loss = -entropy
    policy_loss = policy_gradient_loss + entropy_coefficient * entropy_loss
    outside_recent_band = (ratio - recent_ratio).abs() > clip_radius
    return {
        "policy_loss": policy_loss,
        "policy_gradient_loss": policy_gradient_loss,
        "ratio": ratio.mean(),
        "recent_ratio": recent_ratio.mean(),
        "ratio_abs_log": log_ratio.abs().mean(),
        # Same reverse-KL approximation used by Stable-Baselines3 PPO.
        "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
        "behavior_kl": (behavior_log_prob - current_log_prob).mean(),
        "recent_approx_kl": (recent_log_prob - current_log_prob).mean(),
        "recent_log_shift": (recent_log_prob - current_log_prob).abs().mean(),
        "clip_fraction": outside_recent_band.float().mean(),
        "entropy": entropy,
        "entropy_loss": entropy_loss,
    }


def contracting_factor(
    update_step: torch.Tensor | int,
    contraction_steps: int,
    minimum: float = 0.05,
    power: float = 2.0,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Polynomial OFP time-contraction schedule from one toward ``minimum``."""

    if contraction_steps <= 0:
        raise ValueError("contraction_steps must be positive")
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum must be in [0, 1]")
    step = torch.as_tensor(update_step, dtype=torch.float32, device=device)
    progress = (step / float(contraction_steps)).clamp_max(1.0)
    return minimum + (1.0 - minimum) * (1.0 - progress).pow(power)


def _mean_squared_l2(residual: torch.Tensor) -> torch.Tensor:
    return residual.square().mean(dim=-1).mean()


def recent_policy_ofp_losses(
    policy: IntervalFlowPolicy,
    ema_teacher: IntervalFlowPolicy,
    recent_policy: IntervalFlowPolicy,
    observation: torch.Tensor,
    update_step: torch.Tensor | int,
    contraction_steps: int,
    condition_dropout: float = 0.1,
    guidance_scale: float = 1.0,
    minimum_contraction: float = 0.05,
    minimum_interval: float = 1e-3,
    enable_guidance: bool = True,
) -> Mapping[str, torch.Tensor]:
    """OFP boundary, consistency and self-guidance losses.

    The OFP paper uses expert actions as endpoints.  In this online RL variant,
    a frozen recent policy supplies the endpoint and its latent is reused for a
    paired transport target.  The residual exploration Gaussian is deliberately
    not distilled a second time.
    """

    if not 0.0 <= condition_dropout < 1.0:
        raise ValueError("condition_dropout must be in [0, 1)")
    if not 0.0 <= minimum_interval < 1.0:
        raise ValueError("minimum_interval must be in [0, 1)")
    observation = torch.as_tensor(observation, dtype=torch.float32)
    batch_size = observation.shape[0]
    action_dim = policy.action_dim
    device = observation.device
    shape = (batch_size, action_dim)
    time_shape = (batch_size, 1)

    recent_noise = torch.randn(shape, dtype=observation.dtype, device=device)
    with torch.no_grad():
        recent_mean = recent_policy.one_step_mean(observation, recent_noise)
    target_action = recent_mean.detach()
    keep_condition = (
        torch.rand(time_shape, dtype=observation.dtype, device=device)
        >= condition_dropout
    ).to(observation.dtype)

    # Boundary anchoring (the t == r endpoint condition).
    flow_noise = recent_noise
    flow_time = torch.rand(time_shape, dtype=observation.dtype, device=device)
    flow_state = (1.0 - flow_time) * flow_noise + flow_time * target_action
    flow_prediction = policy.velocity(
        observation,
        flow_state,
        flow_time,
        flow_time,
        condition_mask=keep_condition,
    )
    flow_loss = _mean_squared_l2(flow_prediction - (target_action - flow_noise))

    # Self-consistency with a forward-only EMA teacher.
    t = torch.rand(time_shape, dtype=observation.dtype, device=device) * (
        1.0 - minimum_interval
    )
    remaining = 1.0 - t - minimum_interval
    r = t + minimum_interval + remaining * torch.rand_like(t)
    rho = contracting_factor(
        update_step,
        contraction_steps,
        minimum=minimum_contraction,
        device=device,
    )
    m = t + (r - t) * rho * torch.rand_like(t)
    state_t = (1.0 - t) * recent_noise + t * target_action
    state_m = (1.0 - m) * recent_noise + m * target_action
    with torch.no_grad():
        teacher_velocity = ema_teacher.velocity(
            observation,
            state_m,
            m,
            r,
            condition_mask=keep_condition,
        )
    predicted_state_r = state_m + (r - m) * teacher_velocity
    consistency_target = ((predicted_state_r - state_t) / (r - t)).detach()
    consistency_prediction = policy.velocity(
        observation,
        state_t,
        t,
        r,
        condition_mask=keep_condition,
    )
    consistency_loss = _mean_squared_l2(consistency_prediction - consistency_target)

    if enable_guidance:
        guidance_t = torch.rand(time_shape, dtype=observation.dtype, device=device)
        guidance_state = (1.0 - guidance_t) * recent_noise + guidance_t * target_action
        one_step_velocity = policy.velocity(
            observation,
            guidance_state,
            guidance_t,
            torch.ones_like(guidance_t),
        )
        generated_action = guidance_state + (1.0 - guidance_t) * one_step_velocity
        reprime_time = 1e-3 + (1.0 - 2e-3) * torch.rand(
            time_shape, dtype=observation.dtype, device=device
        )
        reprime_noise = torch.randn(shape, dtype=observation.dtype, device=device)
        reprime_state = (
            1.0 - reprime_time
        ) * reprime_noise + reprime_time * generated_action.detach()
        unconditional_mask = torch.zeros_like(reprime_time)
        conditional_mask = torch.ones_like(reprime_time)
        with torch.no_grad():
            teacher_unconditional = ema_teacher.velocity(
                observation,
                reprime_state,
                reprime_time,
                reprime_time,
                condition_mask=unconditional_mask,
            )
            teacher_conditional = ema_teacher.velocity(
                observation,
                reprime_state,
                reprime_time,
                reprime_time,
                condition_mask=conditional_mask,
            )
        conditional_gap = teacher_unconditional - teacher_conditional
        guidance_target = (
            one_step_velocity - guidance_scale * conditional_gap
        ).detach()
        guidance_loss = _mean_squared_l2(one_step_velocity - guidance_target)
    else:
        guidance_loss = torch.zeros((), dtype=observation.dtype, device=device)

    return {
        "flow_loss": flow_loss,
        "consistency_loss": consistency_loss,
        "guidance_loss": guidance_loss,
        "contraction": rho,
        "target_action_rms": target_action.square().mean().sqrt(),
    }
