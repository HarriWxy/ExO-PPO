"""ExO direct-ratio and recent-policy OFP self-distillation objectives."""

from __future__ import annotations

from typing import Mapping

import tensorflow as tf

from .models import IntervalFlowPolicy


def smooth_exo_ratio(
    ratio: tf.Tensor,
    center: tf.Tensor,
    clip_radius: float,
    beta: float,
) -> tf.Tensor:
    """ExO's smooth exponential extension around a per-sample center.

    Inside ``[center - clip_radius, center + clip_radius]`` this is the identity.
    Outside the interval it approaches a finite exponential tail.  ``center`` is
    the recent-policy/behavior ratio, so replay samples from several behavior
    policies retain a proximal constraint around the latest policy.

    smooth_exo_ratio() doesn't validate clip_radius/beta (e.g., beta<=0 causes division by zero). The PyTorch version guards this, and validate_config() only protects the CLI path. Add local validation (raise ValueError) to make the function safe when used from other call sites/tests.
    """

    ratio = tf.cast(ratio, tf.float32)
    center = tf.stop_gradient(tf.cast(center, tf.float32))
    beta_tensor = tf.cast(beta, ratio.dtype)
    radius = tf.cast(clip_radius, ratio.dtype)
    upper = center + radius
    lower = tf.maximum(center - radius, 0.0)

    # ``tf.where`` evaluates both branches.  Clamp the unused branch's exponent
    # to a non-positive value so extreme replay ratios cannot create inf/NaN
    # before selection.
    safe_upper_ratio = tf.maximum(ratio, upper)
    safe_lower_ratio = tf.minimum(ratio, lower)
    upper_tail = (
        upper
        + 1.0 / beta_tensor
        - tf.exp(beta_tensor * (upper - safe_upper_ratio)) / beta_tensor
    )
    lower_tail = (
        lower
        - 1.0 / beta_tensor
        + tf.exp(beta_tensor * (safe_lower_ratio - lower)) / beta_tensor
    )
    transformed = tf.where(ratio > upper, upper_tail, ratio)
    transformed = tf.where(ratio < lower, lower_tail, transformed)
    return tf.maximum(transformed, 0.0)


def direct_ratio_exo_loss(
    policy: IntervalFlowPolicy,
    recent_policy: IntervalFlowPolicy,
    observation: tf.Tensor,
    pre_tanh_action: tf.Tensor,
    flow_init: tf.Tensor,
    flow_start: tf.Tensor,
    behavior_log_prob: tf.Tensor,
    advantage: tf.Tensor,
    clip_radius: float,
    beta: float,
    entropy_coefficient: float = 0.0,
    max_log_ratio: float = 20.0,
) -> Mapping[str, tf.Tensor]:
    """Compute ExO-PPO using an exact conditional (augmented-policy) ratio."""

    current_log_prob = policy.conditional_log_prob(
        observation,
        pre_tanh_action,
        flow_init,
        flow_start,
        training=True,
    )
    recent_log_prob = tf.stop_gradient(
        recent_policy.conditional_log_prob(
            observation,
            pre_tanh_action,
            flow_init,
            flow_start,
            training=False,
        )
    )
    behavior_log_prob = tf.cast(behavior_log_prob, tf.float32)
    advantage = tf.cast(advantage, tf.float32)

    log_ratio = tf.clip_by_value(
        current_log_prob - behavior_log_prob,
        -max_log_ratio,
        max_log_ratio,
    )
    recent_log_ratio = tf.clip_by_value(
        recent_log_prob - behavior_log_prob,
        -max_log_ratio,
        max_log_ratio,
    )
    ratio = tf.exp(log_ratio)
    recent_ratio = tf.stop_gradient(tf.exp(recent_log_ratio))
    exo_ratio = smooth_exo_ratio(ratio, recent_ratio, clip_radius, beta)

    direct_surrogate = ratio * advantage
    exo_surrogate = exo_ratio * advantage
    surrogate = tf.minimum(direct_surrogate, exo_surrogate)
    entropy = policy.conditional_entropy()
    policy_loss = -tf.reduce_mean(surrogate) - entropy_coefficient * entropy

    outside_recent_band = tf.abs(ratio - recent_ratio) > clip_radius
    return {
        "policy_loss": policy_loss,
        "ratio": tf.reduce_mean(ratio),
        "recent_ratio": tf.reduce_mean(recent_ratio),
        "ratio_abs_log": tf.reduce_mean(tf.abs(log_ratio)),
        "approx_kl": tf.reduce_mean(behavior_log_prob - current_log_prob),
        "recent_approx_kl": tf.reduce_mean(recent_log_prob - current_log_prob),
        "recent_log_shift": tf.reduce_mean(tf.abs(recent_log_prob - current_log_prob)),
        "clip_fraction": tf.reduce_mean(tf.cast(outside_recent_band, tf.float32)),
        "entropy": entropy,
    }


def contracting_factor(
    update_step: tf.Tensor,
    contraction_steps: int,
    minimum: float = 0.05,
    power: float = 2.0,
) -> tf.Tensor:
    """Polynomial OFP time-contraction schedule from one toward ``minimum``."""

    progress = tf.minimum(
        tf.cast(update_step, tf.float32) / float(max(contraction_steps, 1)),
        1.0,
    )
    return minimum + (1.0 - minimum) * tf.pow(1.0 - progress, power)


def _mean_squared_l2(residual: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.reduce_mean(tf.square(residual), axis=-1))


def recent_policy_ofp_losses(
    policy: IntervalFlowPolicy,
    ema_teacher: IntervalFlowPolicy,
    recent_policy: IntervalFlowPolicy,
    observation: tf.Tensor,
    update_step: tf.Tensor,
    contraction_steps: int,
    condition_dropout: float = 0.1,
    guidance_scale: float = 1.0,
    minimum_contraction: float = 0.05,
    minimum_interval: float = 1e-3,
    enable_guidance: bool = True,
) -> Mapping[str, tf.Tensor]:
    """OFP Eq. 5, Eq. 7 and Eq. 13 with recent-policy action targets.

    The paper uses expert actions as the endpoint distribution.  In this online
    RL adaptation, endpoints are freshly sampled from the frozen recent policy.
    This makes OFP a proximal, self-distilled model of the latest improved policy
    instead of a behavior-cloning model of stale replay actions.
    """

    observation = tf.cast(observation, tf.float32)
    batch_size = tf.shape(observation)[0]
    action_dim = policy.action_dim
    shape = (batch_size, action_dim)
    time_shape = (batch_size, 1)

    # The latest frozen policy supplies the online analogue of expert data.
    recent_noise = tf.random.normal(shape)
    recent_mean = recent_policy.one_step_mean(observation, recent_noise, training=False)
    # Distil the recent transport distribution, not its auxiliary Gaussian
    # exploration kernel.  Including that kernel here and adding it again when
    # acting would recursively inflate the marginal variance after each sync.
    target_action = tf.stop_gradient(recent_mean)

    keep_condition = tf.cast(
        tf.random.uniform(time_shape) >= condition_dropout, tf.float32
    )

    # Boundary anchoring (paper Eq. 7): instantaneous velocity at t == r.
    # Reuse the recent policy's latent as an OT-style paired endpoint.  Unlike
    # an offline expert action, an online recent-policy action has a known
    # generating latent; preserving that coupling avoids needless transport
    # variance and makes the proximal flow target exact at initialization.
    flow_noise = recent_noise
    flow_time = tf.random.uniform(time_shape, minval=0.0, maxval=1.0)
    flow_state = (1.0 - flow_time) * flow_noise + flow_time * target_action
    flow_prediction = policy.velocity(
        observation,
        flow_state,
        flow_time,
        flow_time,
        condition_mask=keep_condition,
        training=True,
    )
    flow_loss = _mean_squared_l2(flow_prediction - (target_action - flow_noise))

    # Self-consistency (paper Eq. 4-6), using a forward-only EMA target.
    t = tf.random.uniform(time_shape, minval=0.0, maxval=1.0 - minimum_interval)
    remaining = 1.0 - t - minimum_interval
    r = t + minimum_interval + remaining * tf.random.uniform(time_shape)
    rho = contracting_factor(
        update_step,
        contraction_steps,
        minimum=minimum_contraction,
    )
    m = t + (r - t) * rho * tf.random.uniform(time_shape)
    consistency_noise = recent_noise
    state_t = (1.0 - t) * consistency_noise + t * target_action
    state_m = (1.0 - m) * consistency_noise + m * target_action
    teacher_velocity = ema_teacher.velocity(
        observation,
        state_m,
        m,
        r,
        condition_mask=keep_condition,
        training=False,
    )
    predicted_state_r = state_m + (r - m) * teacher_velocity
    consistency_target = tf.stop_gradient((predicted_state_r - state_t) / (r - t))
    consistency_prediction = policy.velocity(
        observation,
        state_t,
        t,
        r,
        condition_mask=keep_condition,
        training=True,
    )
    consistency_loss = _mean_squared_l2(consistency_prediction - consistency_target)

    # Self-guidance (paper Eq. 12-13).  The target's stop-gradient gives the
    # current branch the classifier-free conditional direction without a
    # pretrained score network.
    if enable_guidance:
        guidance_t = tf.random.uniform(time_shape, minval=0.0, maxval=1.0)
        guidance_noise = recent_noise
        guidance_state = (
            1.0 - guidance_t
        ) * guidance_noise + guidance_t * target_action
        one_step_velocity = policy.velocity(
            observation,
            guidance_state,
            guidance_t,
            tf.ones_like(guidance_t),
            training=True,
        )
        generated_action = guidance_state + (1.0 - guidance_t) * one_step_velocity
        reprime_time = tf.random.uniform(time_shape, minval=1e-3, maxval=1.0 - 1e-3)
        reprime_noise = tf.random.normal(shape)
        reprime_state = (
            1.0 - reprime_time
        ) * reprime_noise + reprime_time * tf.stop_gradient(generated_action)
        conditional_mask = tf.ones(time_shape, dtype=tf.float32)
        unconditional_mask = tf.zeros(time_shape, dtype=tf.float32)
        teacher_unconditional = ema_teacher.velocity(
            observation,
            reprime_state,
            reprime_time,
            reprime_time,
            condition_mask=unconditional_mask,
            training=False,
        )
        teacher_conditional = ema_teacher.velocity(
            observation,
            reprime_state,
            reprime_time,
            reprime_time,
            condition_mask=conditional_mask,
            training=False,
        )
        conditional_gap = teacher_unconditional - teacher_conditional
        guidance_target = tf.stop_gradient(
            one_step_velocity - guidance_scale * conditional_gap
        )
        guidance_loss = _mean_squared_l2(one_step_velocity - guidance_target)
    else:
        guidance_loss = tf.zeros((), dtype=tf.float32)

    return {
        "flow_loss": flow_loss,
        "consistency_loss": consistency_loss,
        "guidance_loss": guidance_loss,
        "contraction": rho,
        "target_action_rms": tf.sqrt(tf.reduce_mean(tf.square(target_action))),
    }
