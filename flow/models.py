"""Neural networks for the direct-ratio one-step flow policy.

The policy is represented as an augmented stochastic policy

    z ~ N(0, I)
    mean = z + (r - t) * u_theta(z, t, r | observation)
    pre_tanh_action ~ N(mean, diag(std_theta ** 2))

Conditioning the likelihood on ``z`` makes the policy likelihood tractable while
the marginal action distribution remains an expressive continuous mixture.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import tensorflow as tf


LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass
class PolicySample:
    """A policy sample and the variables required for a direct ratio."""

    pre_tanh_action: tf.Tensor
    flow_init: tf.Tensor
    flow_start: tf.Tensor
    log_prob: tf.Tensor
    mean: tf.Tensor


class IntervalFlowPolicy(tf.keras.Model):
    """Observation-conditioned interval-average velocity policy.

    ``velocity(obs, z_t, t, r)`` models the OFP interval-average velocity
    ``u_theta(z_t, t, r | obs)``.  A learned null context supports the
    conditional/unconditional branches required by OFP self-guidance.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        initial_log_std: float = -1.0,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
        name: str = "interval_flow_policy",
    ) -> None:
        super().__init__(name=name)
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.initial_log_std = float(initial_log_std)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        self.context_layers = [
            tf.keras.layers.Dense(size, activation=tf.nn.silu)
            for size in self.hidden_sizes
        ]
        self.null_context = self.add_weight(
            name="null_context",
            shape=(self.hidden_sizes[-1],),
            initializer="zeros",
            trainable=True,
        )
        self.flow_layers = [
            tf.keras.layers.Dense(size, activation=tf.nn.silu)
            for size in self.hidden_sizes
        ]
        self.velocity_head = tf.keras.layers.Dense(
            self.action_dim,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="velocity",
        )
        self.log_std = self.add_weight(
            name="log_std",
            shape=(self.action_dim,),
            initializer=tf.keras.initializers.Constant(self.initial_log_std),
            trainable=True,
        )

    def _context(
        self,
        observation: tf.Tensor,
        condition_mask: tf.Tensor | None,
        training: bool,
    ) -> tf.Tensor:
        context = tf.cast(observation, tf.float32)
        for layer in self.context_layers:
            context = layer(context, training=training)

        batch_size = tf.shape(context)[0]
        if condition_mask is None:
            condition_mask = tf.ones((batch_size, 1), dtype=context.dtype)
        else:
            condition_mask = tf.cast(condition_mask, context.dtype)
            condition_mask = tf.reshape(condition_mask, (batch_size, 1))
        null_context = tf.broadcast_to(self.null_context[None, :], tf.shape(context))
        return condition_mask * context + (1.0 - condition_mask) * null_context

    def velocity(
        self,
        observation: tf.Tensor,
        state: tf.Tensor,
        start_time: tf.Tensor,
        end_time: tf.Tensor,
        condition_mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Predict ``u_theta(state, start_time, end_time | observation)``.

        Time tensors must be broadcastable to ``[batch, 1]``.  Keeping them as
        explicit tensors avoids retracing in TensorFlow-compiled train steps.
        """

        observation = tf.cast(observation, tf.float32)
        state = tf.cast(state, tf.float32)
        batch_size = tf.shape(state)[0]
        start_time = tf.reshape(tf.cast(start_time, state.dtype), (batch_size, 1))
        end_time = tf.reshape(tf.cast(end_time, state.dtype), (batch_size, 1))
        interval = end_time - start_time
        time_features = tf.concat(
            [
                start_time,
                end_time,
                interval,
                tf.sin(math.pi * start_time),
                tf.cos(math.pi * start_time),
                tf.sin(math.pi * end_time),
                tf.cos(math.pi * end_time),
            ],
            axis=-1,
        )
        context = self._context(observation, condition_mask, training)
        hidden = tf.concat([state, time_features, context], axis=-1)
        for layer in self.flow_layers:
            hidden = layer(hidden, training=training)
        return self.velocity_head(hidden, training=training)

    def interval_mean(
        self,
        observation: tf.Tensor,
        flow_init: tf.Tensor,
        flow_start: tf.Tensor,
        condition_mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Generate the terminal mean with one interval-flow evaluation."""

        batch_size = tf.shape(flow_init)[0]
        flow_start = tf.reshape(tf.cast(flow_start, tf.float32), (batch_size, 1))
        end_time = tf.ones_like(flow_start)
        average_velocity = self.velocity(
            observation,
            flow_init,
            flow_start,
            end_time,
            condition_mask=condition_mask,
            training=training,
        )
        return flow_init + (1.0 - flow_start) * average_velocity

    def one_step_mean(
        self,
        observation: tf.Tensor,
        noise: tf.Tensor,
        condition_mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Generate ``epsilon + u(epsilon, 0, 1 | observation)``."""

        batch_size = tf.shape(noise)[0]
        return self.interval_mean(
            observation,
            noise,
            tf.zeros((batch_size, 1), dtype=tf.float32),
            condition_mask=condition_mask,
            training=training,
        )

    def std(self) -> tf.Tensor:
        return tf.exp(
            tf.clip_by_value(self.log_std, self.min_log_std, self.max_log_std)
        )

    def conditional_log_prob(
        self,
        observation: tf.Tensor,
        pre_tanh_action: tf.Tensor,
        flow_init: tf.Tensor,
        flow_start: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Log p(pre_tanh_action | observation, flow_init).

        The environment-side tanh/affine Jacobian is deliberately omitted: it is
        identical in numerator and denominator and therefore cancels in all
        direct importance ratios used by the trainer.
        """

        mean = self.interval_mean(
            observation,
            flow_init,
            flow_start,
            training=training,
        )
        return self.log_prob_from_mean(pre_tanh_action, mean)

    def log_prob_from_mean(
        self,
        pre_tanh_action: tf.Tensor,
        mean: tf.Tensor,
    ) -> tf.Tensor:
        """Evaluate the residual Gaussian without another flow network call."""

        std = self.std()[None, :]
        normalized = (tf.cast(pre_tanh_action, tf.float32) - mean) / std
        per_dimension = -0.5 * (
            tf.square(normalized) + 2.0 * tf.math.log(std) + LOG_TWO_PI
        )
        return tf.reduce_sum(per_dimension, axis=-1)

    def conditional_entropy(self) -> tf.Tensor:
        """Entropy of the residual Gaussian conditional on the flow latent."""

        effective_log_std = tf.clip_by_value(
            self.log_std, self.min_log_std, self.max_log_std
        )
        return tf.reduce_sum(effective_log_std + 0.5 * (1.0 + LOG_TWO_PI))

    def sample(
        self,
        observation: tf.Tensor,
        flow_init: tf.Tensor | None = None,
        flow_start: tf.Tensor | None = None,
        residual_noise: tf.Tensor | None = None,
        deterministic: bool = False,
        training: bool = False,
    ) -> PolicySample:
        """Sample once and return every value needed to replay its ratio."""

        observation = tf.cast(observation, tf.float32)
        batch_size = tf.shape(observation)[0]
        if flow_init is None:
            flow_init = tf.random.normal((batch_size, self.action_dim))
        else:
            flow_init = tf.cast(flow_init, tf.float32)
        if flow_start is None:
            flow_start = tf.zeros((batch_size, 1), dtype=tf.float32)
        else:
            flow_start = tf.reshape(tf.cast(flow_start, tf.float32), (batch_size, 1))

        mean = self.interval_mean(
            observation,
            flow_init,
            flow_start,
            training=training,
        )
        if deterministic:
            pre_tanh_action = mean
        else:
            if residual_noise is None:
                residual_noise = tf.random.normal(tf.shape(mean))
            pre_tanh_action = mean + self.std()[None, :] * residual_noise
        log_prob = self.log_prob_from_mean(pre_tanh_action, mean)
        return PolicySample(
            pre_tanh_action=pre_tanh_action,
            flow_init=flow_init,
            flow_start=flow_start,
            log_prob=log_prob,
            mean=mean,
        )


class ValueNetwork(tf.keras.Model):
    """Small state-value network for MuJoCo state observations."""

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (128, 128),
        name: str = "value_network",
    ) -> None:
        super().__init__(name=name)
        self.hidden_layers = [
            tf.keras.layers.Dense(int(size), activation=tf.nn.silu)
            for size in hidden_sizes
        ]
        self.value_head = tf.keras.layers.Dense(
            1, kernel_initializer="zeros", name="value"
        )

    def call(self, observation: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden = tf.cast(observation, tf.float32)
        for layer in self.hidden_layers:
            hidden = layer(hidden, training=training)
        return tf.squeeze(self.value_head(hidden, training=training), axis=-1)


def build_policy_copy(
    source: IntervalFlowPolicy,
    name: str,
    trainable: bool = False,
) -> IntervalFlowPolicy:
    """Create a built policy with weights copied from ``source``."""

    copied = IntervalFlowPolicy(
        obs_dim=source.obs_dim,
        action_dim=source.action_dim,
        hidden_sizes=source.hidden_sizes,
        initial_log_std=source.initial_log_std,
        min_log_std=source.min_log_std,
        max_log_std=source.max_log_std,
        name=name,
    )
    dummy_obs = tf.zeros((1, source.obs_dim), dtype=tf.float32)
    dummy_noise = tf.zeros((1, source.action_dim), dtype=tf.float32)
    copied.one_step_mean(dummy_obs, dummy_noise)
    source.one_step_mean(dummy_obs, dummy_noise)
    copied.set_weights(source.get_weights())
    copied.trainable = trainable
    return copied


def update_ema(
    target: IntervalFlowPolicy,
    source: IntervalFlowPolicy,
    decay: tf.Tensor | float,
) -> None:
    """Update all policy weights, including the residual log standard deviation."""

    decay = tf.cast(decay, tf.float32)
    one_minus_decay = 1.0 - decay
    for target_weight, source_weight in zip(target.weights, source.weights):
        target_weight.assign(decay * target_weight + one_minus_decay * source_weight)
