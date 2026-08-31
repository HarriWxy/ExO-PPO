"""Train direct-ratio ExO-PPO with a recent-policy one-step flow.

Run from the repository root:

    python -m flow.train --env-id Walker2d-v5

The implementation intentionally keeps the behavior likelihood, recent-policy
likelihood, flow latent and pre-tanh action in replay.  Dropping any of those
would silently turn the direct ratio into an approximation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Sequence

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gymnasium as gym
import numpy as np
import tensorflow as tf

from flow.buffer import (
    ReplayWindow,
    Rollout,
    RunningMeanStd,
    flatten_rollout,
    generalized_advantage_estimate,
)
from flow.models import (
    IntervalFlowPolicy,
    ValueNetwork,
    build_policy_copy,
    update_ema,
)
from flow.objectives import direct_ratio_exo_loss, recent_policy_ofp_losses


@dataclass(frozen=True)
class TrainConfig:
    env_id: str = "HalfCheetah-v5"
    seed: int = 0
    total_steps: int = 1_000_000
    num_envs: int = 4
    rollout_steps: int = 256
    replay_rollouts: int = 4
    warmup_rollouts: int = 4
    update_epochs: int = 2
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95

    hidden_sizes: tuple[int, ...] = (128, 128)
    initial_log_std: float = -1.0
    actor_learning_rate: float = 2e-4
    critic_learning_rate: float = 2e-4
    max_grad_norm: float = 1.0

    exo_clip_radius: float = 0.2
    exo_beta: float = 5.0
    entropy_coefficient: float = 0.0
    max_log_ratio: float = 12.0
    target_kl: float = 0.0

    ofp_coefficient: float = 0.1
    flow_mix: float = 0.8
    consistency_mix: float = 0.2
    guidance_mix: float = 0.05
    guidance_scale: float = 1.0
    condition_dropout: float = 0.1
    contraction_steps: int = 50_000
    minimum_contraction: float = 0.05
    ema_initial_decay: float = 0.75
    ema_max_decay: float = 0.9999
    ema_ramp_steps: int = 10_000

    warm_start_time: float = 0.0
    eval_every_rollouts: int = 20
    eval_episodes: int = 5
    stochastic_eval: bool = False
    log_dir: str = "logs"
    checkpoint_dir: str = ""


def _parse_hidden_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError(
            "hidden sizes must be comma-separated positive integers"
        )
    return sizes


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Direct-ratio ExO-PPO + recent-policy one-step flow"
    )
    parser.add_argument("--env-id", default=TrainConfig.env_id)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--total-steps", type=int, default=TrainConfig.total_steps)
    parser.add_argument("--num-envs", type=int, default=TrainConfig.num_envs)
    parser.add_argument("--rollout-steps", type=int, default=TrainConfig.rollout_steps)
    parser.add_argument(
        "--replay-rollouts", type=int, default=TrainConfig.replay_rollouts
    )
    parser.add_argument(
        "--warmup-rollouts", type=int, default=TrainConfig.warmup_rollouts
    )
    parser.add_argument("--update-epochs", type=int, default=TrainConfig.update_epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    parser.add_argument("--gae-lambda", type=float, default=TrainConfig.gae_lambda)

    parser.add_argument(
        "--hidden-sizes",
        type=_parse_hidden_sizes,
        default=TrainConfig.hidden_sizes,
        help="comma-separated MLP widths",
    )
    parser.add_argument(
        "--initial-log-std", type=float, default=TrainConfig.initial_log_std
    )
    parser.add_argument(
        "--actor-lr",
        dest="actor_learning_rate",
        type=float,
        default=TrainConfig.actor_learning_rate,
    )
    parser.add_argument(
        "--critic-lr",
        dest="critic_learning_rate",
        type=float,
        default=TrainConfig.critic_learning_rate,
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=TrainConfig.max_grad_norm
    )

    parser.add_argument(
        "--exo-clip-radius", type=float, default=TrainConfig.exo_clip_radius
    )
    parser.add_argument("--exo-beta", type=float, default=TrainConfig.exo_beta)
    parser.add_argument(
        "--entropy-coef",
        dest="entropy_coefficient",
        type=float,
        default=TrainConfig.entropy_coefficient,
    )
    parser.add_argument(
        "--max-log-ratio", type=float, default=TrainConfig.max_log_ratio
    )
    parser.add_argument("--target-kl", type=float, default=TrainConfig.target_kl)

    parser.add_argument(
        "--ofp-coef",
        dest="ofp_coefficient",
        type=float,
        default=TrainConfig.ofp_coefficient,
    )
    parser.add_argument("--flow-mix", type=float, default=TrainConfig.flow_mix)
    parser.add_argument(
        "--consistency-mix", type=float, default=TrainConfig.consistency_mix
    )
    parser.add_argument("--guidance-mix", type=float, default=TrainConfig.guidance_mix)
    parser.add_argument(
        "--guidance-scale", type=float, default=TrainConfig.guidance_scale
    )
    parser.add_argument(
        "--condition-dropout", type=float, default=TrainConfig.condition_dropout
    )
    parser.add_argument(
        "--contraction-steps", type=int, default=TrainConfig.contraction_steps
    )
    parser.add_argument(
        "--minimum-contraction",
        type=float,
        default=TrainConfig.minimum_contraction,
    )
    parser.add_argument(
        "--ema-initial-decay",
        type=float,
        default=TrainConfig.ema_initial_decay,
    )
    parser.add_argument(
        "--ema-max-decay", type=float, default=TrainConfig.ema_max_decay
    )
    parser.add_argument(
        "--ema-ramp-steps", type=int, default=TrainConfig.ema_ramp_steps
    )

    parser.add_argument(
        "--warm-start-time", type=float, default=TrainConfig.warm_start_time
    )
    parser.add_argument(
        "--eval-every-rollouts",
        type=int,
        default=TrainConfig.eval_every_rollouts,
    )
    parser.add_argument("--eval-episodes", type=int, default=TrainConfig.eval_episodes)
    parser.add_argument("--stochastic-eval", action="store_true", default=False)
    parser.add_argument("--log-dir", default=TrainConfig.log_dir)
    parser.add_argument("--checkpoint-dir", default=TrainConfig.checkpoint_dir)
    args = parser.parse_args(argv)
    config = TrainConfig(**vars(args))
    validate_config(config)
    return config


def validate_config(config: TrainConfig) -> None:
    positive_integer_fields = (
        "total_steps",
        "num_envs",
        "rollout_steps",
        "replay_rollouts",
        "warmup_rollouts",
        "update_epochs",
        "batch_size",
        "contraction_steps",
        "ema_ramp_steps",
        "eval_every_rollouts",
        "eval_episodes",
    )
    for field in positive_integer_fields:
        if getattr(config, field) <= 0:
            raise ValueError(f"{field} must be positive")
    if config.warmup_rollouts > config.replay_rollouts:
        raise ValueError("warmup_rollouts cannot exceed replay_rollouts")
    if not 0.0 <= config.condition_dropout < 1.0:
        raise ValueError("condition_dropout must be in [0, 1)")
    if not 0.0 <= config.warm_start_time < 1.0:
        raise ValueError("warm_start_time must be in [0, 1)")
    if config.exo_beta <= 0.0 or config.exo_clip_radius <= 0.0:
        raise ValueError("ExO beta and clip radius must be positive")
    if config.max_log_ratio <= 0.0:
        raise ValueError("max_log_ratio must be positive")
    if not 0.0 < config.gamma <= 1.0 or not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in their valid ranges")
    if config.actor_learning_rate <= 0.0 or config.critic_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    loss_weights = (
        config.ofp_coefficient,
        config.flow_mix,
        config.consistency_mix,
        config.guidance_mix,
        config.guidance_scale,
        config.entropy_coefficient,
        config.target_kl,
    )
    if any(weight < 0.0 for weight in loss_weights):
        raise ValueError("loss weights and target_kl cannot be negative")
    if not 0.0 <= config.minimum_contraction <= 1.0:
        raise ValueError("minimum_contraction must be in [0, 1]")
    if not 0.0 <= config.ema_initial_decay <= config.ema_max_decay < 1.0:
        raise ValueError("EMA decays must satisfy 0 <= initial <= max < 1")


def make_vector_env(env_id: str, num_envs: int) -> gym.vector.VectorEnv:
    """Create a synchronous vector env with same-step autoreset when available."""

    factories: list[Callable[[], gym.Env[Any, Any]]] = [
        lambda env_id=env_id: gym.make(env_id) for _ in range(num_envs)
    ]
    autoreset_mode = getattr(gym.vector, "AutoresetMode", None)
    if autoreset_mode is not None:
        try:
            return gym.vector.SyncVectorEnv(
                factories, autoreset_mode=autoreset_mode.SAME_STEP
            )
        except TypeError:
            pass
    return gym.vector.SyncVectorEnv(factories)


def action_from_pre_tanh(
    pre_tanh_action: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """Map an unconstrained action to arbitrary finite Box bounds."""

    center = (action_high + action_low) * 0.5
    scale = (action_high - action_low) * 0.5
    return center + scale * np.tanh(pre_tanh_action)


class Trainer:
    def __init__(self, config: TrainConfig, obs_dim: int, action_dim: int) -> None:
        self.config = config
        self.policy = IntervalFlowPolicy(
            obs_dim,
            action_dim,
            hidden_sizes=config.hidden_sizes,
            initial_log_std=config.initial_log_std,
        )
        dummy_obs = tf.zeros((1, obs_dim), dtype=tf.float32)
        dummy_noise = tf.zeros((1, action_dim), dtype=tf.float32)
        self.policy.one_step_mean(dummy_obs, dummy_noise)
        self.recent_policy = build_policy_copy(
            self.policy, "recent_policy", trainable=False
        )
        self.ema_teacher = build_policy_copy(
            self.policy, "ema_teacher", trainable=False
        )
        self.value = ValueNetwork(config.hidden_sizes)
        self.value(dummy_obs)

        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.actor_learning_rate, epsilon=1e-5
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.critic_learning_rate, epsilon=1e-5
        )
        self.update_step = tf.Variable(0, dtype=tf.int64, trainable=False)

    def snapshot_recent_policy(self) -> None:
        """Freeze the proximal center for the complete replay update."""

        self.recent_policy.set_weights(self.policy.get_weights())

    def _ema_decay(self) -> tf.Tensor:
        progress = tf.cast(self.update_step, tf.float32) / float(
            self.config.ema_ramp_steps
        )
        progress = progress / (1.0 + progress)
        decay = self.config.ema_initial_decay + progress * (
            self.config.ema_max_decay - self.config.ema_initial_decay
        )
        return tf.minimum(decay, self.config.ema_max_decay)

    @tf.function(reduce_retracing=True)
    def actor_train_step(
        self,
        observation: tf.Tensor,
        pre_tanh_action: tf.Tensor,
        flow_init: tf.Tensor,
        flow_start: tf.Tensor,
        behavior_log_prob: tf.Tensor,
        advantage: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            exo = direct_ratio_exo_loss(
                self.policy,
                self.recent_policy,
                observation,
                pre_tanh_action,
                flow_init,
                flow_start,
                behavior_log_prob,
                advantage,
                clip_radius=self.config.exo_clip_radius,
                beta=self.config.exo_beta,
                entropy_coefficient=self.config.entropy_coefficient,
                max_log_ratio=self.config.max_log_ratio,
            )
            if self.config.ofp_coefficient > 0.0:
                ofp = recent_policy_ofp_losses(
                    self.policy,
                    self.ema_teacher,
                    self.recent_policy,
                    observation,
                    self.update_step,
                    contraction_steps=self.config.contraction_steps,
                    condition_dropout=self.config.condition_dropout,
                    guidance_scale=self.config.guidance_scale,
                    minimum_contraction=self.config.minimum_contraction,
                    enable_guidance=self.config.guidance_mix > 0.0,
                )
                self_distill_loss = self.config.ofp_coefficient * (
                    self.config.flow_mix * ofp["flow_loss"]
                    + self.config.consistency_mix * ofp["consistency_loss"]
                    + self.config.guidance_mix * ofp["guidance_loss"]
                )
            else:
                zero = tf.zeros((), dtype=tf.float32)
                ofp = {
                    "flow_loss": zero,
                    "consistency_loss": zero,
                    "guidance_loss": zero,
                    "contraction": zero,
                    "target_action_rms": zero,
                }
                self_distill_loss = zero
            total_loss = exo["policy_loss"] + self_distill_loss
            tf.debugging.check_numerics(total_loss, "non-finite actor loss")

        variables = self.policy.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        gradients, gradient_norm = tf.clip_by_global_norm(
            gradients, self.config.max_grad_norm
        )
        self.actor_optimizer.apply_gradients(zip(gradients, variables))
        ema_decay = self._ema_decay()
        update_ema(self.ema_teacher, self.policy, ema_decay)
        self.update_step.assign_add(1)

        return {
            **exo,
            **ofp,
            "actor_loss": total_loss,
            "self_distill_loss": self_distill_loss,
            "actor_grad_norm": gradient_norm,
            "ema_decay": ema_decay,
        }

    @tf.function(reduce_retracing=True)
    def critic_train_step(
        self, observation: tf.Tensor, returns: tf.Tensor
    ) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            prediction = self.value(observation, training=True)
            residual = tf.cast(returns, tf.float32) - prediction
            absolute = tf.abs(residual)
            quadratic = tf.minimum(absolute, 1.0)
            linear = absolute - quadratic
            critic_loss = tf.reduce_mean(0.5 * tf.square(quadratic) + linear)
            tf.debugging.check_numerics(critic_loss, "non-finite critic loss")
        variables = self.value.trainable_variables
        gradients = tape.gradient(critic_loss, variables)
        gradients, gradient_norm = tf.clip_by_global_norm(
            gradients, self.config.max_grad_norm
        )
        self.critic_optimizer.apply_gradients(zip(gradients, variables))
        return {
            "critic_loss": critic_loss,
            "critic_grad_norm": gradient_norm,
            "value_mean": tf.reduce_mean(prediction),
        }

    def train_replay(
        self, replay: ReplayWindow, rng: np.random.Generator
    ) -> dict[str, float]:
        arrays = replay.arrays()
        advantages = arrays["advantages"]
        arrays["advantages"] = (
            (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ).astype(np.float32)

        self.snapshot_recent_policy()
        metric_sums: dict[str, float] = {}
        metric_count = 0
        sample_count = len(replay)
        stop_early = False
        for _ in range(self.config.update_epochs):
            order = rng.permutation(sample_count)
            for offset in range(0, sample_count, self.config.batch_size):
                indices = order[offset : offset + self.config.batch_size]
                observation = tf.convert_to_tensor(
                    arrays["observations"][indices], tf.float32
                )
                actor_metrics = self.actor_train_step(
                    observation,
                    tf.convert_to_tensor(
                        arrays["pre_tanh_actions"][indices], tf.float32
                    ),
                    tf.convert_to_tensor(arrays["flow_init"][indices], tf.float32),
                    tf.convert_to_tensor(arrays["flow_start"][indices], tf.float32),
                    tf.convert_to_tensor(
                        arrays["behavior_log_prob"][indices], tf.float32
                    ),
                    tf.convert_to_tensor(arrays["advantages"][indices], tf.float32),
                )
                critic_metrics = self.critic_train_step(
                    observation,
                    tf.convert_to_tensor(arrays["returns"][indices], tf.float32),
                )
                batch_metrics = {**actor_metrics, **critic_metrics}
                for key, value in batch_metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                metric_count += 1

                if (
                    self.config.target_kl > 0.0
                    and float(actor_metrics["recent_log_shift"]) > self.config.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break

        averaged = {
            key: value / max(metric_count, 1) for key, value in metric_sums.items()
        }
        averaged["early_stop"] = float(stop_early)
        averaged["minibatches"] = float(metric_count)
        return averaged


def collect_rollout(
    env: gym.vector.VectorEnv,
    trainer: Trainer,
    observation: np.ndarray,
    observation_stats: RunningMeanStd,
    action_low: np.ndarray,
    action_high: np.ndarray,
    episode_returns: np.ndarray,
    episode_lengths: np.ndarray,
    previous_action: np.ndarray,
    has_previous_action: np.ndarray,
) -> tuple[
    Rollout,
    np.ndarray,
    list[float],
    list[int],
    np.ndarray,
    np.ndarray,
]:
    config = trainer.config
    observations: list[np.ndarray] = []
    pre_tanh_actions: list[np.ndarray] = []
    flow_initializations: list[np.ndarray] = []
    flow_start_times: list[np.ndarray] = []
    behavior_log_probs: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    dones: list[np.ndarray] = []
    values: list[np.ndarray] = []
    completed_returns: list[float] = []
    completed_lengths: list[int] = []

    for _ in range(config.rollout_steps):
        observation_tensor = tf.convert_to_tensor(observation, tf.float32)
        batch_size = observation.shape[0]
        pure_noise = np.random.normal(size=(batch_size, trainer.policy.action_dim))
        if config.warm_start_time > 0.0:
            active = has_previous_action.astype(np.float32)[:, None]
            start_value = active * config.warm_start_time
            flow_init = (
                1.0 - start_value
            ) * pure_noise + start_value * previous_action
            flow_start = start_value
        else:
            ## Since collect_rollout() is already NumPy-driven (env stepping, storage), consider generating pure_noise directly in NumPy (or using the existing rng) and only converting to Tensor once when calling policy.sample().
            flow_init = pure_noise
            flow_start = np.zeros((batch_size, 1), dtype=np.float32)

        sample = trainer.policy.sample(
            observation_tensor,
            flow_init=tf.convert_to_tensor(flow_init, tf.float32),
            flow_start=tf.convert_to_tensor(flow_start, tf.float32),
            deterministic=False,
            training=False,
        )
        value = trainer.value(observation_tensor, training=False)
        pre_tanh = sample.pre_tanh_action.numpy()
        env_action = action_from_pre_tanh(pre_tanh, action_low, action_high)
        next_raw_observation, reward, terminated, truncated, _ = env.step(env_action)
        done = np.logical_or(terminated, truncated)

        observations.append(observation.copy())
        pre_tanh_actions.append(pre_tanh)
        flow_initializations.append(sample.flow_init.numpy())
        flow_start_times.append(sample.flow_start.numpy())
        behavior_log_probs.append(sample.log_prob.numpy())
        rewards.append(np.asarray(reward, dtype=np.float32))
        dones.append(np.asarray(done, dtype=np.float32))
        values.append(value.numpy())

        episode_returns += reward
        episode_lengths += 1
        for env_index in np.flatnonzero(done):
            completed_returns.append(float(episode_returns[env_index]))
            completed_lengths.append(int(episode_lengths[env_index]))
            episode_returns[env_index] = 0.0
            episode_lengths[env_index] = 0

        previous_action[:] = pre_tanh
        has_previous_action[:] = True
        has_previous_action[done] = False
        previous_action[done] = 0.0

        observation_stats.update(next_raw_observation)
        observation = observation_stats.normalize(next_raw_observation)

    next_value = trainer.value(
        tf.convert_to_tensor(observation, tf.float32), training=False
    ).numpy()
    reward_array = np.asarray(rewards, dtype=np.float32)
    done_array = np.asarray(dones, dtype=np.float32)
    value_array = np.asarray(values, dtype=np.float32)
    advantages, returns = generalized_advantage_estimate(
        reward_array,
        done_array,
        value_array,
        next_value,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    rollout = flatten_rollout(
        np.asarray(observations, dtype=np.float32),
        np.asarray(pre_tanh_actions, dtype=np.float32),
        np.asarray(flow_initializations, dtype=np.float32),
        np.asarray(flow_start_times, dtype=np.float32),
        np.asarray(behavior_log_probs, dtype=np.float32),
        advantages,
        returns,
    )
    return (
        rollout,
        observation,
        completed_returns,
        completed_lengths,
        previous_action,
        has_previous_action,
    )


def evaluate(
    config: TrainConfig,
    policy: IntervalFlowPolicy,
    observation_stats: RunningMeanStd,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[float, float]:
    env = gym.make(config.env_id)
    scores: list[float] = []
    try:
        for episode in range(config.eval_episodes):
            raw_observation, _ = env.reset(seed=config.seed + 10_000 + episode)
            observation = observation_stats.normalize(raw_observation[None, :])
            previous_action = np.zeros((1, policy.action_dim), dtype=np.float32)
            has_previous = False
            done = False
            score = 0.0
            while not done:
                pure_noise = tf.zeros((1, policy.action_dim), tf.float32)
                if config.stochastic_eval:
                    pure_noise = tf.random.normal((1, policy.action_dim))
                if config.warm_start_time > 0.0 and has_previous:
                    start_time = np.asarray(
                        [[config.warm_start_time]], dtype=np.float32
                    )
                    flow_init = (
                        1.0 - config.warm_start_time
                    ) * pure_noise.numpy() + config.warm_start_time * previous_action
                else:
                    start_time = np.zeros((1, 1), dtype=np.float32)
                    flow_init = pure_noise.numpy()
                sample = policy.sample(
                    tf.convert_to_tensor(observation, tf.float32),
                    flow_init=tf.convert_to_tensor(flow_init, tf.float32),
                    flow_start=tf.convert_to_tensor(start_time, tf.float32),
                    deterministic=not config.stochastic_eval,
                    training=False,
                )
                previous_action[:] = sample.pre_tanh_action.numpy()
                has_previous = True
                env_action = action_from_pre_tanh(
                    previous_action[0], action_low, action_high
                )
                raw_observation, reward, terminated, truncated, _ = env.step(env_action)
                observation = observation_stats.normalize(raw_observation[None, :])
                score += float(reward)
                done = bool(terminated or truncated)
            scores.append(score)
    finally:
        env.close()
    return float(np.mean(scores)), float(np.std(scores))


def write_summaries(
    writer: tf.summary.SummaryWriter,
    metrics: dict[str, float],
    environment_steps: int,
) -> None:
    with writer.as_default():
        for name, value in metrics.items():
            tf.summary.scalar(name, value, step=environment_steps)
        writer.flush()


def run(config: TrainConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    env = make_vector_env(config.env_id, config.num_envs)
    observation_space = env.single_observation_space
    action_space = env.single_action_space
    if (
        not isinstance(observation_space, gym.spaces.Box)
        or len(observation_space.shape) != 1
    ):
        raise TypeError("this entry point requires a flat Box observation space")
    if not isinstance(action_space, gym.spaces.Box) or len(action_space.shape) != 1:
        raise TypeError("this entry point requires a flat Box action space")
    if not np.all(np.isfinite(action_space.low)) or not np.all(
        np.isfinite(action_space.high)
    ):
        raise ValueError("action bounds must be finite for the tanh transform")

    obs_dim = int(observation_space.shape[0])
    action_dim = int(action_space.shape[0])
    trainer = Trainer(config, obs_dim, action_dim)
    replay = ReplayWindow(config.replay_rollouts)
    observation_stats = RunningMeanStd((obs_dim,))

    raw_observation, _ = env.reset(seed=config.seed)
    observation_stats.update(raw_observation)
    observation = observation_stats.normalize(raw_observation)
    episode_returns = np.zeros(config.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(config.num_envs, dtype=np.int64)
    previous_action = np.zeros((config.num_envs, action_dim), dtype=np.float32)
    has_previous_action = np.zeros(config.num_envs, dtype=bool)

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config.env_id}-direct-ratio-recent-ofp-{timestamp}"
    log_path = Path(config.log_dir) / run_name
    writer = tf.summary.create_file_writer(str(log_path))
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    environment_steps = 0
    rollout_index = 0
    best_score = -float("inf")
    started = time.monotonic()
    try:
        while environment_steps < config.total_steps:
            (
                rollout,
                observation,
                completed_returns,
                completed_lengths,
                previous_action,
                has_previous_action,
            ) = collect_rollout(
                env,
                trainer,
                observation,
                observation_stats,
                action_space.low.astype(np.float32),
                action_space.high.astype(np.float32),
                episode_returns,
                episode_lengths,
                previous_action,
                has_previous_action,
            )
            replay.append(rollout)
            rollout_index += 1
            environment_steps += config.rollout_steps * config.num_envs

            metrics: dict[str, float] = {
                "replay/samples": float(len(replay)),
                "replay/rollouts": float(replay.rollout_count),
                "time/steps_per_second": environment_steps
                / max(time.monotonic() - started, 1e-6),
            }
            if completed_returns:
                metrics["rollout/episode_return"] = float(np.mean(completed_returns))
                metrics["rollout/episode_length"] = float(np.mean(completed_lengths))

            if replay.rollout_count >= config.warmup_rollouts:
                train_metrics = trainer.train_replay(replay, rng)
                metrics.update(
                    {f"train/{key}": value for key, value in train_metrics.items()}
                )

            if rollout_index % config.eval_every_rollouts == 0:
                score, score_std = evaluate(
                    config,
                    trainer.policy,
                    observation_stats,
                    action_space.low.astype(np.float32),
                    action_space.high.astype(np.float32),
                )
                metrics["eval/return"] = score
                metrics["eval/return_std"] = score_std
                if score > best_score:
                    best_score = score
                    if config.checkpoint_dir:
                        checkpoint_path = Path(config.checkpoint_dir)
                        checkpoint_path.mkdir(parents=True, exist_ok=True)
                        trainer.policy.save_weights(
                            str(checkpoint_path / f"{run_name}.weights.h5")
                        )

            write_summaries(writer, metrics, environment_steps)
            concise = {
                key: round(value, 4)
                for key, value in metrics.items()
                if key
                in {
                    "rollout/episode_return",
                    "train/actor_loss",
                    "train/critic_loss",
                    "train/ratio",
                    "train/recent_ratio",
                    "train/flow_loss",
                    "train/consistency_loss",
                    "eval/return",
                }
            }
            print(
                f"rollout={rollout_index} steps={environment_steps} metrics={concise}",
                flush=True,
            )
    finally:
        env.close()


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
