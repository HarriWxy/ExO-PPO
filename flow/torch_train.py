"""Train direct-ratio ExO-PPO with a recent-policy one-step flow in PyTorch.

Run from the repository root::

    python -m flow.torch_train --env-id Walker2d-v5 --device auto

The TensorFlow entry point in :mod:`flow.train` and this module intentionally
share the NumPy replay/GAE implementation but have no runtime dependency on
each other.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from .buffer import (
    ReplayWindow,
    Rollout,
    RunningMeanStd,
    flatten_rollout,
    generalized_advantage_estimate,
)
from .torch_models import (
    IntervalFlowPolicy,
    ValueNetwork,
    build_policy_copy,
    update_ema,
)
from .torch_objectives import direct_ratio_exo_loss, recent_policy_ofp_losses


class _NullSummaryWriter:
    """Small fallback so training still works without the optional TensorBoard."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class TorchTrainConfig:
    env_id: str = "Walker2d-v5"
    env_backend: str = "envpool" # gymnasium
    seed: int = 0
    total_steps: int = 1_000_000
    num_envs: int = 16
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
    device: str = "auto"
    envpool_num_threads: int = 0
    log_dir: str = "logs"
    checkpoint_dir: str = ""


def _parse_hidden_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError(
            "hidden sizes must be comma-separated positive integers"
        )
    return sizes


def parse_args(argv: Sequence[str] | None = None) -> TorchTrainConfig:
    parser = argparse.ArgumentParser(
        description="PyTorch direct-ratio ExO-PPO + recent-policy one-step flow"
    )
    parser.add_argument("--env-id", default=TorchTrainConfig.env_id)
    parser.add_argument(
        "--env-backend",
        choices=("gymnasium", "envpool"),
        default=TorchTrainConfig.env_backend,
        help="parallel environment implementation",
    )
    parser.add_argument("--seed", type=int, default=TorchTrainConfig.seed)
    parser.add_argument("--total-steps", type=int, default=TorchTrainConfig.total_steps)
    parser.add_argument("--num-envs", type=int, default=TorchTrainConfig.num_envs)
    parser.add_argument(
        "--rollout-steps", type=int, default=TorchTrainConfig.rollout_steps
    )
    parser.add_argument(
        "--replay-rollouts", type=int, default=TorchTrainConfig.replay_rollouts
    )
    parser.add_argument(
        "--warmup-rollouts", type=int, default=TorchTrainConfig.warmup_rollouts
    )
    parser.add_argument(
        "--update-epochs", type=int, default=TorchTrainConfig.update_epochs
    )
    parser.add_argument("--batch-size", type=int, default=TorchTrainConfig.batch_size)
    parser.add_argument("--gamma", type=float, default=TorchTrainConfig.gamma)
    parser.add_argument("--gae-lambda", type=float, default=TorchTrainConfig.gae_lambda)

    parser.add_argument(
        "--hidden-sizes",
        type=_parse_hidden_sizes,
        default=TorchTrainConfig.hidden_sizes,
        help="comma-separated MLP widths",
    )
    parser.add_argument(
        "--initial-log-std",
        type=float,
        default=TorchTrainConfig.initial_log_std,
    )
    parser.add_argument(
        "--actor-lr",
        dest="actor_learning_rate",
        type=float,
        default=TorchTrainConfig.actor_learning_rate,
    )
    parser.add_argument(
        "--critic-lr",
        dest="critic_learning_rate",
        type=float,
        default=TorchTrainConfig.critic_learning_rate,
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=TorchTrainConfig.max_grad_norm
    )

    parser.add_argument(
        "--exo-clip-radius", type=float, default=TorchTrainConfig.exo_clip_radius
    )
    parser.add_argument("--exo-beta", type=float, default=TorchTrainConfig.exo_beta)
    parser.add_argument(
        "--entropy-coef",
        dest="entropy_coefficient",
        type=float,
        default=TorchTrainConfig.entropy_coefficient,
    )
    parser.add_argument(
        "--max-log-ratio", type=float, default=TorchTrainConfig.max_log_ratio
    )
    parser.add_argument("--target-kl", type=float, default=TorchTrainConfig.target_kl)

    parser.add_argument(
        "--ofp-coef",
        dest="ofp_coefficient",
        type=float,
        default=TorchTrainConfig.ofp_coefficient,
    )
    parser.add_argument("--flow-mix", type=float, default=TorchTrainConfig.flow_mix)
    parser.add_argument(
        "--consistency-mix",
        type=float,
        default=TorchTrainConfig.consistency_mix,
    )
    parser.add_argument(
        "--guidance-mix", type=float, default=TorchTrainConfig.guidance_mix
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=TorchTrainConfig.guidance_scale
    )
    parser.add_argument(
        "--condition-dropout",
        type=float,
        default=TorchTrainConfig.condition_dropout,
    )
    parser.add_argument(
        "--contraction-steps",
        type=int,
        default=TorchTrainConfig.contraction_steps,
    )
    parser.add_argument(
        "--minimum-contraction",
        type=float,
        default=TorchTrainConfig.minimum_contraction,
    )
    parser.add_argument(
        "--ema-initial-decay",
        type=float,
        default=TorchTrainConfig.ema_initial_decay,
    )
    parser.add_argument(
        "--ema-max-decay",
        type=float,
        default=TorchTrainConfig.ema_max_decay,
    )
    parser.add_argument(
        "--ema-ramp-steps", type=int, default=TorchTrainConfig.ema_ramp_steps
    )

    parser.add_argument(
        "--warm-start-time", type=float, default=TorchTrainConfig.warm_start_time
    )
    parser.add_argument(
        "--eval-every-rollouts",
        type=int,
        default=TorchTrainConfig.eval_every_rollouts,
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=TorchTrainConfig.eval_episodes
    )
    parser.add_argument("--stochastic-eval", action="store_true", default=False)
    parser.add_argument(
        "--device", default=TorchTrainConfig.device, help="auto, cpu, cuda, or cuda:N"
    )
    parser.add_argument(
        "--envpool-num-threads",
        type=int,
        default=TorchTrainConfig.envpool_num_threads,
        help="EnvPool worker threads; 0 uses EnvPool's batch-size default",
    )
    parser.add_argument("--log-dir", default=TorchTrainConfig.log_dir)
    parser.add_argument("--checkpoint-dir", default=TorchTrainConfig.checkpoint_dir)
    args = parser.parse_args(argv)
    config = TorchTrainConfig(**vars(args))
    validate_config(config)
    return config


def validate_config(config: TorchTrainConfig) -> None:
    if config.env_backend not in {"gymnasium", "envpool"}:
        raise ValueError("env_backend must be 'gymnasium' or 'envpool'")
    if config.envpool_num_threads < 0:
        raise ValueError("envpool_num_threads must be non-negative")
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


def resolve_device(value: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def make_vector_env(
    env_id: str,
    num_envs: int,
    *,
    backend: str = "gymnasium",
    seed: int = 0,
    envpool_num_threads: int = 0,
) -> Any:
    """Create the fixed-width batched environment used by the collector.

    ``envpool`` is deliberately created in synchronous mode by setting
    ``batch_size == num_envs``.  EnvPool still executes the batch in its native
    C++ thread pool, while the collector can keep one stable row per env for
    GAE, warm-start actions and episode accounting.  The asynchronous EnvPool
    API returns whichever env IDs finish first and would require a different
    replay layout.
    """

    if backend == "envpool":
        try:
            import envpool
        except ImportError as error:
            raise RuntimeError(
                "--env-backend envpool requires the optional 'envpool' package"
            ) from error
        kwargs: dict[str, Any] = {
            "env_type": "gymnasium",
            "num_envs": num_envs,
            "batch_size": num_envs,
            "seed": seed,
        }
        if envpool_num_threads > 0:
            kwargs["num_threads"] = envpool_num_threads
        try:
            env = envpool.make(env_id, **kwargs)
        except Exception as error:
            raise RuntimeError(
                f"EnvPool could not create {env_id!r}; check envpool.list_all_envs()"
            ) from error
        # Current EnvPool Gymnasium wrappers expose ``single_*`` aliases.  The
        # run loop also falls back to ``observation_space``/``action_space`` for
        # older wrappers, so no mutation of the extension object is needed here.
        return env

    if backend != "gymnasium":
        raise ValueError("backend must be 'gymnasium' or 'envpool'")

    # Gymnasium's synchronous vector env keeps a stable row per environment.

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


def reset_vector_env(env: Any, *, backend: str, seed: int) -> tuple[Any, Any]:
    """Reset either backend while tolerating older EnvPool return conventions."""

    if backend == "envpool":
        result = env.reset()
    else:
        result = env.reset(seed=seed)
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


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
    """PyTorch actor, recent snapshot, EMA teacher and critic."""

    def __init__(
        self,
        config: TorchTrainConfig,
        obs_dim: int,
        action_dim: int,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or resolve_device(config.device)
        self.policy = IntervalFlowPolicy(
            obs_dim,
            action_dim,
            hidden_sizes=config.hidden_sizes,
            initial_log_std=config.initial_log_std,
        ).to(self.device)
        dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32, device=self.device)
        dummy_noise = torch.zeros(
            (1, action_dim), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            self.policy.one_step_mean(dummy_obs, dummy_noise)
        self.recent_policy = build_policy_copy(self.policy, trainable=False)
        self.ema_teacher = build_policy_copy(self.policy, trainable=False)
        self.value = ValueNetwork(config.hidden_sizes).to(self.device)
        with torch.no_grad():
            self.value(dummy_obs)

        self.actor_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.actor_learning_rate, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=config.critic_learning_rate, eps=1e-5
        )
        self.update_step = 0

    def snapshot_recent_policy(self) -> None:
        """Freeze the proximal center for one complete replay update."""

        self.recent_policy.load_state_dict(self.policy.state_dict())
        self.recent_policy.eval()

    def _ema_decay(self) -> float:
        progress = self.update_step / float(self.config.ema_ramp_steps)
        progress = progress / (1.0 + progress)
        decay = self.config.ema_initial_decay + progress * (
            self.config.ema_max_decay - self.config.ema_initial_decay
        )
        return min(decay, self.config.ema_max_decay)

    def actor_train_step(
        self,
        observation: torch.Tensor,
        pre_tanh_action: torch.Tensor,
        flow_init: torch.Tensor,
        flow_start: torch.Tensor,
        behavior_log_prob: torch.Tensor,
        advantage: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.policy.train()
        self.actor_optimizer.zero_grad(set_to_none=True)
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
            zero = torch.zeros((), dtype=torch.float32, device=self.device)
            ofp = {
                "flow_loss": zero,
                "consistency_loss": zero,
                "guidance_loss": zero,
                "contraction": zero,
                "target_action_rms": zero,
            }
            self_distill_loss = zero
        total_loss = exo["policy_loss"] + self_distill_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("non-finite actor loss")
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        ).to(self.device)
        self.actor_optimizer.step()
        ema_decay = self._ema_decay()
        update_ema(self.ema_teacher, self.policy, ema_decay)
        self.update_step += 1
        return {
            **exo,
            **ofp,
            "actor_loss": total_loss.detach(),
            "self_distill_loss": self_distill_loss.detach(),
            "actor_grad_norm": torch.as_tensor(
                gradient_norm, dtype=torch.float32, device=self.device
            ).detach(),
            "ema_decay": torch.as_tensor(ema_decay, device=self.device),
        }

    def critic_train_step(
        self, observation: torch.Tensor, returns: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self.value.train()
        self.critic_optimizer.zero_grad(set_to_none=True)
        prediction = self.value(observation)
        critic_loss = F.smooth_l1_loss(prediction, returns, reduction="mean")
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("non-finite critic loss")
        critic_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.value.parameters(), self.config.max_grad_norm
        ).to(self.device)
        self.critic_optimizer.step()
        return {
            "critic_loss": critic_loss.detach(),
            "critic_grad_norm": torch.as_tensor(
                gradient_norm, device=self.device
            ).detach(),
            "value_mean": prediction.detach().mean(),
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

                def tensor(name: str) -> torch.Tensor:
                    return torch.as_tensor(
                        arrays[name][indices], dtype=torch.float32, device=self.device
                    )

                observation = tensor("observations")
                actor_metrics = self.actor_train_step(
                    observation,
                    tensor("pre_tanh_actions"),
                    tensor("flow_init"),
                    tensor("flow_start"),
                    tensor("behavior_log_prob"),
                    tensor("advantages"),
                )
                critic_metrics = self.critic_train_step(observation, tensor("returns"))
                for key, value in {**actor_metrics, **critic_metrics}.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(
                        value.detach().cpu()
                    )
                metric_count += 1

                if (
                    self.config.target_kl > 0.0
                    and float(actor_metrics["recent_log_shift"].detach().cpu())
                    > self.config.target_kl
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
    env: Any,
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

    trainer.policy.eval()
    trainer.value.eval()
    for _ in range(config.rollout_steps):
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=trainer.device
        )
        batch_size = observation.shape[0]
        pure_noise = torch.randn(
            (batch_size, trainer.policy.action_dim),
            dtype=torch.float32,
            device=trainer.device,
        )
        if config.warm_start_time > 0.0:
            active = torch.as_tensor(
                has_previous_action.astype(np.float32),
                dtype=torch.float32,
                device=trainer.device,
            ).reshape(batch_size, 1)
            start_tensor = active * config.warm_start_time
            previous_tensor = torch.as_tensor(
                previous_action, dtype=torch.float32, device=trainer.device
            )
            flow_init_tensor = (
                1.0 - start_tensor
            ) * pure_noise + start_tensor * previous_tensor
        else:
            start_tensor = torch.zeros(
                (batch_size, 1), dtype=torch.float32, device=trainer.device
            )
            flow_init_tensor = pure_noise

        with torch.no_grad():
            sample = trainer.policy.sample(
                observation_tensor,
                flow_init=flow_init_tensor,
                flow_start=start_tensor,
                deterministic=False,
            )
            value = trainer.value(observation_tensor)
        pre_tanh = sample.pre_tanh_action.cpu().numpy()
        env_action = action_from_pre_tanh(pre_tanh, action_low, action_high)
        next_raw_observation, reward, terminated, truncated, _ = env.step(env_action)
        done = np.logical_or(terminated, truncated)

        observations.append(observation.copy())
        pre_tanh_actions.append(pre_tanh)
        flow_initializations.append(sample.flow_init.cpu().numpy())
        flow_start_times.append(sample.flow_start.cpu().numpy())
        behavior_log_probs.append(sample.log_prob.cpu().numpy())
        rewards.append(np.asarray(reward, dtype=np.float32))
        dones.append(np.asarray(done, dtype=np.float32))
        values.append(value.cpu().numpy())

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

    with torch.no_grad():
        next_value = (
            trainer.value(
                torch.as_tensor(observation, dtype=torch.float32, device=trainer.device)
            )
            .cpu()
            .numpy()
        )
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
    config: TorchTrainConfig,
    policy: IntervalFlowPolicy,
    observation_stats: RunningMeanStd,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[float, float]:
    env = gym.make(config.env_id)
    scores: list[float] = []
    policy.eval()
    try:
        device = next(policy.parameters()).device
        for episode in range(config.eval_episodes):
            raw_observation, _ = env.reset(seed=config.seed + 10_000 + episode)
            observation = observation_stats.normalize(raw_observation[None, :])
            previous_action = np.zeros((1, policy.action_dim), dtype=np.float32)
            has_previous = False
            done = False
            score = 0.0
            while not done:
                pure_noise = torch.zeros(
                    (1, policy.action_dim), dtype=torch.float32, device=device
                )
                if config.stochastic_eval:
                    pure_noise = torch.randn_like(pure_noise)
                if config.warm_start_time > 0.0 and has_previous:
                    start_time = np.asarray(
                        [[config.warm_start_time]], dtype=np.float32
                    )
                    flow_init = (
                        (1.0 - config.warm_start_time) * pure_noise.cpu().numpy()
                        + config.warm_start_time * previous_action
                    )
                else:
                    start_time = np.zeros((1, 1), dtype=np.float32)
                    flow_init = pure_noise.cpu().numpy()
                with torch.no_grad():
                    sample = policy.sample(
                        torch.as_tensor(
                            observation, dtype=torch.float32, device=device
                        ),
                        flow_init=torch.as_tensor(
                            flow_init, dtype=torch.float32, device=device
                        ),
                        flow_start=torch.as_tensor(
                            start_time, dtype=torch.float32, device=device
                        ),
                        deterministic=not config.stochastic_eval,
                    )
                previous_action[:] = sample.pre_tanh_action.cpu().numpy()
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


def run(config: TorchTrainConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed)
    device = resolve_device(config.device)

    env = make_vector_env(
        config.env_id,
        config.num_envs,
        backend=config.env_backend,
        seed=config.seed,
        envpool_num_threads=config.envpool_num_threads,
    )
    observation_space = getattr(env, "single_observation_space", env.observation_space)
    action_space = getattr(env, "single_action_space", env.action_space)
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
    trainer = Trainer(config, obs_dim, action_dim, device=device)
    replay = ReplayWindow(config.replay_rollouts)
    observation_stats = RunningMeanStd((obs_dim,))

    raw_observation, _ = reset_vector_env(
        env, backend=config.env_backend, seed=config.seed
    )
    observation_stats.update(raw_observation)
    observation = observation_stats.normalize(raw_observation)
    episode_returns = np.zeros(config.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(config.num_envs, dtype=np.int64)
    previous_action = np.zeros((config.num_envs, action_dim), dtype=np.float32)
    has_previous_action = np.zeros(config.num_envs, dtype=bool)

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"{config.env_id}-torch-{config.env_backend}-direct-ratio-recent-ofp-"
        f"{timestamp}"
    )
    log_path = Path(config.log_dir) / run_name
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # pragma: no cover - depends on the local install
        SummaryWriter = _NullSummaryWriter
        print(
            "warning: tensorboard is not installed; continuing without event logs",
            flush=True,
        )
    writer = SummaryWriter(log_dir=str(log_path))

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
                        torch.save(
                            {
                                "policy": trainer.policy.state_dict(),
                                "value": trainer.value.state_dict(),
                                "update_step": trainer.update_step,
                                "observation_mean": observation_stats.mean,
                                "observation_var": observation_stats.var,
                                "observation_count": observation_stats.count,
                                "config": asdict(config),
                            },
                            checkpoint_path / f"{run_name}.pt",
                        )

            for name, value in metrics.items():
                writer.add_scalar(name, value, environment_steps)
            writer.flush()
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
        writer.close()
        env.close()


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
