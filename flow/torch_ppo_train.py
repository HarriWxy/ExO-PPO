"""Train a standard PPO baseline with the flow experiment's shared framework.

This entry point intentionally reuses the environment, rollout collector,
observation normalization, evaluation, checkpoint, TensorBoard, and CLI
framework from :mod:`flow.torch_train`.  The policy is the same
``IntervalFlowPolicy``; only the actor objective is changed to the standard
PPO clipped surrogate.

The shared CLI keeps ``--exo-clip-radius`` as the clip-width argument so that
the baseline can be launched with the same command-line configuration as the
ExO run.  In this module it is interpreted as PPO's epsilon.  The other
ExO/OFP-specific arguments are accepted for CLI compatibility but are not
used by the PPO objective.

Run from the repository root::

    python -m flow.torch_ppo_train --env-id Walker2d-v5 --device auto

With ``--replay-N 1 --warmup-rollouts 1`` this is the usual on-policy PPO
update.  Keeping the replay-size option equal to the ExO run gives a
replay-compatible PPO baseline, which is useful when comparing the two
objectives under the same data pipeline.
"""

from __future__ import annotations

from dataclasses import asdict
import datetime as dt
import json
from pathlib import Path
import random
import sys
import time
from typing import Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from .buffer import ReplayWindow, RunningMeanStd
from .torch_models import IntervalFlowPolicy, ValueNetwork
from .torch_train import (
    TorchTrainConfig,
    _NullSummaryWriter,
    action_from_pre_tanh,
    collect_rollout,
    evaluate,
    make_vector_env,
    parse_args as parse_shared_args,
    reset_vector_env,
    resolve_device,
)


class PPOTrainer:
    """Actor-critic trainer using the standard PPO clipped objective."""

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
        self.value = ValueNetwork(config.hidden_sizes).to(self.device)

        # Materialize lazy/value layers before the first optimizer step.
        dummy_obs = torch.zeros(
            (1, obs_dim), dtype=torch.float32, device=self.device
        )
        dummy_noise = torch.zeros(
            (1, action_dim), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            self.policy.one_step_mean(dummy_obs, dummy_noise)
            self.value(dummy_obs)

        self.actor_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.actor_learning_rate, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=config.critic_learning_rate, eps=1e-5
        )
        self.update_step = 0

    def actor_train_step(
        self,
        observation: torch.Tensor,
        pre_tanh_action: torch.Tensor,
        flow_init: torch.Tensor,
        flow_start: torch.Tensor,
        behavior_log_prob: torch.Tensor,
        advantage: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply one minibatch update of the standard PPO actor loss."""

        self.policy.train()
        self.actor_optimizer.zero_grad(set_to_none=True)
        current_log_prob = self.policy.conditional_log_prob(
            observation, pre_tanh_action, flow_init, flow_start
        )
        behavior_log_prob = torch.as_tensor(
            behavior_log_prob, dtype=torch.float32, device=self.device
        )
        advantage = torch.as_tensor(
            advantage, dtype=torch.float32, device=self.device
        )

        # The flow latent and pre-tanh action are stored in the rollout, so the
        # conditional augmented-policy ratio is tractable and exact.
        log_ratio = (current_log_prob - behavior_log_prob).clamp(
            -self.config.max_log_ratio, self.config.max_log_ratio
        )
        ratio = log_ratio.exp()
        clipped_ratio = ratio.clamp(
            1.0 - self.config.exo_clip_radius,
            1.0 + self.config.exo_clip_radius,
        )
        unclipped_surrogate = ratio * advantage
        clipped_surrogate = clipped_ratio * advantage
        surrogate = torch.minimum(unclipped_surrogate, clipped_surrogate)
        policy_loss = -surrogate.mean()
        entropy = self.policy.conditional_entropy()
        actor_loss = policy_loss - self.config.entropy_coefficient * entropy

        if not torch.isfinite(actor_loss):
            raise FloatingPointError("non-finite PPO actor loss")
        actor_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        )
        self.actor_optimizer.step()
        self.update_step += 1

        gradient_norm = torch.as_tensor(
            gradient_norm, dtype=torch.float32, device=self.device
        ).detach()
        return {
            "actor_loss": actor_loss.detach(),
            "policy_loss": policy_loss.detach(),
            "ratio": ratio.detach().mean(),
            "ratio_abs_log": log_ratio.detach().abs().mean(),
            # This non-negative first-order KL estimate is more stable for
            # target-KL early stopping than simply averaging -log_ratio.
            "approx_kl": (ratio - 1.0 - log_ratio).detach().mean(),
            "clip_fraction": (
                (ratio - clipped_ratio).detach().abs() > 1e-8
            ).float().mean(),
            "entropy": entropy.detach(),
            "actor_grad_norm": gradient_norm,
        }

    def critic_train_step(
        self, observation: torch.Tensor, returns: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Apply one minibatch update of the value network."""

        self.value.train()
        self.critic_optimizer.zero_grad(set_to_none=True)
        prediction = self.value(observation)
        critic_loss = F.smooth_l1_loss(prediction, returns, reduction="mean")
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("non-finite PPO critic loss")
        critic_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.value.parameters(), self.config.max_grad_norm
        )
        self.critic_optimizer.step()
        return {
            "critic_loss": critic_loss.detach(),
            "critic_grad_norm": torch.as_tensor(
                gradient_norm, dtype=torch.float32, device=self.device
            ).detach(),
            "value_mean": prediction.detach().mean(),
        }

    def train_replay(
        self, replay: ReplayWindow, rng: np.random.Generator
    ) -> dict[str, float]:
        """Optimize PPO on the current replay window.

        With ``replay.max_rollouts == 1`` this is on-policy PPO.  A larger
        window intentionally matches the ExO trainer's data pipeline; the
        stored behavior log probability makes the ratio well-defined, but the
        resulting baseline should be described as replay-compatible PPO.
        """

        arrays = replay.arrays()
        advantages = arrays["advantages"]
        arrays["advantages"] = (
            (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ).astype(np.float32)

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
                        arrays[name][indices],
                        dtype=torch.float32,
                        device=self.device,
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
                critic_metrics = self.critic_train_step(
                    observation, tensor("returns")
                )
                for key, value in {**actor_metrics, **critic_metrics}.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(
                        value.detach().cpu()
                    )
                metric_count += 1

                if (
                    self.config.target_kl > 0.0
                    and float(actor_metrics["approx_kl"].detach().cpu())
                    > self.config.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break

        averaged = {
            key: value / max(metric_count, 1)
            for key, value in metric_sums.items()
        }
        averaged["early_stop"] = float(stop_early)
        averaged["minibatches"] = float(metric_count)
        return averaged


def parse_args(argv: Sequence[str] | None = None) -> TorchTrainConfig:
    """Use the shared CLI, accepting both replay option names."""

    args = list(sys.argv[1:] if argv is None else argv)
    if hasattr(TorchTrainConfig, "replay_N"):
        source, target = "--replay-rollouts", "--replay-N"
    else:
        source, target = "--replay-N", "--replay-rollouts"
    for index, argument in enumerate(args):
        if argument == source:
            args[index] = target
        elif argument.startswith(source + "="):
            args[index] = target + argument[len(source) :]
    return parse_shared_args(args)


def _replay_capacity(config: TorchTrainConfig) -> int:
    """Read the replay-size field from either shared config revision."""

    if hasattr(config, "replay_N"):
        return int(config.replay_N)
    return int(config.replay_rollouts)


def _replay_option_name() -> str:
    return "--replay-N" if hasattr(TorchTrainConfig, "replay_N") else "--replay-rollouts"


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
    observation_space = getattr(
        env, "single_observation_space", env.observation_space
    )
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
    trainer = PPOTrainer(config, obs_dim, action_dim, device=device)
    replay_capacity = _replay_capacity(config)
    replay = ReplayWindow(replay_capacity)
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
        f"{config.env_id}/torch-{config.env_backend}-"
        f"replayN:{replay_capacity}-ppo-{timestamp}"
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

    if replay_capacity > 1:
        print(
            f"info: {_replay_option_name()} > 1 uses replay-compatible PPO; set "
            f"{_replay_option_name()} 1 --warmup-rollouts 1 for on-policy PPO",
            flush=True,
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
                metrics["rollout/episode_return"] = float(
                    np.mean(completed_returns)
                )
                metrics["rollout/episode_length"] = float(
                    np.mean(completed_lengths)
                )

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
                        checkpoint_name = run_name.replace("/", "_") + ".pt"
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
                            checkpoint_path / checkpoint_name,
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
                    "train/approx_kl",
                    "train/clip_fraction",
                    "eval/return",
                }
            }
            print(
                f"rollout={rollout_index} steps={environment_steps} "
                f"metrics={concise}",
                flush=True,
            )
    finally:
        writer.close()
        env.close()


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
