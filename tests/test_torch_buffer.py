from __future__ import annotations

import unittest
from dataclasses import replace

try:
    import torch
except ImportError:  # pragma: no cover - dependency check for lightweight clones
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchBufferTests(unittest.TestCase):
    def test_timeout_bootstrap_is_separate_from_gae_continuation(self) -> None:
        from flow.torch_buffer import generalized_advantage_estimate

        rewards = torch.ones((2, 1))
        terminated = torch.zeros((2, 1), dtype=torch.bool)
        truncated = torch.tensor([[False], [True]])
        values = torch.zeros_like(rewards)
        next_values = torch.tensor([[0.0], [2.0]])

        bootstrapped, _ = generalized_advantage_estimate(
            rewards,
            terminated,
            truncated,
            values,
            next_values,
            gamma=1.0,
            gae_lambda=1.0,
            bootstrap_truncated=True,
        )
        finite_horizon, _ = generalized_advantage_estimate(
            rewards,
            terminated,
            truncated,
            values,
            next_values,
            gamma=1.0,
            gae_lambda=1.0,
            bootstrap_truncated=False,
        )

        self.assertTrue(torch.equal(bootstrapped[:, 0], torch.tensor([4.0, 3.0])))
        self.assertTrue(torch.equal(finite_horizon[:, 0], torch.tensor([2.0, 1.0])))

    def test_running_moments_round_trip(self) -> None:
        from flow.torch_buffer import TorchRunningMeanStd

        normalizer = TorchRunningMeanStd((2,))
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        normalizer.update(values)
        state = normalizer.state_dict()

        restored = TorchRunningMeanStd((2,))
        restored.load_state_dict(state)
        self.assertTrue(
            torch.allclose(normalizer.normalize(values), restored.normalize(values))
        )

    def test_device_replay_supports_asymmetric_critic_observations(self) -> None:
        from flow.torch_buffer import TorchReplayWindow, TorchRollout
        from flow.torch_train import TorchTrainConfig, Trainer

        config = replace(
            TorchTrainConfig(),
            hidden_sizes=(8, 8),
            update_epochs=1,
            batch_size=4,
            ofp_coefficient=0.0,
        )
        trainer = Trainer(
            config,
            obs_dim=3,
            action_dim=2,
            critic_obs_dim=4,
            device=torch.device("cpu"),
        )
        actor_observations = torch.randn((8, 3))
        critic_observations = torch.randn((8, 4))
        with torch.no_grad():
            sample = trainer.policy.sample(actor_observations)
        replay = TorchReplayWindow(max_rollouts=1)
        replay.append(
            TorchRollout(
                actor_observations=actor_observations,
                critic_observations=critic_observations,
                pre_tanh_actions=sample.pre_tanh_action,
                flow_init=sample.flow_init,
                flow_start=sample.flow_start,
                behavior_log_prob=sample.log_prob,
                advantages=torch.randn(8),
                returns=torch.randn(8),
            )
        )

        metrics = trainer.train_torch_replay(replay)
        self.assertIn("actor_loss", metrics)
        self.assertIn("critic_loss", metrics)
        self.assertTrue(
            all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        )


if __name__ == "__main__":
    unittest.main()
