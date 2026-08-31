from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - dependency check for lightweight clones
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchFlowObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        from flow.torch_models import IntervalFlowPolicy, build_policy_copy

        torch.manual_seed(7)
        self.policy = IntervalFlowPolicy(3, 2, hidden_sizes=(16, 16))
        observation = torch.zeros((4, 3))
        noise = torch.zeros((4, 2))
        self.policy.one_step_mean(observation, noise)
        self.recent = build_policy_copy(self.policy)
        self.teacher = build_policy_copy(self.policy)

    def test_same_policy_has_unit_direct_and_recent_ratios(self) -> None:
        from flow.torch_objectives import direct_ratio_exo_loss

        observation = torch.randn((4, 3))
        flow_init = torch.randn((4, 2))
        flow_start = torch.zeros((4, 1))
        sample = self.policy.sample(
            observation, flow_init=flow_init, flow_start=flow_start
        )
        result = direct_ratio_exo_loss(
            self.policy,
            self.recent,
            observation,
            sample.pre_tanh_action,
            sample.flow_init,
            sample.flow_start,
            sample.log_prob,
            torch.ones(4),
            clip_radius=0.2,
            beta=5.0,
        )
        self.assertAlmostEqual(float(result["ratio"]), 1.0, places=5)
        self.assertAlmostEqual(float(result["recent_ratio"]), 1.0, places=5)

    def test_policy_broadcasts_singleton_times_and_condition_masks(self) -> None:
        observation = torch.randn((4, 3))
        state = torch.randn((4, 2))
        with torch.no_grad():
            self.policy.velocity_head.weight.fill_(1.0)
        reference = self.policy.velocity(
            observation,
            state,
            torch.full((4, 1), 0.25),
            torch.full((4, 1), 0.75),
            condition_mask=torch.zeros((4, 1)),
        )
        for broadcast_start, broadcast_end, broadcast_mask in (
            (0.25, 0.75, 0.0),
            (torch.tensor([[0.25]]), torch.tensor([[0.75]]), torch.tensor([[0.0]])),
        ):
            actual = self.policy.velocity(
                observation,
                state,
                broadcast_start,
                broadcast_end,
                condition_mask=broadcast_mask,
            )
            self.assertEqual(tuple(actual.shape), (4, 2))
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(torch.allclose(actual, reference, atol=1e-6))
        interval_result = self.policy.interval_mean(observation, state, 0.25)
        self.assertEqual(tuple(interval_result.shape), (4, 2))

    def test_recent_ratio_stays_fixed_when_online_policy_moves(self) -> None:
        from flow.torch_objectives import direct_ratio_exo_loss

        observation = torch.randn((16, 3))
        flow_init = torch.randn((16, 2))
        flow_start = torch.zeros((16, 1))
        sample = self.policy.sample(
            observation,
            flow_init=flow_init,
            flow_start=flow_start,
            residual_noise=torch.zeros((16, 2)),
        )
        with torch.no_grad():
            self.policy.velocity_head.bias.add_(torch.tensor([0.3, -0.2]))
        result = direct_ratio_exo_loss(
            self.policy,
            self.recent,
            observation,
            sample.pre_tanh_action,
            sample.flow_init,
            sample.flow_start,
            sample.log_prob,
            torch.ones(16),
            clip_radius=0.2,
            beta=5.0,
        )
        self.assertAlmostEqual(float(result["recent_ratio"]), 1.0, places=5)
        self.assertGreater(float(result["recent_log_shift"]), 0.01)

    def test_exo_transform_is_identity_inside_recent_band(self) -> None:
        from flow.torch_objectives import smooth_exo_ratio

        ratio = torch.tensor([0.9, 1.0, 1.1])
        transformed = smooth_exo_ratio(ratio, torch.ones_like(ratio), 0.2, 5.0)
        self.assertTrue(torch.allclose(transformed, ratio, atol=1e-6))

    def test_exo_transform_and_gradient_are_finite_for_extreme_ratios(self) -> None:
        from flow.torch_objectives import smooth_exo_ratio

        ratio = torch.tensor([1e-8, 1.0, 1e8], requires_grad=True)
        center = torch.tensor([1e8, 1.0, 1e-8])
        transformed = smooth_exo_ratio(ratio, center, 0.2, 5.0)
        transformed.sum().backward()
        self.assertTrue(torch.isfinite(transformed).all())
        self.assertTrue(torch.isfinite(ratio.grad).all())

    def test_ofp_losses_have_finite_policy_gradients(self) -> None:
        from flow.torch_objectives import recent_policy_ofp_losses

        observation = torch.randn((8, 3))
        with torch.no_grad():
            self.policy.velocity_head.bias.add_(torch.tensor([0.25, -0.10]))
        losses = recent_policy_ofp_losses(
            self.policy,
            self.teacher,
            self.recent,
            observation,
            update_step=10,
            contraction_steps=100,
        )
        total = (
            losses["flow_loss"] + losses["consistency_loss"] + losses["guidance_loss"]
        )
        total.backward()
        gradients = [
            parameter.grad
            for parameter in self.policy.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertGreater(float(losses["flow_loss"]), 0.0)
        global_norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
        self.assertGreater(float(global_norm), 0.0)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_value_network_accepts_flat_observations(self) -> None:
        from flow.torch_models import ValueNetwork

        value = ValueNetwork((8, 8))
        prediction = value(torch.randn((5, 3)))
        self.assertEqual(tuple(prediction.shape), (5,))
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
