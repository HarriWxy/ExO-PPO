from __future__ import annotations

import unittest

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - dependency check for lightweight clones
    tf = None


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class FlowObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        from flow.models import IntervalFlowPolicy, build_policy_copy

        tf.random.set_seed(7)
        self.policy = IntervalFlowPolicy(3, 2, hidden_sizes=(16, 16))
        observation = tf.zeros((4, 3), tf.float32)
        noise = tf.zeros((4, 2), tf.float32)
        self.policy.one_step_mean(observation, noise)
        self.recent = build_policy_copy(self.policy, "test_recent")
        self.teacher = build_policy_copy(self.policy, "test_teacher")

    def test_same_policy_has_unit_direct_and_recent_ratios(self) -> None:
        from flow.objectives import direct_ratio_exo_loss

        observation = tf.random.normal((4, 3))
        flow_init = tf.random.normal((4, 2))
        flow_start = tf.zeros((4, 1))
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
            tf.ones((4,), tf.float32),
            clip_radius=0.2,
            beta=5.0,
        )
        self.assertAlmostEqual(float(result["ratio"]), 1.0, places=5)
        self.assertAlmostEqual(float(result["recent_ratio"]), 1.0, places=5)

    def test_policy_broadcasts_singleton_times_and_condition_masks(self) -> None:
        observation = tf.random.normal((4, 3))
        state = tf.random.normal((4, 2))
        self.policy.velocity_head.kernel.assign(
            tf.ones_like(self.policy.velocity_head.kernel)
        )
        start_time = tf.fill((4, 1), tf.constant(0.25, tf.float32))
        end_time = tf.fill((4, 1), tf.constant(0.75, tf.float32))
        condition_mask = tf.zeros((4, 1), tf.float32)
        reference = self.policy.velocity(
            observation,
            state,
            start_time,
            end_time,
            condition_mask=condition_mask,
        )

        for broadcast_start, broadcast_end, broadcast_mask in (
            (0.25, 0.75, 0.0),
            (tf.constant([[0.25]]), tf.constant([[0.75]]), tf.constant([[0.0]])),
        ):
            actual = self.policy.velocity(
                observation,
                state,
                broadcast_start,
                broadcast_end,
                condition_mask=broadcast_mask,
            )
            self.assertEqual(tuple(actual.shape), (4, 2))
            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(actual))))
            self.assertTrue(bool(tf.reduce_all(tf.abs(actual - reference) < 1e-6)))

        interval_result = self.policy.interval_mean(observation, state, 0.25)
        self.assertEqual(tuple(interval_result.shape), (4, 2))

    def test_recent_ratio_stays_fixed_when_online_policy_moves(self) -> None:
        from flow.objectives import direct_ratio_exo_loss

        observation = tf.random.normal((16, 3))
        flow_init = tf.random.normal((16, 2))
        flow_start = tf.zeros((16, 1))
        behavior_sample = self.policy.sample(
            observation,
            flow_init=flow_init,
            flow_start=flow_start,
            residual_noise=tf.zeros((16, 2)),
        )
        self.policy.velocity_head.bias.assign_add(tf.constant([0.3, -0.2], tf.float32))
        result = direct_ratio_exo_loss(
            self.policy,
            self.recent,
            observation,
            behavior_sample.pre_tanh_action,
            behavior_sample.flow_init,
            behavior_sample.flow_start,
            behavior_sample.log_prob,
            tf.ones((16,), tf.float32),
            clip_radius=0.2,
            beta=5.0,
        )
        self.assertAlmostEqual(float(result["recent_ratio"]), 1.0, places=5)
        self.assertGreater(float(result["recent_log_shift"]), 0.01)

    def test_exo_transform_is_identity_inside_recent_band(self) -> None:
        from flow.objectives import smooth_exo_ratio

        ratio = tf.constant([0.9, 1.0, 1.1], tf.float32)
        center = tf.ones_like(ratio)
        transformed = smooth_exo_ratio(ratio, center, 0.2, 5.0)
        self.assertTrue(bool(tf.reduce_all(tf.abs(transformed - ratio) < 1e-6)))

    def test_exo_transform_and_gradient_are_finite_for_extreme_ratios(self) -> None:
        from flow.objectives import smooth_exo_ratio

        ratio = tf.Variable([1e-8, 1.0, 1e8], dtype=tf.float32)
        center = tf.constant([1e8, 1.0, 1e-8], tf.float32)
        with tf.GradientTape() as tape:
            transformed = smooth_exo_ratio(ratio, center, 0.2, 5.0)
            loss = tf.reduce_sum(transformed)
        gradient = tape.gradient(loss, ratio)
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(transformed))))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(gradient))))

    def test_ofp_losses_have_finite_policy_gradients(self) -> None:
        from flow.objectives import recent_policy_ofp_losses

        observation = tf.random.normal((8, 3))
        self.policy.velocity_head.bias.assign_add(
            tf.constant([0.25, -0.10], tf.float32)
        )
        with tf.GradientTape() as tape:
            losses = recent_policy_ofp_losses(
                self.policy,
                self.teacher,
                self.recent,
                observation,
                update_step=tf.constant(10, tf.int64),
                contraction_steps=100,
            )
            total = (
                losses["flow_loss"]
                + losses["consistency_loss"]
                + losses["guidance_loss"]
            )
        gradients = tape.gradient(total, self.policy.trainable_variables)
        non_null = [gradient for gradient in gradients if gradient is not None]
        self.assertTrue(non_null)
        self.assertGreater(float(losses["flow_loss"]), 0.0)
        self.assertGreater(float(tf.linalg.global_norm(non_null)), 0.0)
        self.assertTrue(
            bool(
                tf.reduce_all(
                    [
                        tf.reduce_all(tf.math.is_finite(gradient))
                        for gradient in non_null
                    ]
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
