from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - dependency check for lightweight clones
    np = None


@unittest.skipIf(np is None, "NumPy is not installed")
class BufferTests(unittest.TestCase):
    def test_generalized_advantage_estimate(self) -> None:
        from flow.buffer import generalized_advantage_estimate

        rewards = np.asarray([[1.0], [1.0]], dtype=np.float32)
        dones = np.asarray([[0.0], [1.0]], dtype=np.float32)
        values = np.zeros_like(rewards)
        advantages, returns = generalized_advantage_estimate(
            rewards,
            dones,
            values,
            next_value=np.asarray([0.0], dtype=np.float32),
            gamma=1.0,
            gae_lambda=1.0,
        )
        np.testing.assert_allclose(advantages[:, 0], [2.0, 1.0])
        np.testing.assert_allclose(returns[:, 0], [2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
