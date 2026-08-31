from __future__ import annotations

import unittest

import numpy as np

try:
    import envpool
    import gymnasium
except ImportError:  # pragma: no cover - optional dependency check
    envpool = None
    gymnasium = None


@unittest.skipIf(envpool is None, "EnvPool is not installed")
class TorchEnvPoolTests(unittest.TestCase):
    def test_sync_batch_backend_keeps_num_envs_and_gymnasium_api(self) -> None:
        from flow.torch_train import make_vector_env, reset_vector_env

        env = make_vector_env(
            "Pendulum-v1",
            2,
            backend="envpool",
            seed=13,
            envpool_num_threads=2,
        )
        try:
            self.assertEqual(env.num_envs, 2)
            self.assertIsInstance(env.single_observation_space, gymnasium.spaces.Box)
            observation, _ = reset_vector_env(env, backend="envpool", seed=13)
            self.assertEqual(observation.shape, (2, 3))
            action = np.zeros((2, 1), dtype=np.float32)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            self.assertEqual(next_observation.shape, (2, 3))
            self.assertEqual(reward.shape, (2,))
            self.assertEqual(terminated.shape, (2,))
            self.assertEqual(truncated.shape, (2,))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
