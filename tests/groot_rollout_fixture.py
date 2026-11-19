import gymnasium as gym
import numpy as np


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)

    def __init__(self, env_idx=0, total_n_envs=1, instance_count=10):
        self._all_test_instances = list(range(instance_count))
        per_env = len(self._all_test_instances) // total_n_envs
        start = env_idx * per_env
        end = len(self._all_test_instances) if env_idx == total_n_envs - 1 else start + per_env
        self._instance_indices_this_env = list(range(start, end))
        self._instance_idx_pointer = 0

    def reset(self, **kwargs):
        self._instance_idx_pointer = (self._instance_idx_pointer + 1) % len(self._instance_indices_this_env)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0, True, False, {
            "success": [True], "valid": [True], "task_progress": [75.0], "rewards": [0],
        }


def create_tiny_env(env_name, env_idx, total_n_envs, wrapper_configs):
    return TinyEnv(env_idx, total_n_envs)


class Policy:
    def reset(self, options=None):
        pass

    def get_action(self, observations, options=None):
        return np.zeros((observations.shape[0], 1), dtype=np.float32), {}


class UnevenEnv(TinyEnv):
    def reset(self, **kwargs):
        self.steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        self.steps += 1
        # First partition has one-step episodes, second has three-step episodes.
        ended = self.steps >= (1 if self._instance_indices_this_env[0] == 0 else 3)
        return np.zeros(1, dtype=np.float32), 0, ended, False, {
            'success': [True], 'valid': [True], 'task_progress': [75.0], 'rewards': [0],
        }


def create_uneven_env(env_name, env_idx, total_n_envs, wrapper_configs):
    import os
    return UnevenEnv(env_idx, total_n_envs, int(os.environ.get('GROOT_TEST_INSTANCE_COUNT', '10')))
