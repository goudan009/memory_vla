"""A native time-limit signal must stop the chunk immediately."""

import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np


def test_native_truncation_stops_before_remaining_chunk_actions():
    source = Path('/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1/gr00t/eval/sim/wrapper/multistep_wrapper.py')
    spec = importlib.util.spec_from_file_location('native_timeout_wrapper', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Env(gym.Env):
        observation_space = gym.spaces.Dict({'state.x': gym.spaces.Box(0, 100, (1,), dtype=np.float32)})
        action_space = gym.spaces.Dict({'action.x': gym.spaces.Box(-1, 1, (1,), dtype=np.float32)})
        def reset(self, **kwargs):
            self.steps = 0
            return {'state.x': np.zeros(1, np.float32)}, {}
        def step(self, action):
            self.steps += 1
            return {'state.x': np.array([self.steps], np.float32)}, 0, False, self.steps >= 10, {'success': False}

    base = Env()
    env = module.MultiStepWrapper(base, np.array([0]), np.array([0]), n_action_steps=8,
                                  max_episode_steps=999999999)
    env.reset()
    action = {'action.x': np.zeros((8, 1), np.float32)}
    assert env.step(action)[3] is False
    _, _, terminated, truncated, info = env.step(action)
    assert truncated and not terminated
    assert base.steps == 10
    assert len(info['rewards']) == 2
