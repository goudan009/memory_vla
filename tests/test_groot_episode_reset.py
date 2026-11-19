"""A stateful policy must see reset observations before its first new-episode action."""

import os
from pathlib import Path
import subprocess


def test_next_step_autoreset_does_not_pollute_new_episode_memory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    repo = Path('/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(root / 'scripts'), str(repo)])
    code = r'''
import gymnasium as gym
import numpy as np
from gr00t.eval import rollout_policy as r

class Env(gym.Env):
    observation_space = gym.spaces.Box(0, 1, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(0, 1, (1,), dtype=np.float32)
    def reset(self, **kwargs):
        self.steps = 0
        return np.zeros(1, np.float32), {}
    def step(self, action):
        self.steps += 1
        return np.ones(1, np.float32), 0, self.steps == 2, False, {'success': [False]}

class Policy:
    calls = 0
    resets = []
    def reset(self, options=None):
        self.first = True
        self.resets.append(options)
    def get_action(self, obs, options=None):
        if self.first:
            assert not obs.any(), 'terminal observation contaminated next episode'
        self.first = False
        self.calls += 1
        return np.zeros((1, 1), np.float32), {}

r.create_eval_env = lambda **kwargs: Env()
p = Policy()
r.run_rollout_gymnasium_policy('sim_behavior_r1_pro/tiny', p, r.WrapperConfigs(), n_episodes=2, n_envs=1)
assert p.calls == 4
assert [x['episode_ids'][0].rsplit('/', 1)[-1] for x in p.resets] == ['episode0', 'episode1', 'episode2']
'''
    result = subprocess.run(['/nvme3/envs/b1k2025-sim/bin/python', '-c', code],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
