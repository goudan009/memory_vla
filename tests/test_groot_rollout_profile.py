import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROOT_REPO = Path(os.environ.get("GROOT_REPO", "/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1"))
SIM_PYTHON = Path(os.environ.get("B1K_PYTHON", "/nvme3/envs/b1k2025-sim/bin/python"))


@pytest.mark.skipif(not SIM_PYTHON.exists() or not GROOT_REPO.is_dir(), reason="requires simulator Python and GR00T")
@pytest.mark.parametrize("episode_steps,profile_cycles", [(1, None), (105, 100)])
def test_rollout_profile_and_episode_recording_with_tiny_gym_env(tmp_path, episode_steps, profile_cycles):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT / "tests"), str(GROOT_REPO)))
    code = r'''
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

from groot_eval_results import install_episode_recording
from gr00t.eval import rollout_policy

outdir = Path(sys.argv[1])
episode_steps = int(sys.argv[2])
profile_cycles = None if sys.argv[3] == "None" else int(sys.argv[3])
(outdir / "run_meta.json").write_text(json.dumps({
    "task_id": 21, "task_name": "tiny", "episodes": 1, "seed": 17,
}))
install_episode_recording(outdir)
reset_random = []

class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
    def __init__(self):
        self._instance_indices_this_env = [0]
        self._all_test_instances = [3]
        self._instance_idx_pointer = 0
    def reset(self, **kwargs):
        reset_random.append(np.random.random())
        self._instance_idx_pointer = 1
        self.steps = 0
        return np.zeros(1, dtype=np.float32), {}
    def step(self, action):
        self.steps += 1
        return np.zeros(1, dtype=np.float32), 0, self.steps >= episode_steps, False, {
            "success": [True], "valid": [True], "task_progress": [75.0], "rewards": [0],
        }

rollout_policy.create_eval_env = lambda *args, **kwargs: TinyEnv()
class Policy:
    def reset(self, options=None): pass
    def get_action(self, observations, options=None): return np.zeros((1, 1), dtype=np.float32), {}

rollout_policy.run_rollout_gymnasium_policy(
    "sim_behavior_r1_pro/tiny", Policy(), rollout_policy.WrapperConfigs(), n_episodes=1, n_envs=1,
    profile_cycles=profile_cycles,
)
assert reset_random[0] == np.random.RandomState(17).random_sample()
'''
    result = subprocess.run(
        [str(SIM_PYTHON), "-c", code, str(tmp_path), str(episode_steps), str(profile_cycles)],
        cwd=GROOT_REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    episode = __import__("json").loads((tmp_path / "episodes.jsonl").read_text().splitlines()[0])
    profile = __import__("json").loads((tmp_path / "profiling.json").read_text())
    assert episode["task"] == "tiny"
    assert episode["env_idx"] == 0
    assert episode["success"] is True
    assert episode["task_progress"] == 75.0
    assert episode["episode_length"] == episode_steps
    assert episode["duration_s"] >= 0
    assert profile["valid_episodes"] == 1
    assert profile["valid_episodes_per_hour"] > 0
    assert profile["policy_get_action"]["count"] == min(episode_steps, profile_cycles or episode_steps)
    if profile_cycles is not None:
        diagnostic = __import__("json").loads((tmp_path / "diagnostic_profile.json").read_text())
        assert diagnostic["cycles"] == profile_cycles
        assert diagnostic["policy_get_action"]["count"] == profile_cycles
    for key in ("environment_initialization", "reset", "policy_get_action", "env_step", "episode_total"):
        assert profile[key]["count"] > 0
        assert profile[key]["mean_s"] >= 0
        assert profile[key]["p95_s"] >= 0


@pytest.mark.skipif(not SIM_PYTHON.exists() or not GROOT_REPO.is_dir(), reason="requires simulator Python and GR00T")
@pytest.mark.parametrize("public_output", [False, True])
def test_async_workers_persist_unique_instance_and_env_indices(tmp_path, public_output):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT / "tests"), str(GROOT_REPO)))
    code = r'''
import json
import sys
from pathlib import Path

from groot_eval_results import install_episode_recording
from groot_eval_output import update_results
from gr00t.eval import rollout_policy
from groot_rollout_fixture import Policy, create_tiny_env

outdir = Path(sys.argv[1])
public = outdir / "public"
public.mkdir()
update_results(public, updates={"evaluation_mode": "official-eval", "status": "running"})
(outdir / "run_meta.json").write_text(json.dumps({
    "task_id": 21, "task_name": "tiny", "episodes": 2,
    **({"gpu_index": 1, "public_output_dir": str(public)} if sys.argv[2] == "True" else {}),
}))
install_episode_recording(outdir)
rollout_policy.create_eval_env = create_tiny_env
rollout_policy.run_rollout_gymnasium_policy(
    "sim_behavior_r1_pro/tiny", Policy(), rollout_policy.WrapperConfigs(), n_episodes=2, n_envs=2,
)
'''
    result = subprocess.run(
        [str(SIM_PYTHON), "-c", code, str(tmp_path), str(public_output)],
        cwd=GROOT_REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = [__import__("json").loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert sorted(record["env_idx"] for record in records) == [0, 1]
    assert sorted(record["instance_id"] for record in records) == [0, 5]
    summary = __import__("json").loads((tmp_path / "results.json").read_text())
    assert summary["recorded_episodes"] == 2
    assert summary["completed_episodes"] == 2
    assert summary["success_rate"] == 1.0
    if public_output:
        public_records = [__import__("json").loads(line) for line in
                          (tmp_path / "public/episodes.jsonl").read_text().splitlines()]
        assert sorted(record["env_idx"] for record in public_records) == [0, 1]
        assert sorted(record["instance_id"] for record in public_records) == [0, 5]
        assert [record["episode_index"] for record in public_records] == [1, 2]
        assert all(record["gpu_index"] == 1 and record["task_id"] == 21 for record in public_records)
        assert all("env_chunk_times_s" not in record for record in public_records)
        public_summary = __import__("json").loads((tmp_path / "public/results.json").read_text())
        assert public_summary["success_rate"] == 1.0 and public_summary["completed_episodes"] == 2


@pytest.mark.skipif(not SIM_PYTHON.exists() or not GROOT_REPO.is_dir(), reason="requires simulator Python and GR00T")
@pytest.mark.parametrize('seed_mode', ['sequence', 'fixed', 'random'])
@pytest.mark.parametrize('total_episodes', [10, 5])
def test_async_uneven_workers_stop_at_assigned_quota(tmp_path, seed_mode, total_episodes):
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(PROJECT_ROOT / 'scripts'), str(PROJECT_ROOT / 'tests'), str(GROOT_REPO)))
    env['GROOT_TEST_INSTANCE_COUNT'] = str(total_episodes)
    code = r'''
import json
import sys
from pathlib import Path
from groot_eval_results import install_episode_recording
from gr00t.eval import rollout_policy
from groot_rollout_fixture import Policy, create_uneven_env
outdir = Path(sys.argv[1])
total = int(sys.argv[3])
(outdir / 'run_meta.json').write_text(json.dumps({
    'task_id': 21, 'task_name': 'tiny', 'episodes': total, 'n_envs': 2,
    'seed': 42, 'seed_mode': sys.argv[2], 'limit_env_episodes': True,
}))
install_episode_recording(outdir)
rollout_policy.create_eval_env = create_uneven_env
rollout_policy.run_rollout_gymnasium_policy(
    'sim_behavior_r1_pro/tiny', Policy(), rollout_policy.WrapperConfigs(), n_episodes=total, n_envs=2,
)
'''
    result = subprocess.run([str(SIM_PYTHON), '-c', code, str(tmp_path), seed_mode, str(total_episodes)],
                            env=env, cwd=GROOT_REPO, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    import json
    import numpy as np
    records = [json.loads(line) for line in (tmp_path / 'episodes.jsonl').read_text().splitlines()]
    assert len(records) == total_episodes
    first_quota = total_episodes // 2
    for idx in (0, 1):
        group = [row for row in records if row['env_idx'] == idx]
        count = first_quota if idx == 0 else total_episodes - first_quota
        assert len(group) == count
        assert [row['instance_id'] for row in group] == list(range(idx * first_quota, idx * first_quota + count))
        expected = ([42] * count if seed_mode == 'fixed' else
                    [int(np.random.SeedSequence([42, idx, ep]).generate_state(1)[0]) for ep in range(count)]
                    if seed_mode == 'random' else list(range(42 + idx, 42 + idx + count)))
        assert [row['seed'] for row in group] == expected
        assert [row['sim_steps'] for row in group] == ([1] * count if idx == 0 else [3] * count)
