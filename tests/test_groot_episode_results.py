import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/groot_eval_results.py"
spec = importlib.util.spec_from_file_location("groot_eval_results", MODULE_PATH)
results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results)


def make_output(tmp_path, episodes=2):
    (tmp_path / "run_meta.json").write_text(json.dumps({
        "task_id": 21, "task_name": "collecting_childrens_toys", "episodes": episodes,
    }))
    return tmp_path


def test_each_episode_updates_results_and_keeps_invalid_attempts(tmp_path):
    writer = results.EpisodeWriter(make_output(tmp_path))
    writer.append(instance_id=4, success=True, task_progress=100.0, q_score=1.0, valid=True, sim_steps=8)
    first = json.loads((tmp_path / "results.json").read_text())
    assert first["completed_episodes"] == 1
    assert first["success_rate"] == 1.0
    assert first["task_progress"] == [100.0]
    assert first["status"] == "running"
    assert first["mean_q_score"] == 1.0
    assert first["q_score_episodes"] == 1
    first_record = json.loads((tmp_path / "episodes.jsonl").read_text().splitlines()[0])
    assert first_record["seed"] == 0
    assert first_record["q_score"] == 1.0
    assert first_record["task"] == "collecting_childrens_toys"
    assert first_record["env_idx"] == 0
    assert first_record["episode_length"] == 8
    assert first_record["duration_s"] >= 0
    profile = json.loads((tmp_path / "profiling.json").read_text())
    assert "p95_s" in profile["episode_total"]
    assert profile["valid_episodes_per_hour"] >= 0
    writer.append(instance_id=7, success=False, task_progress=0.0, q_score=1.0, valid=False, sim_steps=2)
    writer.append(instance_id=8, success=False, task_progress=25.0, q_score=0.0, valid=True, sim_steps=16)
    summary = results.finish_run(tmp_path, 0)
    assert summary["status"] == "complete"
    assert summary["completed_episodes"] == 2
    assert summary["recorded_episodes"] == 3
    assert summary["invalid_episodes"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["mean_task_progress"] == 62.5
    assert summary["mean_q_score"] == 0.5
    assert summary["q_score_episodes"] == 2
    assert summary["missing_q_score_episodes"] == 0
    assert "q_score" in (tmp_path / "manifest.csv").read_text().splitlines()[0]
    assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 3
    assert len((tmp_path / "manifest.csv").read_text().splitlines()) == 4


@pytest.mark.parametrize("return_code,interrupted,status", [(1, False, "failed"), (-15, True, "interrupted")])
def test_completed_episode_survives_runner_failure(tmp_path, return_code, interrupted, status):
    writer = results.EpisodeWriter(make_output(tmp_path))
    writer.append(instance_id=0, success=False, task_progress=50.0, q_score=0.25, valid=True, sim_steps=10)
    summary = results.finish_run(tmp_path, return_code, interrupted)
    assert summary["status"] == status
    assert summary["completed_episodes"] == 1
    assert summary["task_progress"] == [50.0]
    assert summary["success_rate"] == 0.0
    assert summary["mean_q_score"] == 0.25
    assert summary["runner_exit_code"] == return_code


def test_debug_records_cannot_be_reported_as_official_success_rate(tmp_path):
    make_output(tmp_path, episodes=1)
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    meta.update(preset="fast-debug", runner_max_episode_steps=1200)
    (tmp_path / "run_meta.json").write_text(json.dumps(meta))
    writer = results.EpisodeWriter(tmp_path)
    writer.append(instance_id=0, success=True, task_progress=100.0, q_score=1.0, valid=True, sim_steps=1200)
    summary = results.finish_run(tmp_path, 0)
    assert summary["status"] == "complete"
    assert summary["evaluation_mode"] == "fast-debug"
    assert summary["official_episodes"] == 0
    assert summary["success_rate"] is None
    assert summary["mean_q_score"] is None
    assert summary["q_score_episodes"] == 1
    assert json.loads((tmp_path / "episodes.jsonl").read_text())["evaluation_mode"] == "fast-debug"
    assert json.loads((tmp_path / "profiling.json").read_text())["valid_episodes_per_hour"] is None


def test_recorder_saves_terminal_info_before_vector_autoreset(tmp_path, monkeypatch):
    gym = pytest.importorskip("gymnasium")
    np = pytest.importorskip("numpy")
    make_output(tmp_path)
    # Register the original constructor for restoration after installing the hook.
    monkeypatch.setattr(gym.vector, "SyncVectorEnv", gym.vector.SyncVectorEnv)

    class FakeBehavior(gym.Env):
        observation_space = gym.spaces.Box(0, 100, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)

        def __init__(self):
            self._instance_indices_this_env = [0, 1]
            self._all_test_instances = [4, 7]
            self._instance_idx_pointer = 0
            self.resets = 0

        def reset(self, **kwargs):
            self.resets += 1
            if self.resets == 3:
                raise RuntimeError("reset failed after second episode")
            self._instance_idx_pointer = (self._instance_idx_pointer + 1) % 2
            self.steps = 0
            return np.array([0], dtype=np.float32), {"task_progress": 0.0}

        def step(self, action):
            self.steps += 1
            return np.array([self.steps], dtype=np.float32), 0, self.steps == 2, False, {
                **({"q_score": [0.0, 0.25]} if self.steps == 2 else {}),
                "success": [False, self.steps == 2],
                "task_progress": [10.0, 60.0],
                "valid": [True],
                "rewards": np.zeros(2),
            }

    results.install_episode_recording(tmp_path)
    env = gym.vector.SyncVectorEnv([FakeBehavior], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    try:
        env.reset()
        env.step(np.zeros((1, 1)))
        assert not (tmp_path / "episodes.jsonl").exists()
        env.step(np.zeros((1, 1)))
        first = json.loads((tmp_path / "episodes.jsonl").read_text())
        assert first["instance_id"] == 4
        assert first["task_progress"] == 60.0
        assert first["q_score"] == 0.25
        assert first["success"] is True
        assert first["sim_steps"] == 4
        assert first["seed"] == 0
        env.step(np.zeros((1, 1)))
        with pytest.raises(RuntimeError, match="reset failed"):
            env.step(np.zeros((1, 1)))
        records = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
        assert [record["instance_id"] for record in records] == [4, 7]
        assert [record["seed"] for record in records] == [0, 1]
        assert [record["q_score"] for record in records] == [0.25, 0.25]
        assert all(record["terminated"] and not record["truncated"] for record in records)
        assert json.loads((tmp_path / "results.json").read_text())["completed_episodes"] == 2
    finally:
        env.close()


def test_legacy_missing_q_is_not_zero_when_resuming(tmp_path):
    writer = results.EpisodeWriter(make_output(tmp_path))
    writer.append(instance_id=0, success=False, task_progress=50.0, valid=True, sim_steps=8)
    record = json.loads((tmp_path / "episodes.jsonl").read_text())
    del record["q_score"]  # Existing runs predate the Q field.
    (tmp_path / "episodes.jsonl").write_text(json.dumps(record) + "\n")
    writer = results.EpisodeWriter(tmp_path)
    assert json.loads((tmp_path / "results.json").read_text())["mean_q_score"] is None
    writer.append(instance_id=1, success=False, task_progress=25.0, q_score=0.2, valid=True, sim_steps=8)
    summary = results.finish_run(tmp_path, 0)
    assert summary["mean_q_score"] == 0.2
    assert summary["q_score_episodes"] == 1
    assert summary["missing_q_score_episodes"] == 1
    assert summary["completed_episodes"] == 2


def test_two_envs_keep_terminal_q_separate(tmp_path):
    gym = pytest.importorskip("gymnasium")
    np = pytest.importorskip("numpy")
    writer = results.EpisodeWriter(make_output(tmp_path))

    class FakeBehavior(gym.Env):
        observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        action_space = observation_space

        def __init__(self, q):
            self.q = q

        def reset(self, **kwargs):
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(1, dtype=np.float32), 0, True, False, {
                "success": [False], "valid": [True], "q_score": [self.q],
                "task_progress": [60.0], "rewards": [0],
            }

    env = gym.vector.SyncVectorEnv(
        writer.wrap_env_fns([lambda: FakeBehavior(0.1), lambda: FakeBehavior(0.8)]),
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        env.reset()
        env.step(np.zeros((2, 1), dtype=np.float32))
        records = writer.load_records()
        assert [(record["env_idx"], record["q_score"]) for record in records] == [(0, 0.1), (1, 0.8)]
        summary = json.loads((tmp_path / "results.json").read_text())
        assert summary["mean_q_score"] == pytest.approx(0.45)
        assert summary["q_score_episodes"] == 2
        assert summary["mean_task_progress"] == 60.0
    finally:
        env.close()
