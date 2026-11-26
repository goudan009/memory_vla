"""Video identities follow the locked public episode journal, including resumes."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/scripts"))
spec = importlib.util.spec_from_file_location("named_video_results", ROOT / "evaluation/scripts/groot_eval_results.py")
results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results)


def test_episode_video_names_follow_public_number_across_gpus_and_resume(tmp_path):
    public = tmp_path / "run"
    public.mkdir()
    (public / "results.json").write_text(json.dumps({"evaluation_mode": "official-eval"}))
    for number, (gpu, env_idx, instance, seed, attempt) in enumerate([
        (0, 1, 5, 1, "gpu0"), (1, 0, 6, 2, "gpu1"),
        (0, 0, 2, 2, "gpu0_resume2"),
    ], 1):
        outdir = public / ".runtime" / attempt / "task21"
        videos = outdir / "videos"
        videos.mkdir(parents=True)
        source = videos / "random-uuid.mp4"
        source.write_bytes(b"already-flushed-video")
        (outdir / "run_meta.json").write_text(json.dumps({
            "task_id": 21, "task_name": "collecting_childrens_toys", "episodes": 10,
            "gpu_index": gpu, "public_output_dir": str(public),
        }))
        writer = results.EpisodeWriter(outdir, env_idx=env_idx)
        writer.append(instance_id=instance, seed=seed, success=False,
                      task_progress=20, valid=True, sim_steps=8, video_source=source)
        expected = f"episode_{number:03d}_instance{instance}_seed{seed}_gpu{gpu}_env{env_idx}_s0.mp4"
        assert not source.exists()
        assert (videos / expected).read_bytes() == b"already-flushed-video"
        public_records = [json.loads(line) for line in (public / "episodes.jsonl").read_text().splitlines()]
        record = public_records[-1]
        assert record["episode_index"] == number
        assert record["video_file"] == f"videos/{attempt}/{expected}"
        local_record = json.loads((outdir / "episodes.jsonl").read_text().splitlines()[-1])
        assert local_record["video_file"] == record["video_file"]
        # The next appended episode must not change the previous association.
        assert len({record["video_file"] for record in public_records}) == number


SIM_PYTHON = Path("/nvme3/envs/b1k2025-sim/bin/python")
GROOT_REPO = Path("/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1")


@pytest.mark.skipif(not SIM_PYTHON.exists() or not GROOT_REPO.exists(), reason="requires simulator Python")
@pytest.mark.parametrize("source", [
    ROOT / "evaluation/sim/wrapper/video_recording_wrapper.py",
    GROOT_REPO / "gr00t/eval/sim/wrapper/video_recording_wrapper.py",
], ids=["shared-adapter", "active-runner"])
@pytest.mark.parametrize('task_id', [21, 40])
def test_real_recorder_is_named_at_termination_and_keeps_identity_after_reset(tmp_path, source, task_id):
    code = r'''
import importlib.util
import json
import os
from pathlib import Path
import sys
import av
import gymnasium as gym
import numpy as np
from types import ModuleType, SimpleNamespace

from groot_eval_results import record_episode_env

spec = importlib.util.spec_from_file_location("recorder_under_test", sys.argv[1])
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)
public = Path(sys.argv[2])
task_id = int(sys.argv[3])
# Use the real MP4 recorder, with a minimal simulator clock for the trace observer.
# No simulator scene, CUDA allocation or official metric substitution is needed.
if task_id == 40:
    og = ModuleType('omnigibson')
    og.sim = SimpleNamespace(current_time=10.)
    states = ModuleType('omnigibson.object_states')
    for name in ('Open', 'ToggledOn', 'Inside', 'Cooked'):
        setattr(states, name, object())
    og.object_states = states
    sys.modules['omnigibson'] = og
    sys.modules['omnigibson.object_states'] = states
    os.environ['GROOT_TASK40_TRACE'] = '1'
outdir = public / ".runtime/gpu0/task21"
outdir.mkdir(parents=True)
(public / "results.json").write_text(json.dumps({"evaluation_mode": "official-eval"}))
(outdir / "run_meta.json").write_text(json.dumps({
    "task_id": task_id, "task_name": "fixture_task", "episodes": 2,
    "gpu_index": 0, "public_output_dir": str(public), "seed": 0,
}))

class TinyBehavior(gym.Env):
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    observation_space = gym.spaces.Dict({
        "video.front": gym.spaces.Box(0, 255, shape=(16, 16, 3), dtype=np.uint8),
    })
    def __init__(self):
        self._instance_indices_this_env = [0, 1]
        self._all_test_instances = [4, 7]
        self._instance_idx_pointer = 0
        if task_id == 40:
            self._task = self.task = SimpleNamespace(_termination_config={}, _info={'done':{'success':False}})
            self._current_step = 0
            self.scene = SimpleNamespace(objects=[], _initial_file={
                'state': {'registry': {'object_registry': {}}}})
            self.robots = []
    def reset(self, **kwargs):
        self._instance_idx_pointer = (self._instance_idx_pointer + 1) % 2
        self.steps = 0
        if task_id == 40:
            self._current_step = 0
        self.observation = {"video.front": np.zeros((16, 16, 3), dtype=np.uint8)}
        if task_id == 40:
            self.observation.update({f'video.observation.images.rgb.{camera}_256_256':
                np.zeros((16, 16, 3), dtype=np.uint8)
                for camera in ('head', 'left_wrist', 'right_wrist')})
        return self.observation, {}
    def step(self, action):
        self.steps += 1
        if task_id == 40:
            self._current_step += 1
            og.sim.current_time += 1/30
        self.result = (self.observation, 0.0, False, self.steps == 3, {
            "success": False, "valid": True, "rewards": [0.0], "task_progress": 60.0,
        })
        return self.result

inner = TinyBehavior()
recorder = video.VideoRecorder.create_h264(fps=20)
wrapper = video.VideoRecordingWrapper(inner, recorder, video_dir=outdir / "videos",
    steps_per_render=1, overlay_text=False)
env = record_episode_env(lambda: wrapper, outdir, env_idx=0)
saved = []
for number, instance in enumerate([4, 7], 1):
    env.reset()
    for step in range(3):
        result = env.step(np.zeros(1, dtype=np.float32))
        assert result[0] is inner.observation
        assert result[2:4] == (False, step == 2)
    records = [json.loads(line) for line in (public / "episodes.jsonl").read_text().splitlines()]
    record = records[-1]
    expected = f"episode_{number:03d}_instance{instance}_seed{number-1}_gpu0_env0_s0.mp4"
    assert record["video_file"] == f"videos/gpu0/{expected}"
    assert record["instance_id"] == instance
    assert record["sim_steps"] == 3 and record["valid"] and record["truncated"]
    path = outdir / "videos" / expected
    # Already playable at termination, without a subsequent reset or close.
    assert not recorder.is_ready() and wrapper.file_path is None
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 3
    if task_id == 40:
        rows = [json.loads(line) for line in (public/'stages/video_gpu0_env0.jsonl').read_text().splitlines()]
        rows = [row for row in rows if row['episode_number'] == number]
        assert [row['frame_index'] for row in rows] == [0, 1, 2]
        assert [row['action_steps'] for row in rows] == [1, 2, 3]
        assert all(row['instance_id'] == instance and row['seed'] == number-1 for row in rows)
        assert rows[-1]['sim_time_s'] == og.sim.current_time
        assert record['native_termination'] == {'success':False}
    saved.append(path)
    assert all(path.exists() for path in saved)
env.reset()
env.step(np.zeros(1, dtype=np.float32))
unfinished = wrapper.file_path
env.close()
assert unfinished.exists() and all(path.exists() for path in saved)
with av.open(str(unfinished)) as container:
    assert len(list(container.decode(video=0))) == 1
'''
    result = subprocess.run(
        [str(SIM_PYTHON), "-s", "-c", code, str(source), str(tmp_path), str(task_id)],
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'evaluation/scripts'}:{GROOT_REPO}"},
        text=True, capture_output=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
