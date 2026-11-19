"""Real MP4 encoding checks for the shared adapter and active external runner."""

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
GROOT_REPO = Path(os.environ.get("GROOT_REPO", "/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1"))
SIM_PYTHON = Path(os.environ.get("B1K_PYTHON", "/nvme3/envs/b1k2025-sim/bin/python"))


@pytest.mark.skipif(not SIM_PYTHON.exists() or not GROOT_REPO.is_dir(), reason="requires GR00T simulator Python")
@pytest.mark.parametrize("source", [
    ROOT / "evaluation/sim/wrapper/video_recording_wrapper.py",
    GROOT_REPO / "gr00t/eval/sim/wrapper/video_recording_wrapper.py",
], ids=["shared-adapter", "active-runner"])
def test_retains_failed_timeout_debug_and_final_episode_videos(tmp_path, source):
    code = r'''
import importlib.util
from pathlib import Path
import sys

import av
import gymnasium as gym
import numpy as np

spec = importlib.util.spec_from_file_location("video_under_test", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
outdir = Path(sys.argv[2])

class TinyEnv(gym.Env):
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    observation_space = gym.spaces.Dict({
        "video.front": gym.spaces.Box(0, 255, shape=(16, 16, 3), dtype=np.uint8),
    })
    def __init__(self, success, terminated, truncated):
        self.success, self.terminated, self.truncated = success, terminated, truncated
        self.closed = False
    def reset(self, **kwargs):
        self.observation = {"video.front": np.zeros((16, 16, 3), dtype=np.uint8)}
        return self.observation, {"seed": kwargs.get("seed")}
    def step(self, action):
        self.result = (self.observation, 0.0, self.terminated, self.truncated, {"success": self.success})
        return self.result
    def close(self):
        self.closed = True

def assert_playable(path):
    with av.open(str(path)) as video:
        frames = list(video.decode(video=0))
    assert len(frames) == 3, (path, len(frames))
    assert frames[0].width == 16 and frames[0].height == 16

# Native timeout and runner truncation must both retain failures. A final
# episode must also flush its encoder when no subsequent reset takes place.
for name, success, terminated, truncated, max_steps, boundary in [
    ("native_timeout", False, False, True, 999999999, "reset"),
    ("debug_truncation", False, False, True, 256, "reset"),
    ("unfinished", False, False, False, 999999999, "reset"),
    ("success", True, True, False, 999999999, "reset"),
    ("runner_limit", False, False, True, 3, "reset"),
    ("last_timeout", False, False, True, 999999999, "close"),
    ("last_unfinished", False, False, False, 999999999, "close"),
]:
    directory = outdir / name
    recorder = module.VideoRecorder.create_h264(fps=20)
    inner = TinyEnv(success, terminated, truncated)
    env = module.VideoRecordingWrapper(inner, recorder, video_dir=directory,
        steps_per_render=1, max_episode_steps=max_steps, overlay_text=False)
    env.reset(seed=7)
    for _ in range(3):
        result = env.step(np.zeros(1, dtype=np.float32))
        assert result is inner.result  # Observations and termination are untouched.
    if boundary == "reset":
        env.reset(seed=8)
    else:
        env.close()
        assert inner.closed and not recorder.is_ready()
    files = list(directory.glob("*.mp4"))
    assert len(files) == 1, (name, files)
    assert_playable(files[0])
    env.close()
    assert list(directory.glob("*.mp4")) == files
'''
    environment = {**os.environ, "PYTHONPATH": str(GROOT_REPO)}
    result = subprocess.run(
        [str(SIM_PYTHON), "-s", "-c", code, str(source), str(tmp_path)],
        env=environment, text=True, capture_output=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not SIM_PYTHON.exists() or not GROOT_REPO.is_dir(), reason="requires GR00T simulator Python")
@pytest.mark.parametrize("source", [
    ROOT / "evaluation/sim/wrapper/video_recording_wrapper.py",
    GROOT_REPO / "gr00t/eval/sim/wrapper/video_recording_wrapper.py",
], ids=["shared-adapter", "active-runner"])
def test_caption_is_small_outside_camera_pixels_and_does_not_mutate_observations(tmp_path, source):
    code = r'''
import importlib.util
from pathlib import Path
import sys
import av
import gymnasium as gym
import numpy as np
spec=importlib.util.spec_from_file_location("video_under_test",sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Env(gym.Env):
    observation_space=gym.spaces.Dict({})
    action_space=gym.spaces.Box(-1,1,(1,),np.float32)
    def reset(self,**kwargs):
        self.obs={"video.head":np.full((256,256,3),90,np.uint8),
                  "annotation.human.coarse_action":"turn on radio"}
        return self.obs,{}
    def step(self,action):
        return self.obs,0.,False,False,{"success":False}
inner=Env();recorder=module.VideoRecorder.create_h264(fps=20)
env=module.VideoRecordingWrapper(inner,recorder,video_dir=Path(sys.argv[2]),steps_per_render=1)
env.reset()
for label in ("turn on radio", "long instruction "*12):
    inner.obs["annotation.human.coarse_action"]=label
    result=env.step(np.zeros(1,np.float32))
    assert np.all(result[0]["video.head"]==90), "caption modified policy observation"
env.close()
with av.open(str(next(Path(sys.argv[2]).glob("*.mp4")))) as video:
    frames=list(video.decode(video=0))
assert len(frames)==2
for frame in frames:
    assert frame.width==256 and frame.height==280, (frame.width,frame.height)
    pixels=frame.to_ndarray(format="rgb24")
    assert np.abs(pixels[:254].astype(float)-90).mean()<3
'''
    result = subprocess.run(
        [str(SIM_PYTHON), "-s", "-c", code, str(source), str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(GROOT_REPO)}, text=True, capture_output=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
