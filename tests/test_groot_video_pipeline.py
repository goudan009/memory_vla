"""Join the real rollout, H.264 encoder, queue and public artifact publisher.

Only the simulator, GPU preflight and remote model are replaced with fixtures.
This checks artifact plumbing without starting another Isaac Sim on busy GPUs.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import signal
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
GROOT = Path("/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1")
PYTHON = Path("/nvme3/envs/b1k2025-sim/bin/python")


@pytest.mark.skipif(not PYTHON.exists() or not GROOT.is_dir(), reason="requires simulator Python and fixed GR00T checkout")
@pytest.mark.parametrize("n_envs", [1, 2])
@pytest.mark.parametrize("ending", ["complete", "interrupt", "exception", "physx", "invalid"])
def test_real_rollout_archives_every_failed_episode_as_playable_mp4(tmp_path, n_envs, ending):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("run_groot_priority_eval.sh", "groot_eval_results.py", "groot_eval_output.py", "groot_policy_lifecycle.py"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    for name in ("groot_tasks.py", "groot_live_preview.py"):
        shutil.copy2(ROOT / "evaluation/scripts" / name, scripts / name)
    tools = tmp_path / "evaluation/scripts"
    tools.mkdir(parents=True)
    shutil.copy2(ROOT / "evaluation/scripts/run_groot_eval.sh", tools / "run_groot_eval.sh")
    repo = tmp_path / "groot"
    runner = repo / "gr00t/eval/rollout_policy.py"
    runner.parent.mkdir(parents=True)
    fixture = scripts / "video_pipeline_fixture.py"
    fixture.write_text('''
import os
from pathlib import Path
import time
import gymnasium as gym
import numpy as np
from gr00t.eval import rollout_policy as r

class TinyBehavior(gym.Env):
    observation_space = gym.spaces.Dict({
        **{f"video.{camera}": gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)
           for camera in ("head", "left", "right")},
        "state.robot": gym.spaces.Box(-1, 1, (1,), np.float32),
        "annotation.human.coarse_action": gym.spaces.Text(512),
    })
    action_space = gym.spaces.Dict({"action.robot": gym.spaces.Box(-1, 1, (1,), np.float32)})
    def __init__(self, env_idx, total_n_envs):
        print(f"VIDEO_TEST_ENV_PID {os.getpid()}", flush=True)
        self.env_idx = env_idx
        self._physx_crashed = False
        self._all_test_instances = list(range(2 * total_n_envs))
        self._instance_indices_this_env = np.array_split(np.arange(2 * total_n_envs), total_n_envs)[env_idx]
        self._instance_idx_pointer = 0
    def reset(self, **kwargs):
        if self._physx_crashed:
            return self.obs, {}  # Reproduce BEHAVIOR's permanently broken reset.
        self._instance_idx_pointer += 1
        assert self._instance_idx_pointer <= 2, "invalid Episode was retried"
        self.steps = 0
        self.obs = {f"video.{camera}": np.full((64, 64, 3), 80, np.uint8)
                    for camera in ("head", "left", "right")}
        self.obs.update({"state.robot": np.zeros(1, np.float32),
                         "annotation.human.coarse_action": "test"})
        return self.obs, {}
    def step(self, action):
        self.steps += 1
        assert self.steps <= 193  # Native timeout must break the final chunk.
        if (self.env_idx == 0 and self._instance_idx_pointer == 2 and self.steps >= 121
                and os.environ["VIDEO_TEST_ENDING"] in ("physx", "invalid")):
            self._physx_crashed = os.environ["VIDEO_TEST_ENDING"] == "physx"
            return self.obs, 0., True, True, {
                "success": False, "valid": False, "q_score": 0., "task_progress": 0.,
            }
        return self.obs, 0., False, self.steps == 193, {
            "success": False, "valid": True, "q_score": .25, "task_progress": 33.,
        }

def tiny_env(env_name, env_idx, total_n_envs):
    return TinyBehavior(env_idx, total_n_envs)

r.get_gym_env = tiny_env
real_create = r.create_eval_env

def create_eval_env(**kwargs):
    return real_create(**kwargs)

class TinyPolicy:
    def reset(self, options=None):
        pass
    def get_action(self, obs, options=None):
        self.calls = getattr(self, "calls", 0) + 1
        if self.calls == 42:
            if os.environ["VIDEO_TEST_ENDING"] == "exception":
                raise RuntimeError("injected policy failure")
            if os.environ["VIDEO_TEST_ENDING"] == "interrupt":
                Path(os.environ["VIDEO_TEST_READY"]).touch()
                time.sleep(1)
        assert all(obs[f"video.{camera}"].shape[1:] == (1, 64, 64, 3)
                   for camera in ("head", "left", "right"))
        return {"action.robot": np.zeros((len(obs["state.robot"]), 8, 1), np.float32)}, {}
''')
    runner.write_text('''
import argparse
import os
import sys
sys.path.insert(0, os.environ["AUDIT_REAL_GROOT"])
import video_pipeline_fixture as fixture
from gr00t.eval import rollout_policy as r
r.create_eval_env = fixture.create_eval_env
r.create_gr00t_sim_policy = lambda *args, **kwargs: fixture.TinyPolicy()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name")
    parser.add_argument("--n_envs", type=int)
    parser.add_argument("--n_episodes", type=int)
    parser.add_argument("--max_episode_steps", type=int)
    args, _ = parser.parse_known_args()
    assert args.max_episode_steps == 999999999
    print("results:", r.run_gr00t_sim_policy(
        env_name=args.env_name, n_episodes=args.n_episodes, n_envs=args.n_envs,
        max_episode_steps=args.max_episode_steps, n_action_steps=8,
        record_video=True, video_steps_per_render=96), flush=True)
''')
    wrapper = tmp_path / "fake_b1k_python"
    wrapper.write_text(f'''#!{PYTHON}
import sys
import types
index = sys.argv.index("-c") + 1
code = sys.argv[index]
sys.argv = [sys.argv[0]] + sys.argv[index + 1:]
if "MODALITY_KEYS" in code:
    print("PING True MODALITY_KEYS []")
else:
    sys.modules["isaacsim"] = types.ModuleType("isaacsim")
    exec(code)
''')
    wrapper.chmod(0o755)
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/sh\nprintf '0\\n'\n")
    smi.chmod(0o755)
    environment = {**os.environ,
        "PROJECT_ROOT": str(tmp_path), "GROOT_REPO": str(repo),
        "AUDIT_REAL_GROOT": str(GROOT), "B1K_PYTHON_LAUNCHER": str(wrapper),
        "B1K_PYTHON": str(PYTHON), "OMNIGIBSON_ROOT": str(tmp_path),
        "BDDL_ROOT": str(tmp_path), "BEHAVIOR_DATA": str(tmp_path),
        "PYTHONNOUSERSITE": "1", "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "VIDEO_TEST_ENDING": ending, "VIDEO_TEST_READY": str(tmp_path / "ready"),
    }
    command = ["bash", str(tools / "run_groot_eval.sh"), "--mode", f"single{n_envs}",
         "--gpu-index", "0", "--task", "26", "--episodes-per-gpu", str(n_envs * 2),
         "--instances-gpu0", *map(str, range(n_envs * 2))]
    if ending == "interrupt":
        with (tmp_path / "launcher.log").open("w") as stream:
            process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 60
                while not (tmp_path / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.02)
                assert (tmp_path / "ready").exists(), (tmp_path / "launcher.log").read_text()
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=30)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        result = subprocess.CompletedProcess(command, process.returncode, (tmp_path / "launcher.log").read_text(), "")
    else:
        result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=90)
    output = tmp_path / "evaluation/V_1/baseline/groot_task26_1"
    log = (output / "gpu0.log").read_text() if (output / "gpu0.log").exists() else ""
    assert (result.returncode == 0) == (ending == "complete"), result.stdout + result.stderr + log
    summary = json.loads((output / "results.json").read_text())
    records = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    assert summary["status"] == {"complete": "complete", "interrupt": "interrupted"}.get(ending, "failed")
    assert summary["ok"] == (ending == "complete")
    invalid = [record for record in records if not record["valid"]]
    valid = [record for record in records if record["valid"]]
    assert summary["completed_episodes"] == len(valid) == (2 * n_envs if ending == "complete" else n_envs)
    assert summary["invalid_episodes"] == len(invalid) == (1 if ending in ("physx", "invalid") else 0)
    assert summary["success_rate"] == 0 and summary["mean_q_score"] == .25
    assert all(not record["success"] and record["sim_steps"] == 193 for record in valid)
    expected_instances = list(range(2 * n_envs)) if ending == "complete" else list(range(0, 2 * n_envs, 2))
    assert sorted(record["instance_id"] for record in valid) == expected_instances
    if invalid:
        reason = summary["failure_reason"]
        assert reason["code"] == ("unrecoverable_physx_error" if ending == "physx" else "invalid_episode")
        assert reason["env_idx"] == 0 and reason["instance_id"] == 1 and reason["seed"] == 1
        assert reason == invalid[0]["failure_reason"] == summary["per_gpu"]["0"]["summary"]["failure_reason"]
        assert invalid[0]["sim_steps"] == 121
        assert "EVAL_INVALID_EPISODE" in log
    assert summary["saved_video_count"] == len(summary["videos"]) == 2 * n_envs
    for relative in summary["videos"]:
        assert (output / relative).is_file()
    assert {item.name for item in output.iterdir()} == {"results.json", "episodes.jsonl", "gpu0.log", "videos"}
    # Decode the actual published MP4s, including each slot's final episode.
    decoded = subprocess.run([str(PYTHON), "-s", "-c", '''
import av
from pathlib import Path
import sys
for path in Path(sys.argv[1]).rglob("*.mp4"):
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) in (1, 2), (str(path), len(frames))
        assert (frames[0].width, frames[0].height) == (192, 88)  # 24px caption footer.
print("all published videos decode")
''', str(output / "videos")], env=environment, text=True, capture_output=True, timeout=15)
    assert decoded.returncode == 0, decoded.stdout + decoded.stderr
    for line in log.splitlines():
        if line.startswith("VIDEO_TEST_ENV_PID "):
            assert not Path(f"/proc/{int(line.split()[-1])}").exists(), line
