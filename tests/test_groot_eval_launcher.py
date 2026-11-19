import os
import json
import subprocess
import shutil
import time
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "run_groot_priority_eval.sh"
GROOT_REPO = Path(os.environ.get(
    "GROOT_REPO", "/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1"
))
SIM_PYTHON = Path(os.environ.get(
    "B1K_PYTHON", "/nvme3/envs/b1k2025-sim/bin/python"
))
N1D6_PYTHON = Path(os.environ.get("N1D6_PYTHON", str(GROOT_REPO / ".venv/bin/python")))


def _run_dry_run(tmp_path, *args):
    env = os.environ.copy()
    env.pop("OMNIGIBSON_DATA_PATH", None)
    env.pop("BEHAVIOR_DATA", None)
    env.update(
        {
            "PROJECT_ROOT": str(tmp_path / "project"),
            "GROOT_REPO": "/opt/Isaac-GR00T-n1d6",
            "B1K_PYTHON_LAUNCHER": "/opt/memory_vla/scripts/b1k_python.sh",
            "B1K_PYTHON": "/opt/b1k2025-sim/bin/python",
            "OMNIGIBSON_ROOT": "/opt/BEHAVIOR-1K/OmniGibson",
            "BDDL_ROOT": "/opt/BEHAVIOR-1K/bddl",
            "RUN_TAG": "test-run",
        }
    )
    return subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", *args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_shows_the_known_n1d6_bootstrap_without_creating_outputs(tmp_path):
    result = _run_dry_run(tmp_path, "--tasks", "21")

    assert result.returncode == 0, result.stderr
    assert "DRY RUN: no evaluation process started" in result.stdout
    assert "OMNIGIBSON_GPU_ID=1" in result.stdout
    assert "CUDA_VISIBLE_DEVICES unset" in result.stdout
    assert "without filtering the simulator site-packages" in result.stdout
    assert "--policy_client_port 5556" in result.stdout
    assert "--env_name sim_behavior_r1_pro/collecting_childrens_toys" in result.stdout
    assert "--max_episode_steps 999999999" in result.stdout
    assert "--video-steps-per-render 24 --record-video" in result.stdout
    assert "PRESET: official-eval" in result.stdout
    assert f"OMNIGIBSON_DATA_PATH={tmp_path / 'project/assets/simulation'} " in result.stdout
    assert not (tmp_path / "project" / "evaluation").exists()


def test_dry_run_uses_the_behavior_simulator_runtime_not_n1d6_venv(tmp_path):
    result = _run_dry_run(tmp_path, "--tasks", "21")

    assert result.returncode == 0, result.stderr
    assert "B1K_PYTHON_LAUNCHER=/opt/memory_vla/scripts/b1k_python.sh" in result.stdout
    assert "B1K_PYTHON=/opt/b1k2025-sim/bin/python" in result.stdout
    assert "OMNIGIBSON_GPU_ID=1" in result.stdout
    assert "N1D6_PYTHON=" not in result.stdout
    assert "CUDA_VISIBLE_DEVICES=1" not in result.stdout
    assert "without filtering the simulator site-packages" in result.stdout
    assert not (tmp_path / "project" / "evaluation").exists()


def test_eval_asset_root_ignores_other_projects_omnigibson_environment(tmp_path):
    env = os.environ.copy()
    env.pop("BEHAVIOR_DATA", None)
    env.update({"PROJECT_ROOT": str(tmp_path / "project"),
                "OMNIGIBSON_DATA_PATH": str(tmp_path / "other-project-assets")})
    command = ["bash", str(LAUNCHER), "--dry-run", "--tasks", "21"]
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert f"OMNIGIBSON_DATA_PATH={tmp_path / 'project/assets/simulation'} " in result.stdout
    env["BEHAVIOR_DATA"] = str(tmp_path / "explicit-assets")
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert f"OMNIGIBSON_DATA_PATH={tmp_path / 'explicit-assets'} " in result.stdout


def test_dry_run_rejects_unknown_task_ids(tmp_path):
    result = _run_dry_run(tmp_path, "--tasks", "99")

    assert result.returncode != 0
    assert "unsupported task id: 99" in result.stderr
    assert not (tmp_path / "project" / "evaluation").exists()


def test_gpu_workers_have_separate_outputs_and_appdata(tmp_path):
    worker0 = _run_dry_run(tmp_path, "--gpu-index", "0", "--tasks", "21", "--run-tag", "pair", "--instance-ids", "0", "2", "4", "6", "8", "--episodes", "5")
    worker1 = _run_dry_run(tmp_path, "--gpu-index", "1", "--tasks", "21", "--policy-port", "5557", "--run-tag", "pair", "--instance-ids", "1", "3", "5", "7", "9", "--episodes", "5")
    assert worker0.returncode == worker1.returncode == 0
    assert "OMNIGIBSON_GPU_ID=0" in worker0.stdout
    assert "priority_pair_gpu0.queue.log" in worker0.stdout
    assert "groot_runtime_gpu0/appdata" in worker0.stdout
    assert "OMNIGIBSON_GPU_ID=1" in worker1.stdout
    assert "priority_pair_gpu1.queue.log" in worker1.stdout
    assert "groot_runtime_gpu1/appdata" in worker1.stdout
    assert "--policy_client_port 5557" in worker1.stdout
    assert "--video-steps-per-render 24" in worker1.stdout
    assert "INSTANCE_IDS: [0, 2, 4, 6, 8]" in worker0.stdout
    assert "INSTANCE_IDS: [1, 3, 5, 7, 9]" in worker1.stdout
    assert "SEED: 0" in worker0.stdout
    assert not (tmp_path / "project" / "evaluation").exists()


def test_dry_run_can_enable_sparse_video_recording(tmp_path):
    result = _run_dry_run(tmp_path, "--tasks", "21", "--record-video", "--video-steps-per-render", "32")
    assert result.returncode == 0, result.stderr
    assert "--video-steps-per-render 32 --record-video" in result.stdout


def test_dry_run_can_disable_video_recording(tmp_path):
    result = _run_dry_run(tmp_path, '--tasks', '21', '--no-record-video')
    assert result.returncode == 0, result.stderr
    assert 'record_video=False' in result.stdout
    assert '--record-video' not in result.stdout


def test_explicit_output_root_contains_queue_log_and_task_results(tmp_path):
    output = tmp_path / 'run' / 'gpu1'
    result = _run_dry_run(tmp_path, '--tasks', '21', '--output-root', str(output))
    assert result.returncode == 0, result.stderr
    assert f'QUEUE_LOG={output}/groot_n1d6_priority_test-run_gpu1.queue.log' in result.stdout
    assert f'RESULTS_JSON={output}/groot_n1d6_task021_collecting_childrens_toys_test-run_gpu1/results.json' in result.stdout
    assert not output.exists()


def test_fast_debug_preset_caps_runner_without_changing_native_horizon(tmp_path):
    result = _run_dry_run(tmp_path, "--preset", "fast-debug", "--seed", "17")
    assert result.returncode == 0, result.stderr
    assert "TASK 21: collecting_childrens_toys episodes=1" in result.stdout
    assert "INSTANCE_IDS: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]" in result.stdout
    assert "--max_episode_steps 1200" in result.stdout
    assert "not an official success-rate sample" in result.stdout
    assert "SEED: 17" in result.stdout


def test_debug_step_limit_is_only_available_for_fast_debug(tmp_path):
    debug = _run_dry_run(tmp_path, "--preset", "fast-debug", "--debug-max-steps", "80")
    official = _run_dry_run(tmp_path, "--preset", "official-eval", "--debug-max-steps", "80")
    assert debug.returncode == 0, debug.stderr
    assert "--max_episode_steps 80" in debug.stdout
    assert official.returncode != 0
    assert "requires --preset fast-debug" in official.stderr


def test_explicit_instance_partition_rejects_duplicates_and_excess_episodes(tmp_path):
    duplicate = _run_dry_run(tmp_path, "--tasks", "21", "--instance-ids", "1", "1")
    excess = _run_dry_run(tmp_path, "--tasks", "21", "--instance-ids", "1", "2", "--episodes", "3")
    assert duplicate.returncode != 0 and "must not contain duplicates" in duplicate.stderr
    assert excess.returncode != 0 and "cannot exceed" in excess.stderr


@pytest.mark.parametrize("n_envs", [1, 2, 4])
def test_dry_run_passes_supported_environment_count(tmp_path, n_envs):
    result = _run_dry_run(tmp_path, "--tasks", "21", "--episodes", str(n_envs), "--n-envs", str(n_envs))
    assert result.returncode == 0, result.stderr
    assert f"n_envs={n_envs}" in result.stdout
    assert f"--n_envs {n_envs}" in result.stdout


@pytest.mark.skipif(not SIM_PYTHON.exists(), reason="requires the simulator Python with gymnasium")
@pytest.mark.parametrize("fail_after_first", [False, True])
@pytest.mark.parametrize("public_entry", [False, True])
def test_launcher_keeps_live_episode_results_without_aggregate_log(tmp_path, fail_after_first, public_entry):
    # Exercise the real launcher/bootstrap with a tiny Gym env instead of Isaac Sim.
    repo = tmp_path / "groot"
    runner = repo / "gr00t/eval/rollout_policy.py"
    runner.parent.mkdir(parents=True)
    runner.write_text('''
import os
import time
from pathlib import Path
import gymnasium as gym
import numpy as np

class FakeBehavior(gym.Env):
    observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}
    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0, True, False, {
            "success": [False], "valid": [True], "q_score": [0.2], "task_progress": [33.0], "rewards": [0],
        }

env = gym.vector.SyncVectorEnv([FakeBehavior])
env.reset()
env.step(np.zeros((1, 1)))
while not Path(os.environ["RELEASE_FILE"]).exists():
    time.sleep(0.01)
if os.environ["FAIL_AFTER_FIRST"] == "1":
    raise RuntimeError("intentional runner failure")
env.reset()
env.step(np.zeros((1, 1)))
env.close()
# Deliberately omit the legacy 'results:' aggregate line.
''')
    wrapper = tmp_path / "fake_b1k_python"
    wrapper.write_text(f'''#!{SIM_PYTHON}
import sys
import types
code_index = sys.argv.index("-c") + 1
code = sys.argv[code_index]
sys.argv = [sys.argv[0]] + sys.argv[code_index + 1:]
if "MODALITY_KEYS" in code:
    print("PING True MODALITY_KEYS []")
else:
    sys.modules["isaacsim"] = types.ModuleType("isaacsim")
    exec(code)
''')
    wrapper.chmod(0o755)
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/sh\nprintf '49140, 0\\n'\n")
    smi.chmod(0o755)
    release = tmp_path / "release"
    env = os.environ.copy()
    env.update({
        "PROJECT_ROOT": str(tmp_path), "GROOT_REPO": str(repo),
        "B1K_PYTHON_LAUNCHER": str(wrapper), "B1K_PYTHON": str(SIM_PYTHON),
        "OMNIGIBSON_ROOT": str(tmp_path), "BDDL_ROOT": str(tmp_path),
        "BEHAVIOR_DATA": str(tmp_path), "RUN_TAG": "live-test",
        "RELEASE_FILE": str(release), "FAIL_AFTER_FIRST": str(int(fail_after_first)),
        "PATH": str(tmp_path) + os.pathsep + env["PATH"],
    })
    command = ["bash", str(LAUNCHER), "--gpu", "0", "--tasks", "21", "--episodes", "2",
               "--output-root", str(tmp_path / 'run' / 'gpu0')]
    output = tmp_path / "run/gpu0/groot_n1d6_task021_collecting_childrens_toys_live-test_gpu0"
    log_name = "run.log"
    if public_entry:
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        for name in ("run_groot_priority_eval.sh", "groot_eval_results.py", "groot_eval_output.py"):
            shutil.copy2(LAUNCHER.parent / name, scripts / name)
        for name in ("groot_tasks.py", "groot_live_preview.py", "b1k_metrics.py"):
            shutil.copy2(LAUNCHER.parents[1] / "evaluation/scripts" / name, scripts / name)
        tools = tmp_path / "evaluation/scripts"
        tools.mkdir(parents=True)
        shutil.copy2(LAUNCHER.parents[1] / "evaluation/scripts/run_groot_eval.sh", tools / "run_groot_eval.sh")
        command = ["bash", str(tools / "run_groot_eval.sh"), "--gpu-index", "0",
                   "--episodes-per-gpu", "2", "--instances-gpu0", "0", "1"]
        output = tmp_path / "evaluation/V_1/baseline/task21_eval01"
        log_name = "gpu0.log"
    process = subprocess.Popen(
        command,
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    summary_path = output / "results.json"
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if summary_path.exists() and json.loads(summary_path.read_text())["completed_episodes"] == 1:
                break
            if process.poll() is not None:
                pytest.fail(process.communicate()[0])
            time.sleep(0.02)
        assert summary_path.exists()
        live = json.loads(summary_path.read_text())
        assert live["completed_episodes"] == 1
        assert live["status"] == "running"
        assert live["mean_task_progress"] == 33.0
        assert live["mean_q_score"] == 0.2
        assert live["q_score_episodes"] == 1
        if public_entry:
            records = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
            assert len(records) == 1 and records[0]["gpu_index"] == 0
            assert records[0]["sim_steps"] == 1
            assert records[0]["q_score"] == 0.2
        assert process.poll() is None
        release.touch()
        log, _ = process.communicate(timeout=10)
        summary = json.loads(summary_path.read_text())
        assert summary["status"] == ("failed" if fail_after_first else "complete"), log
        assert summary["completed_episodes"] == (1 if fail_after_first else 2)
        assert summary["success_rate"] == 0.0
        assert summary["mean_q_score"] == 0.2
        assert summary["q_score_episodes"] == summary["completed_episodes"]
        assert "EPISODE_COMPLETE" in (output / log_name).read_text()
        if public_entry:
            assert {path.name for path in output.iterdir()} == {"results.json", "episodes.jsonl", "gpu0.log"}
            assert summary["per_gpu"]["0"]["performance"]["valid_episodes"] == (1 if fail_after_first else 2)
    finally:
        release.touch()
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)


@pytest.mark.skipif(
    not SIM_PYTHON.exists() or not GROOT_REPO.is_dir(),
    reason="requires the 4090 BEHAVIOR and GR00T checkouts",
)
def test_simulator_python_imports_policy_client_without_transformers():
    env = os.environ.copy()
    for name in (
        "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD",
        "ISAAC_PATH", "CARB_APP_PATH", "EXP_PATH",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = str(GROOT_REPO)
    result = subprocess.run(
        [str(SIM_PYTHON), "-B", "-s", "-c",
         "from gr00t.policy.server_client import PolicyClient; print(PolicyClient.__name__)"],
        cwd=GROOT_REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PolicyClient"


@pytest.mark.skipif(
    not SIM_PYTHON.exists() or not GROOT_REPO.is_dir(),
    reason="requires the 4090 BEHAVIOR and GR00T checkouts",
)
def test_runner_selects_remote_policy_without_importing_transformers():
    env = os.environ.copy()
    for name in (
        "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD",
        "ISAAC_PATH", "CARB_APP_PATH", "EXP_PATH",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = str(GROOT_REPO)
    code = (
        "from gr00t.eval.rollout_policy import create_gr00t_sim_policy; "
        "p=create_gr00t_sim_policy('', None, '127.0.0.1', 59999); "
        "print(type(p).__name__); p.socket.close(); p.context.term()"
    )
    result = subprocess.run(
        [str(SIM_PYTHON), "-B", "-s", "-c", code],
        cwd=GROOT_REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PolicyClient"


@pytest.mark.skipif(
    not N1D6_PYTHON.exists() or not GROOT_REPO.is_dir(),
    reason="requires the 4090 N1.6 inference checkout",
)
def test_lazy_policy_exports_keep_gr00t_policy_public_import():
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"):
        env.pop(name, None)
    result = subprocess.run(
        [str(N1D6_PYTHON), "-B", "-s", "-c",
         "from gr00t.policy import Gr00tPolicy; print(Gr00tPolicy.__name__)"],
        cwd=GROOT_REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Gr00tPolicy"
