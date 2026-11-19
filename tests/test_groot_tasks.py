"""Check the full B1K-2025 task catalog without importing Isaac Sim."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "evaluation/scripts"
ENTRY = TOOLS / "run_groot_eval.sh"
PRIORITY = TOOLS / "run_groot_priority_eval.sh"
GROOT = Path(os.environ.get("GROOT_REPO", "/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1"))
spec = importlib.util.spec_from_file_location("groot_tasks", TOOLS / "groot_tasks.py")
tasks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tasks)


@pytest.mark.parametrize("adapter", [
    GROOT / "gr00t/eval/sim/BEHAVIOR/behavior_env.py",
    ROOT / "evaluation/sim/BEHAVIOR/behavior_env.py",
])
def test_catalog_matches_adapter_task_indices_and_gym_registrations(adapter):
    if not adapter.is_file():
        pytest.skip("adapter is not installed")
    tree = ast.parse(adapter.read_text())
    mapping = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(name, ast.Name) and name.id == "TASK_NAMES_TO_INDICES"
                           for name in node.targets))
    assert tasks.TASK_NAMES == {index: name for name, index in mapping.items()}
    registrations = {
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "register"
        for keyword in node.keywords
        if keyword.arg == "id" and isinstance(keyword.value, ast.Constant)
    }
    assert registrations == {f"sim_behavior_r1_pro/{name}" for name in tasks.TASK_NAMES.values()}
    assert sorted(tasks.TASK_NAMES) == list(range(50))


@pytest.mark.parametrize("task_id", range(50))
def test_every_task_reaches_the_matching_runner_without_loading_isaac(task_id):
    result = subprocess.run([
        "bash", str(ENTRY), "--dry-run", "--task", str(task_id),
        "--preset", "fast-debug", "--debug-max-steps", "256", "--seed", "0",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert plan["task_id"] == task_id
    assert plan["instances"] == {"1": [0]}
    child = subprocess.run(plan["commands"][0] + ["--dry-run"], capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    assert f"--env_name sim_behavior_r1_pro/{tasks.TASK_NAMES[task_id]}" in child.stdout
    assert "--max_episode_steps 256" in child.stdout


def test_list_tasks_reports_actual_local_instance_files(tmp_path):
    directory = tmp_path / "gr00t/eval/sim/BEHAVIOR/test_instances/turning_on_radio"
    directory.mkdir(parents=True)
    for instance in range(10):
        (directory / f"house_task_turning_on_radio_0_{instance}_template-tro_state.json").write_text("{}")
    assert tasks.missing_instances(tmp_path, 0) == []
    assert tasks.missing_instances(tmp_path, 1) == list(range(10))
    for launcher in (ENTRY, PRIORITY):
        result = subprocess.run(["bash", str(launcher), "--list-tasks"],
                                env={**os.environ, "GROOT_REPO": str(tmp_path)},
                                text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        assert len(lines) == 50
        assert "ready (10/10)" in lines[0]
        assert "missing instances: 0,1,2,3,4,5,6,7,8,9" in lines[1]


def test_missing_task_instances_stop_before_simulator_and_output_creation(tmp_path):
    adapter = tmp_path / "gr00t/eval/sim/BEHAVIOR/behavior_env.py"
    adapter.parent.mkdir(parents=True)
    adapter.touch()
    environment = {**os.environ, "GROOT_REPO": str(tmp_path),
                   "B1K_PYTHON_LAUNCHER": "/bin/true", "B1K_PYTHON": "/usr/bin/python3",
                   "OMNIGIBSON_ROOT": str(tmp_path), "BDDL_ROOT": str(tmp_path),
                   "BEHAVIOR_DATA": str(tmp_path)}
    output = tmp_path / "output"
    result = subprocess.run(["bash", str(PRIORITY), "--tasks", "7", "--preset", "fast-debug",
                             "--episodes", "1", "--instance-id", "0", "--output-root", str(output)],
                            env=environment, text=True, capture_output=True)
    assert result.returncode != 0
    assert "Missing Task7 test instance TRO states: [0]" in result.stderr
    assert "No simulator started" in result.stderr
    assert not output.exists()
