"""Dashboard launches the existing CLI with argv, preserving its protocol."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/scripts"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launcher(**kwargs):
    module = load("web_launch", ROOT / "evaluation/scripts/groot_web_launch.py")
    return module.LaunchManager(ROOT, **kwargs)


def test_preview_uses_real_cli_defaults_without_creating_result_folder():
    manager = launcher()
    plan = manager.prepare({"version": "web_preview_test", "group": "baseline",
                            "mode": "single2", "gpu_index": 1, "task": 22})
    assert plan["plan"]["episodes_per_gpu"] == 10
    assert plan["plan"]["n_envs_per_gpu"] == 2
    assert plan["plan"]["record_video"] is True
    assert plan["plan"]["video_speedup"] == 16
    assert plan["plan"]["runner_max_episode_steps"] == 999999999
    assert not Path(plan["plan"]["output"]).exists()
    assert "--dry-run" not in plan["argv"]


def test_task40_plan_records_read_only_stage_trace(monkeypatch):
    monkeypatch.delenv('GROOT_TASK40_TRACE',raising=False)
    assert launcher().prepare(dict(task=40))['plan']['task40_stage_trace'] is True
    assert launcher().prepare(dict(task=0))['plan']['task40_stage_trace'] is False
    monkeypatch.setenv('GROOT_TASK40_TRACE','0')
    assert launcher().prepare(dict(task=40))['plan']['task40_stage_trace'] is False


@pytest.mark.parametrize("payload", [
    {"version": "../../bad"}, {"group": "baseline; touch bad"},
    {"task": 50}, {"policy_port": 0}, {"preset": "official-eval", "debug_max_steps": 256},
    {"command": "touch bad"}, {"gpu_index": [0, 1]},
])
def test_rejects_invalid_or_unrecognized_parameters(payload):
    with pytest.raises(ValueError):
        launcher().prepare(payload)


def test_web_plan_binds_observed_policy_to_training_without_changing_protocol():
    manager=launcher()
    identity=dict(version='V_2',method='memory',training_id='V_2_memory_task22_train02',
        checkpoint='/weights/checkpoints/step-003000',source_digest='frozen-code',policy_seed='42',
        binding='observed_process')
    calls=[]
    manager.policy_resolver=lambda host,port:calls.append((host,port)) or identity
    plan=manager.prepare(dict(version='V_2',group='memory',task=22,policy_port=5560))['plan']
    assert calls==[('172.28.62.17',5560)]
    assert plan['policy_identity']==identity
    assert plan['method']=='memory' and plan['naming_schema']=='groot-runs-v2'
    assert Path(plan['output']).name.startswith('task22_eval')
    assert plan['runner_max_episode_steps']==999999999 and plan['action_chunk']==8
    assert plan['base_seed']==0
    with pytest.raises(ValueError,match='实际方法'):
        manager.prepare(dict(version='V_2',group='baseline',task=22,policy_port=5560))


def test_debug_dual_uses_disjoint_instances_and_ignores_single_gpu_field():
    preview = launcher().prepare({"mode": "dual2", "gpu_index": 1,
                                 "preset": "fast-debug", "debug_max_steps": 256,
                                 "record_video": False, "seed": 7, "policy_port": 5561})
    plan = preview["plan"]
    assert plan["instances"] == {"0": [0, 1], "1": [2, 3]}
    assert plan["runner_max_episode_steps"] == 256
    assert plan["record_video"] is False
    assert plan["policy_server"].endswith(":5561")


def test_background_launch_reports_actual_output_exit_and_prevents_gpu_overlap(tmp_path):
    fake = tmp_path / "launcher.sh"
    plan = json.dumps({"output": str(tmp_path / "real_run"), "instances": {"1": [0]}})
    fake.write_text("#!/bin/sh\nprintf '%s\n' '" + plan + "'\n"
                    "case \"$*\" in *--dry-run*) exit 0;; esac\nsleep 0.3\nexit 7\n")
    manager = launcher(log_root=tmp_path / "logs")
    manager.script = fake
    first = manager.start({"gpu_index": 1})
    assert first["state"] == "running"
    with pytest.raises(ValueError, match="GPU"):
        manager.start({"gpu_index": 1})
    for _ in range(40):
        result = manager.snapshot()[0]
        if result["state"] != "running":
            break
        time.sleep(.025)
    assert result["state"] == "failed" and result["returncode"] == 7
    assert result["output"] == str(tmp_path / "real_run")
    assert Path(result["log"]).is_file()


def test_resume_rejects_paths_outside_evaluation_and_passes_only_explicit_overrides():
    with pytest.raises(ValueError):
        launcher().prepare({"resume_from": "../../etc"})
    argv = launcher().argv({"resume_from": "V_1/baseline/test", "policy_port": 5561})
    assert argv[-2:] == ["--policy-port", "5561"]
    assert "--seed" not in argv and "--version" not in argv


def test_http_pause_delete_use_scoped_manager_without_launching(tmp_path):
    from types import SimpleNamespace
    viewer=load('eval_controls_http',ROOT/'evaluation/scripts/view_results.py')
    calls=[]
    class Handler(viewer.Handler):
        launch_manager=SimpleNamespace(
            pause=lambda value:calls.append(('pause',value)) or dict(state='stopping'),
            delete=lambda value:calls.append(('delete',value)) or dict(state='deleted'))
    server=viewer.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        for action in ['pause','delete']:
            request=Request(f'http://127.0.0.1:{server.server_port}/api/launch/{action}',
                data=b'{"run":"V_2/memory/task21_eval01"}',headers={'Content-Type':'application/json'})
            with urlopen(request) as response:assert response.status==202
        assert calls==[(a,dict(run='V_2/memory/task21_eval01')) for a in ['pause','delete']]
    finally:server.shutdown();server.server_close();thread.join()


def test_http_preview_and_start_use_same_manager_and_json_errors(tmp_path):
    viewer = load("web_launch_viewer", ROOT / "evaluation/scripts/view_results.py")
    manager = launcher(log_root=tmp_path)
    class Handler(viewer.Handler):
        launch_manager = manager
        root = ROOT / "evaluation"
    server = viewer.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url + "/api/launch/options", timeout=3) as response:
            assert len(json.load(response)["tasks"]) == 50
        request = Request(url + "/api/launch/preview", data=json.dumps({"mode": "single2"}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            assert json.load(response)["plan"]["episodes_per_gpu"] == 10
        request = Request(url + "/api/launch/start", data=b'{"task":99}',
                          headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=5)
        assert error.value.code == 400
        assert json.load(error.value)["error"]
        assert manager.snapshot() == []
        fake = tmp_path / "fake.sh"
        plan = json.dumps({"instances": {"1": [0]}, "output": str(tmp_path / "fake_result")})
        fake.write_text("#!/bin/sh\nprintf '%s\n' '" + plan + "'\n")
        manager.script = fake
        request = Request(url + "/api/launch/start", data=b'{"gpu_index":1}',
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            assert response.status == 202 and json.load(response)["pid"] > 0
        with urlopen(url + "/api/launch/jobs", timeout=3) as response:
            assert len(json.load(response)["jobs"]) == 1
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)


def test_workbench_device_config_is_optional_and_validated(tmp_path):
    viewer = load("workbench_device_config", ROOT / "evaluation/scripts/view_results.py")
    assert viewer.load_workbench_devices(tmp_path) == {}

    config = tmp_path / "configs" / "workbench_devices.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"local-test": {"host": "127.0.0.1"}}))
    assert viewer.load_workbench_devices(tmp_path) == {"local-test": {"host": "127.0.0.1"}}

    config.write_text("[]")
    with pytest.raises(ValueError, match="必须是 JSON 对象"):
        viewer.load_workbench_devices(tmp_path)
