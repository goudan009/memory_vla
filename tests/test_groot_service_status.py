"""Check service identity and sockets without launching Kit or a policy model."""

import importlib.util
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


status = load("groot_service_status", "evaluation/scripts/groot_service_status.py")


def test_process_snapshot_counts_rollout_once_and_reports_actual_ports(tmp_path, monkeypatch):
    output = tmp_path / "runtime"
    output.mkdir()
    (output / "run_meta.json").write_text(json.dumps({
        "task_id": 21,
        "public_output_dir": "/project/evaluation/V_1/baseline/groot_task21_2",
    }))
    def process(pid, argv, environment="", state="S"):
        folder = tmp_path / str(pid)
        folder.mkdir()
        (folder / "status").write_text(f"State:\t{state} (test)\n")
        (folder / "cmdline").write_bytes("\0".join(argv).encode())
        (folder / "environ").write_bytes(environment.encode())

    process(10, ["python", "launcher.py", "--n-envs", "2"])
    process(11, ["python", "-s", "-u", "-c",
                 "runpy.run_path('gr00t/eval/rollout_policy.py')", "--env_name", "task21",
                 "--n_envs", "2", "--policy_client_host", "172.28.62.17", "--policy_client_port", "5556"],
            f"OMNIGIBSON_GPU_ID=1\0GROOT_EVAL_OUTPUT_DIR={output}\0PRIVATE_TOKEN=do-not-return\0")
    process(12, ["python", "-c", "from multiprocessing.spawn import spawn_main; spawn_main()"])
    process(13, ["python", "-m", "gr00t.eval.run_gr00t_server", "--port=5556", "--model-path", "/models/base"],
            "CUDA_VISIBLE_DEVICES=0\0")
    process(14, ["python", "-m", "v1_groot_memoryvla.serve", "--port", "5562"])
    process(15, ["python", "-m", "gr00t.eval.run_gr00t_server"], state="Z")
    process(16, ["bash", "run_groot_priority_eval.sh"])
    process(17, ["python", "train.py", "--port", "5556"])
    sockets = '\n'.join([
        'LISTEN 0 128 0.0.0.0:5556 0.0.0.0:* users:(("python",pid=13,fd=9))',
        'LISTEN 0 128 [::]:5556 [::]:* users:(("python",pid=13,fd=10))',
        'LISTEN 0 128 *:8765 *:* users:(("python",pid=99,fd=8))',
    ])
    monkeypatch.setattr(status.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=sockets))
    result = status.collect_snapshot(tmp_path)
    assert result["ok"] is True
    assert [p["pid"] for p in result["processes"]] == [11, 13, 14]
    evaluation, listening, starting = result["processes"]
    assert evaluation["n_envs"] == 2
    assert evaluation["gpu"] == "1"
    assert evaluation["policy_port"] == "5556"
    assert evaluation["task_id"] == 21
    assert evaluation["run_directory"] == "/project/evaluation/V_1/baseline/groot_task21_2"
    assert evaluation["listening_ports"] == []
    assert listening["gpu"] == "0"
    assert listening["listening_ports"] == [5556]
    assert starting["listening_ports"] == []
    assert "PRIVATE_TOKEN" not in json.dumps(result)
    assert "do-not-return" not in json.dumps(result)


def test_remote_failure_is_unknown_and_does_not_hide_local_status(monkeypatch):
    monkeypatch.setattr(status, "collect_snapshot", lambda: {"ok": True, "processes": []})

    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 6)

    monkeypatch.setattr(status.subprocess, "run", unavailable)
    monitor = status.ServiceMonitor()
    monitor.poll()
    snapshot = monitor.snapshot()
    assert snapshot["local"]["ok"] is True
    assert snapshot["remote"]["ok"] is False
    assert "processes" not in snapshot["remote"]


def test_gpu_memory_snapshot_reports_all_cards_without_loading_cuda(tmp_path, monkeypatch):
    calls=[]
    def query(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout='0, NVIDIA A100-SXM4-80GB, 81920, 74240, 99\n1, NVIDIA A100-SXM4-80GB, 81920, 1024, 0\n' if argv[0]=='nvidia-smi' else '')
    monkeypatch.setattr(status.subprocess,'run',query)
    result=status.collect_snapshot(tmp_path)
    assert result['gpus']==[
        dict(index=0,name='NVIDIA A100-SXM4-80GB',total_mb=81920,used_mb=74240,utilization=99),
        dict(index=1,name='NVIDIA A100-SXM4-80GB',total_mb=81920,used_mb=1024,utilization=0)]
    assert sum(argv[0]=='nvidia-smi' for argv in calls)==1


def test_gpu_query_failure_keeps_service_snapshot_available(tmp_path, monkeypatch):
    def query(argv, **kwargs):
        if argv[0]=='nvidia-smi':raise subprocess.TimeoutExpired(argv,2)
        return SimpleNamespace(stdout='')
    monkeypatch.setattr(status.subprocess,'run',query)
    result=status.collect_snapshot(tmp_path)
    assert result['ok'] and result['processes']==[]
    assert result['gpus_error'] and 'gpus' not in result


def test_optional_device_failure_preserves_other_hosts_and_last_seen(monkeypatch):
    monitor=status.ServiceMonitor(additional_hosts={'a6000':'user@optional'})
    monitor._value['remote']={'ok':True,'processes':[{'pid':12}]}
    monkeypatch.setattr(status.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='{"ok":true,"gpus":[],"processes":[]}'))
    monitor.poll_remote('a6000','user@optional')
    last_seen=monitor.snapshot()['a6000']['last_seen']
    def offline(*a,**kw):
        raise subprocess.TimeoutExpired(a[0],6)
    monkeypatch.setattr(status.subprocess,'run',offline)
    monitor.poll_remote('a6000','user@optional')
    snapshot=monitor.snapshot()
    assert snapshot['remote']=={'ok':True,'processes':[{'pid':12}]}
    assert snapshot['a6000']['ok'] is False
    assert snapshot['a6000']['last_seen']==last_seen
    assert 'processes' not in snapshot['a6000'] and 'gpus' not in snapshot['a6000']


def test_optional_device_poll_does_not_block_primary_monitor(monkeypatch):
    entered=threading.Event();release=threading.Event();primary=threading.Event()
    monitor=status.ServiceMonitor(interval=.02,additional_hosts={'a6000':'optional'})
    monkeypatch.setattr(monitor,'poll',lambda:primary.set() or .02)
    def blocked(key,host):
        entered.set();release.wait(2);return .02
    monkeypatch.setattr(monitor,'poll_remote',blocked)
    monitor.start()
    try:
        assert entered.wait(1)
        primary.clear()
        assert primary.wait(1), 'optional SSH blocked primary polling'
    finally:
        release.set();monitor.close()


def test_training_snapshot_classifies_controller_without_counting_rank():
    assert status.classify(['python','-m','torch.distributed.run','--module',
        'v1_groot_memoryvla.experiment.task21_ddp'])=='training'
    assert status.classify(['python','-m','v1_groot_memoryvla.experiment.task21_ddp']) is None


def test_services_endpoint_returns_cached_snapshot_without_ssh():
    viewer = load("service_test_viewer", "evaluation/scripts/view_results.py")
    snapshot = {"local": {"ok": True, "processes": []}, "remote": {"ok": False}}
    viewer.Handler.service_monitor = SimpleNamespace(snapshot=lambda: snapshot)
    server = viewer.ThreadingHTTPServer(("127.0.0.1", 0), viewer.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/services", timeout=2) as response:
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            assert json.load(response) == snapshot
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
