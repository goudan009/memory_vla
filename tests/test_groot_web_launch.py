"""Dashboard launches the existing CLI with argv, preserving its protocol."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
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


def test_exp11_appdata_reuses_only_idle_cache_slots(tmp_path):
    worker = load("task40_cache_test", ROOT / "evaluation/scripts/task40_restore_worker.py")
    runs = tmp_path / "experiment_11" / "runs"
    first, second, third = (runs / name / "appdata" for name in ("first", "second", "third"))
    lease0 = worker.lease_sim_cache(first, gpu=0, slots=1)
    try:
        with pytest.raises(RuntimeError, match="cache slot"):
            worker.lease_sim_cache(second, gpu=0, slots=1)
        with worker.lease_sim_cache(second, gpu=0, slots=2):
            assert (second / "global/cache").resolve() != (first / "global/cache").resolve()
            assert (second / "local/cache").resolve() != (first / "local/cache").resolve()
            with pytest.raises(RuntimeError, match="cache slot"):
                worker.lease_sim_cache(third, gpu=0, slots=2)
    finally:
        lease0.close()
    with worker.lease_sim_cache(third, gpu=0, slots=1):
        assert (third / "global/cache").resolve() == (first / "global/cache").resolve()
        assert (third / "local/cache").resolve() == (first / "local/cache").resolve()
        assert not (third / "global/data").is_symlink()
        assert not (third / "local/logs").is_symlink()


def test_restore_aa_updates_both_keys_and_rejects_renderer_reset():
    sim = load("task40_paired_sim_test", ROOT / "evaluation/scripts/task40_paired_sim.py")
    app = SimpleNamespace(config={"anti_aliasing": 3})
    settings = dict.fromkeys(sim.RENDER_AA_KEYS, 3)
    settings["/rtx-transient/post/aa/limitedOps"] = True
    settings["/rtx/rendermode"] = "RaytracedLighting"
    carb = SimpleNamespace(get=settings.get, set_int=settings.__setitem__, set_bool=settings.__setitem__)
    assert sim.configure_restore_aa(app, carb, 0) == [0, 0]
    assert settings["/rtx-transient/post/aa/limitedOps"] is False
    assert app.config["anti_aliasing"] == 0
    assert settings["/rtx/rendermode"] == "RaytracedLighting"
    settings[sim.RENDER_AA_KEYS[1]] = 3
    with pytest.raises(RuntimeError, match="did not persist"):
        sim.read_restore_aa(carb, 0)


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


def test_task40_restore_preview_uses_remote_policy_and_one_local_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv('GROOT_TASK40_POLICY_HOST', 'a100-test')
    monkeypatch.delenv('GROOT_TASK40_RUNNER', raising=False)
    manager = launcher()
    assert (manager.task40_runner / 'paired_sim_diagnostic.py').is_file()
    manager.root = tmp_path
    manager.task40_worker = tmp_path / 'evaluation/scripts/task40_restore_worker.py'
    manager.task40_runner = tmp_path / 'Output/test_01/experiment_07/scripts'
    manager.task40_output = tmp_path / 'Output/test_01/experiment_07/workbench_runs'
    manager.task40_worker.parent.mkdir(parents=True)
    manager.task40_worker.touch()
    manager.task40_runner.mkdir(parents=True)
    (manager.task40_runner / 'paired_sim_diagnostic.py').touch()
    anchor = tmp_path / 'anchors/door_front'
    (anchor / 'prefix/0000').mkdir(parents=True)
    for name in ('start.json', 'anchor.json', 'sim_checkpoint.pt', 'memory_online.pt',
                 'anchor_live_observation.npz'):
        (anchor / name).write_text('{}')
    (anchor / 'prefix/0000/decision.json').write_text('{}')
    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'config.json').write_text('{}')
    (weights / 'model.safetensors.index.json').write_text('{}')
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]

    result = manager.prepare({'task40_restore': True, 'restore_checkpoint': str(anchor),
        'restore_weights': str(weights), 'sim_gpu': 0,
        'restore_seed': 731, 'restore_cycles': 3, 'restore_port': port,
        'memory_mode': 'A_normal'})
    assert result['plan']['kind'] == 'task40_restore'
    assert result['plan']['instances'] == {'0': [0]}
    assert result['plan']['policy_host'] == 'a100-test'
    assert '--policy-gpu' not in result['argv']
    assert result['argv'][result['argv'].index('--policy-host') + 1] == 'a100-test'
    assert result['plan']['cycles'] == 3 and result['plan']['seed'] == 731
    assert result['argv'][0:3] == ['python3', '-u', str(manager.task40_worker)]
    assert not Path(result['plan']['output']).exists()


def test_task40_restore_rejects_local_policy_and_bad_anchor(tmp_path):
    manager = launcher()
    manager.root = tmp_path
    manager.task40_worker = tmp_path / 'task40_restore_worker.py'
    manager.task40_runner = tmp_path
    manager.task40_worker.touch()
    (tmp_path / 'paired_sim_diagnostic.py').touch()
    anchor = tmp_path / 'anchor'
    (anchor / 'prefix/0000').mkdir(parents=True)
    for name in ('start.json', 'anchor.json', 'sim_checkpoint.pt', 'memory_online.pt',
                 'anchor_live_observation.npz'):
        (anchor / name).write_text('{}')
    (anchor / 'prefix/0000/decision.json').write_text('{}')
    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'config.json').touch()
    (weights / 'model.safetensors.index.json').touch()
    request = {'task40_restore': True, 'restore_checkpoint': str(anchor),
        'restore_weights': str(weights), 'sim_gpu': 0}
    with pytest.raises(ValueError, match='不接受普通评估参数'):
        manager.prepare(dict(request, policy_gpu=0))
    (anchor / 'sim_checkpoint.pt').unlink()
    with pytest.raises(ValueError, match='检查点目录不完整'):
        manager.prepare(request)


@pytest.fixture(params=['experiment_07/episodes', 'experiment_09/runs'])
def task40_failure_archive(tmp_path, monkeypatch, request):
    experiment_root = tmp_path / 'Output/test_01'
    manager = launcher(log_root=tmp_path / 'logs')
    manager.root = tmp_path
    manager.task40_experiment = experiment_root / 'experiment_09'
    monkeypatch.setenv('GROOT_TASK40_ANCHOR', str(tmp_path))
    monkeypatch.setenv('GROOT_TASK40_WEIGHTS', str(tmp_path))
    identity = dict(version='V_2', task40_diagnostics=True, task40_trace=True, checkpoint='/remote/step-004000')
    manager.policy_resolver = lambda host, port: identity

    run = experiment_root / request.param / 'archived-run'
    branch = 'episode_i3_s78123_A_normal'
    checkpoint = run / 'checkpoints' / f'{branch}_failure'
    parent = run / 'metrics' / branch
    checkpoint.mkdir(parents=True)
    for i in range(2):
        (parent / f'{i:04d}').mkdir(parents=True)

    def write(path, data):
        path.write_text(json.dumps(data))

    language = {'annotation.human.coarse_action': ['Make microwave popcorn.']}
    session = dict(session_id='archived-session', episode_ids=[branch], env_ids=[0])
    initial = dict(step=0, Open=False, ToggledOn=False, hinge=0)
    observations = b'archived pre-action observation'
    truth = [dict(initial, step=step, decision=0, hinge=0.012,
                  microwaves={'microwave': {'joint_positions': [0.012]}},
                  robots={'robot_r1': {'joint_positions': [1]}},
                  interaction={'door_body': 'door',
                               'contacts': {'right': [{'body': 'door'}] if step <= 3 else []}})
             for step in range(1, 9)]
    (parent / 'truth.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in truth))
    write(parent / 'trial.json', dict(branch=branch, instance=3, seed=78123, mode='A_normal',
                                     origin=None, start_step=0, end_step=16, stop_reason='decision_limit'))
    write(parent / 'initial.json', dict(policy=dict(seed=78123, policy_weights=identity['checkpoint']),
                                       truth=initial))
    for i, before in enumerate([initial, truth[-1]]):
        entry = parent / f'{i:04d}'
        write(entry / 'decision.json', dict(decision=i, instance=3, seed=78123, start_step=i * 8,
                                           end_step=(i + 1) * 8, executed_steps=8, session=session,
                                           state_before=before, language=language))
        (entry / 'observation.npz').write_bytes(observations)
        (entry / 'executed.npz').write_bytes(b'actions')
    write(checkpoint / 'source.json', dict(branch=branch, decision=1, step=8, language=language,
          evidence=dict(door_contact_steps=[1, 2, 3], final_open=False, final_toggled_on=False),
          memory_replay=dict(online_snapshot_saved=True)))
    write(checkpoint / 'physical.json', {'joints': {'microwave': [0.012], 'robot_r1': [1]}})
    for name in ('sim.pt', 'memory.pt', 'policy_rng_online.pt', 'observation.npz'):
        (checkpoint / name).write_bytes(observations)
    payload = dict(task40_collect=True, restore_checkpoint=checkpoint.relative_to(tmp_path).as_posix(),
                   policy_port=5586)
    return manager, checkpoint, parent, payload, identity


def test_task40_directed_restore_prepares_existing_worker_plan(task40_failure_archive):
    manager, checkpoint, parent, payload, _ = task40_failure_archive
    prepared = manager.prepare(payload)
    plan, argv = prepared['plan'], prepared['argv']
    assert json.loads(argv[-1]) == plan and argv[-2] == '--experiment-plan'
    assert '--collect-instances' not in argv
    assert plan['kind'] == 'task40_collect' and plan['phase'] == 'directed_restore'
    assert plan['group'] == 'experiment_09'
    assert plan['seed'] == 78123 and plan['instances'] == {'0': [3]}
    assert plan['cycles'] == 60 and plan['sim_slots'] == 1
    assert Path(plan['restore_checkpoint']) == checkpoint
    assert Path(plan['lineage']['prefix']) == parent
    assert plan['lineage']['origin_id'] == 'instance=3/seed=78123'
    assert plan['lineage']['start_step'] == 8 and plan['lineage']['start_decision'] == 1
    output = Path(plan['output'])
    assert output.parent == manager.task40_experiment / 'runs' and not output.exists()
    repeated = manager.prepare(dict(payload, restore_repeats=2))
    assert repeated['plan']['restore_repeats'] == 2
    assert repeated['argv'][-5:-2] == ['--restore-repeats', '2', '--renderer-audit']
    aa = manager.prepare(dict(payload, restore_repeats=2, renderer_aa=0))
    assert aa['plan']['renderer_aa'] == 0
    assert aa['argv'][-4:-2] == ['--renderer-aa', '0']
    with pytest.raises(ValueError, match='渲染 AA'):
        manager.prepare(dict(payload, renderer_aa=0))
    with pytest.raises(ValueError, match='渲染 AA'):
        manager.prepare(dict(payload, restore_repeats=2, renderer_aa=False))
    with pytest.raises(ValueError, match='重复次数'):
        manager.prepare(dict(payload, restore_repeats=3))


def test_exp11_can_schedule_real_approach_capture_separately(task40_failure_archive):
    manager, _, _, directed, identity = task40_failure_archive
    request = dict(task40_collect=True, experiment="experiment_11", seed=77001, instances=[4],
                   cycles=300, policy_port=5586, capture_approach=True)
    plan = manager.prepare(request)["plan"]
    assert plan["phase"] == "approach_capture" and plan["capture_approach"] is True
    with pytest.raises(ValueError):
        manager.prepare(dict(request, full_task_eval=True))
    with pytest.raises(ValueError):
        manager.prepare(dict(directed, capture_approach=True))
    identity.update(method="official", task40_trace=False)
    with pytest.raises(ValueError):
        manager.prepare(request)


def test_task40_fixed_action_probe_uses_archived_trajectory_without_policy(task40_failure_archive):
    manager, checkpoint, parent, payload, identity = task40_failure_archive
    manager.policy_resolver = None
    request = dict(payload, restore_repeats=2, renderer_probe=True, cycles=2)
    del request['policy_port']
    plan = manager.prepare(request)['plan']
    assert plan['phase'] == 'renderer_probe'
    assert plan['policy_port'] is None
    assert plan['policy_identity']['checkpoint'] == identity['checkpoint']
    assert Path(plan['restore_checkpoint']) == checkpoint
    assert Path(plan['lineage']['prefix']) == parent
    no_diffuse = manager.prepare(dict(request, renderer_disable_indirect_diffuse=True))
    assert no_diffuse['plan']['renderer_disable_indirect_diffuse'] is True
    assert 'renderer_disable_indirect_diffuse' not in plan
    with pytest.raises(ValueError, match='间接漫反射'):
        manager.prepare(dict(request, renderer_disable_indirect_diffuse='true'))
    with pytest.raises(ValueError, match='间接漫反射'):
        manager.prepare(dict(payload, renderer_disable_indirect_diffuse=True))
    with pytest.raises(ValueError, match='固定动作诊断'):
        manager.prepare(dict(request, policy_port=5586))
    with pytest.raises(ValueError, match='固定动作诊断'):
        manager.prepare(dict(request, restore_repeats=1))


def test_task40_directed_restore_rejects_mismatched_parent(task40_failure_archive):
    manager, checkpoint, parent, payload, _ = task40_failure_archive
    decision = parent / '0000/decision.json'
    original = json.loads(decision.read_text())
    decision.write_text(json.dumps(dict(original, end_step=7)))
    with pytest.raises(ValueError, match='动作块'):
        manager.prepare(payload)
    decision.write_text(json.dumps(original))
    (checkpoint / 'observation.npz').write_bytes(b'different observation')
    with pytest.raises(ValueError, match='锚点观测'):
        manager.prepare(payload)


def test_task40_exp10_routes_to_separate_archive(task40_failure_archive):
    manager, _, _, _, identity = task40_failure_archive
    collect = dict(task40_collect=True, experiment='experiment_10', seed=76500, instances=[2, 4],
                   policy_port=5692)
    exp10 = manager.task40_experiment.with_name('experiment_10')
    plan = manager.prepare(collect)['plan']
    assert plan['group'] == 'experiment_10'
    assert Path(plan['output']).parent == exp10 / 'runs'
    exp10.mkdir(parents=True)
    batch = exp10 / 'batch.json'
    batch.write_text('{}')
    forward = dict(task40_forward=True, experiment='experiment_10', batch=str(batch), policy_port=5698)
    assert Path(manager.prepare(forward)['plan']['output']).parent == exp10 / 'runs'
    with pytest.raises(ValueError, match='所选实验'):
        manager.prepare(dict(forward, experiment='experiment_09'))
    with pytest.raises(ValueError, match='不支持'):
        manager.prepare(dict(collect, experiment='../experiment_08'))
    identity['task40_trace'] = False
    with pytest.raises(ValueError, match='增强诊断'):
        manager.prepare(collect)


def test_exp12_initial_official_collection_uses_registered_remote_weights(task40_failure_archive, monkeypatch):
    manager, _, _, _, identity = task40_failure_archive
    identity.update(method='official', checkpoint='/ada/models/GR00T-N1.6-BEHAVIOR1k')
    monkeypatch.setenv('GROOT_TASK40_WEIGHTS', str(manager.root / 'missing-legacy-weights'))
    prepared = manager.prepare(dict(task40_collect=True, experiment='experiment_12',
                                    seed=77001, instances=[4], policy_host='ada', policy_port=5812))
    assert prepared['plan']['official_policy']
    argv = prepared['argv']
    assert argv[argv.index('--weights') + 1] == identity['checkpoint']
    assert '--collect-instances' in argv


def test_task40_full_task_protocol_is_opt_in_and_initial_only(task40_failure_archive):
    manager, _, _, directed, _ = task40_failure_archive
    collect = dict(task40_collect=True, experiment='experiment_10', seed=76500,
                   instances=[2, 4], policy_port=5692, cycles=300)
    assert 'full_task_eval' not in manager.prepare(collect)['plan']
    plan = manager.prepare(dict(collect, full_task_eval=True))['plan']
    assert plan['full_task_eval'] is True and plan['phase'] == 'full_task_eval'
    assert plan['cycles'] == 300
    for invalid in (dict(collect, experiment='experiment_09'),
                    dict(collect, controlled_reclosure=True), dict(directed, experiment='experiment_10')):
        with pytest.raises(ValueError, match='完整任务评估'):
            manager.prepare(dict(invalid, full_task_eval=True))


def test_task40_directed_restore_requires_exclusive_enhanced_port(task40_failure_archive):
    manager, _, _, payload, identity = task40_failure_archive
    identity['task40_trace'] = False
    with pytest.raises(ValueError, match='增强诊断'):
        manager.prepare(payload)
    identity['task40_trace'] = True
    identity['checkpoint'] = '/remote/other-weights'
    with pytest.raises(ValueError, match='权重一致'):
        manager.prepare(payload)
    identity['checkpoint'] = '/remote/step-004000'
    ordinary = dict(policy_server='172.28.62.17:5586', instances={'1': [0]})
    directed = manager.prepare(payload)['plan']
    for active, incoming in ((ordinary, directed), (directed, ordinary)):
        manager.jobs = [dict(plan=active, process=SimpleNamespace(poll=lambda: None),
                             gpus=set(active['instances']))]
        manager.prepare = lambda request: {'plan': incoming}
        with pytest.raises(ValueError, match='端口已绑定'):
            manager.start(payload)


def test_task40_directed_restore_first_decision_admits_neighbor(task40_failure_archive, monkeypatch):
    manager, _, _, payload, _ = task40_failure_archive
    directed = manager.prepare(payload)['plan']
    folder = Path(directed['output']) / 'metrics' / 'restore_78123_A_normal' / '0001'
    folder.mkdir(parents=True)
    manager.jobs = [dict(plan=directed, process=SimpleNamespace(poll=lambda: None), gpus={'0'})]
    incoming = dict(payload, policy_port=5587, sim_slots=2)
    with pytest.raises(ValueError, match='首个 worker'):
        manager.start(incoming)
    (folder / 'decision.json').write_text('{}')

    class LaunchReached(Exception):
        pass

    def mark_launch(*args, **kwargs):
        raise LaunchReached

    monkeypatch.setattr(subprocess, 'Popen', mark_launch)
    with pytest.raises(LaunchReached):
        manager.start(incoming)


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


def test_http_task40_restore_stop_routes_to_restore_manager():
    from types import SimpleNamespace
    viewer=load('task40_restore_stop_http',ROOT/'evaluation/scripts/view_results.py')
    calls=[]
    class Handler(viewer.Handler):
        launch_manager=SimpleNamespace(
            stop_restore=lambda value:calls.append(value) or dict(state='stopping'))
        evaluation_devices=None
    server=viewer.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        request=Request(f'http://127.0.0.1:{server.server_port}/api/launch/restore-stop',
            data=b'{"job":"restore-job"}',headers={'Content-Type':'application/json'})
        with urlopen(request) as response:
            assert response.status==202 and json.load(response)['state']=='stopping'
        assert calls==[{'job':'restore-job'}]
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


def test_registry_resource_removal_and_restore(tmp_path, monkeypatch):
    viewer = load("registry_viewer", ROOT / "evaluation/scripts/view_results.py")
    resource = tmp_path / "weights.safetensors"
    resource.write_text("fixture")
    annotations = tmp_path / "annotations.json"
    monkeypatch.setattr(viewer, "REGISTRY_ANNOTATIONS", annotations)
    monkeypatch.setattr(viewer.experiment_registry, "cached", lambda: {"items": [
        {"id": "weights:test", "kind": "weights", "path": str(resource)},
        {"id": "dataset:test", "kind": "dataset"},
    ]})

    with viewer.ThreadingHTTPServer(("127.0.0.1", 0), viewer.Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}/api/registry/annotations"
        try:
            for key in ("weights:test", "dataset:test"):
                for hidden in (True, False):
                    request = Request(url, data=json.dumps({"key": key, "hidden": hidden}).encode(),
                                      headers={"Content-Type": "application/json"})
                    with urlopen(request, timeout=3) as response:
                        result = json.load(response)
                    with urlopen(url, timeout=3) as response:
                        assert json.load(response) == result
                    assert (key in result) is hidden
            assert resource.read_text() == "fixture"
        finally:
            server.shutdown()
            thread.join(timeout=3)


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
