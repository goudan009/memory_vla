"""Benchmark job lifecycle and HTTP contracts, without loading a model or simulator."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'evaluation/scripts'))
import benchmark_dashboard as dashboard
import view_results as viewer


@pytest.fixture
def manager(tmp_path, monkeypatch):
    value = dashboard.BenchmarkManager(ROOT)
    value.output = tmp_path
    value.runs = tmp_path/'web'
    original = value.options
    monkeypatch.setattr(dashboard, 'local_gpus', lambda: [dict(index=0, name='test', free_mb=48000, total_mb=49152)])
    # Validate plans independently of machine-specific model/simulator installation.
    def options():
        result = original()
        for item in result['catalog']:
            item.update(available=True, missing=[])
        for suite in result['suites']:
            suite['tasks'] = [dict(id=i, name=f'Task {i}') for i in range(10)]
        return result
    monkeypatch.setattr(value, 'options', options)
    return value


def test_plans_separate_environment_smoke_from_fixed_model_controls(manager):
    plan = manager.prepare(dict(kind='libero', suite='libero_goal', tasks=[1, 3], steps=20, seed=7))
    assert len(plan['stages']) == 2 and plan['config']['tasks'] == [1, 3]
    for stage in plan['stages']:
        assert '--save-frames' in stage['argv'] and '--checkpoint' not in stage['argv']
        assert stage['argv'][0].endswith('env/libero-smoke/bin/python')
    plan = manager.prepare(dict(kind='action', arm='all', steps=100))
    assert [s['report'] for s in plan['stages']] == ['baseline.json', 'direct_empty.json', 'memory.json']
    starts = [s['argv'][s['argv'].index('--checkpoint')+1] for s in plan['stages']]
    assert len(set(starts)) == 1 and starts[0].endswith('groot_n1d6_base')


@pytest.mark.parametrize('payload', [dict(kind=[]), dict(kind='libero', suite=[]),
    dict(kind='libero', tasks=[0, 0]), dict(kind='libero', tasks=[10]),
    dict(kind='libero', tasks=[]), dict(kind='libero', steps=0),
    dict(kind='action', steps=301), dict(kind='action', arm='unknown'),
    dict(gpu=True), dict(gpu=9), dict(command='sh'), dict(checkpoint='/arbitrary')])
def test_invalid_inputs_cannot_launch(manager, payload):
    with pytest.raises(ValueError):
        manager.prepare(payload)
    assert not manager.runs.exists()


def test_memory_capacity_is_checked_before_creating_job(manager, monkeypatch):
    monkeypatch.setattr(dashboard, 'local_gpus', lambda: [dict(index=0, free_mb=8000)])
    assert not manager.prepare(dict(kind='action'))['ready']
    with pytest.raises(ValueError, match='显存'):
        manager.start(dict(kind='action'))
    assert not manager.runs.exists()


def test_active_job_on_other_gpu_does_not_block_start(manager, monkeypatch):
    monkeypatch.setattr(manager, 'snapshot', lambda: {'jobs': [
        {'status': 'running', 'config': {'gpu': 1}}]})
    # Reach process creation without actually launching an external workload.
    def no_launch(*args, **kwargs):
        raise OSError('launch reached')
    monkeypatch.setattr(dashboard.subprocess, 'Popen', no_launch)
    with pytest.raises(OSError, match='launch reached'):
        manager.start({'kind': 'network', 'gpu': 0})


def test_active_job_on_selected_gpu_blocks_duplicate_start(manager, monkeypatch):
    monkeypatch.setattr(manager, 'snapshot', lambda: {'jobs': [
        {'status': 'running', 'config': {'gpu': 0}}]})
    with pytest.raises(ValueError, match='所选GPU'):
        manager.start({'kind': 'network', 'gpu': 0})
    assert not manager.runs.exists()


def test_v2_smoke_plan_records_fixed_initialization_and_test_labels(manager):
    plan = manager.prepare(dict(kind='v2_training_smoke', steps=3, gpu=0))
    assert plan['ready'] and plan['config']['category'] == 'engineering_smoke'
    assert plan['config']['model_version'] == 'v2'
    assert plan['config']['checkpoint'].endswith('v1__test01__exp11__step006000')
    assert 'NOT physical truth' in plan['config']['labels']
    assert plan['stages'][0]['argv'][2] == 'v2_groot_memoryvla.training_smoke'
    for steps in (0, 101, True):
        with pytest.raises(ValueError, match='1–100'):
            manager.prepare(dict(kind='v2_training_smoke', steps=steps))


def test_v2_label_audit_uses_isolated_simulator_and_records_no_training(manager):
    plan = manager.prepare(dict(kind='v2_label_audit', gpu=0))
    assert plan['ready'] and plan['config']['training_steps'] == 0
    assert plan['config']['category'] == 'label_alignment_audit'
    argv = plan['stages'][0]['argv']
    assert 'OMNIGIBSON_GPU_ID=0' in argv
    assert 'OMNIGIBSON_APPDATA_PATH={run}/isaac_appdata' in argv
    assert any(x.endswith('/scripts/b1k_python.sh') for x in argv)
    decode = manager.prepare(dict(kind='v2_state_decode', gpu=0))
    assert decode['config']['training_steps'] == 0
    assert '--deserialize-only' in decode['stages'][0]['argv']
    assert len(plan['stages']) == 3
    assert len({s['report'] for s in plan['stages']}) == 3
    export = manager.prepare(dict(kind='v2_labels_export',gpu=0))
    assert '--all-episodes' in export['stages'][0]['argv']


def test_v2_real_training_is_not_artificial_smoke(manager):
    plan = manager.prepare(dict(kind='v2_train',gpu=0,steps=1000))
    assert plan['config']['category'] == 'exploratory_validation'
    assert plan['config']['steps'] == 1000
    assert plan['config']['label_manifest'].endswith('certified_started_labels.json')
    assert plan['stages'][-1]['argv'][2] == 'v2_groot_memoryvla.task40_training'
    assert len(plan['stages']) == 3
    assert '--all-episodes' in plan['stages'][0]['argv']
    for steps in (3,1001,True):
        with pytest.raises(ValueError):
            manager.prepare(dict(kind='v2_train',steps=steps))


def test_v2_evaluation_requires_complete_final_training_checkpoint(manager):
    run = manager.runs/'20261124-005439-a051a4af'
    run.mkdir(parents=True)
    dashboard.atomic_json(run/'plan.json', dict(config=dict(kind='v2_train')))
    request = dict(kind='v2_evaluate', training_job=run.name)
    with pytest.raises(ValueError, match='1000步'):
        manager.prepare(request)
    checkpoint = run/'v2__test02__exp01__run001__step001000'
    checkpoint.mkdir()
    dashboard.atomic_json(checkpoint/'COMPLETE.json', dict(complete=True))
    dashboard.atomic_json(run/'v2_training.json', dict(passed=True,steps=1000))
    plan = manager.prepare(request)
    assert len(plan['stages']) == 2
    assert plan['config']['checkpoint'] == str(checkpoint)
    assert plan['config']['protocol'].endswith('evaluation_protocol-v003.json')
    assert plan['stages'][0]['argv'][2] == 'v2_groot_memoryvla.task40_evaluation'
    assert plan['stages'][0]['argv'][plan['stages'][0]['argv'].index('--protocol') + 1].endswith(
        'evaluation_protocol-v003.json')
    assert plan['stages'][1]['argv'][1].endswith('v2_task40_paired_evaluation.py')


def test_v2_evaluation_resume_runs_only_missing_pairs_from_same_protocol(manager):
    run = manager.runs/'20261124-005439-a051a4af'
    run.mkdir(parents=True)
    dashboard.atomic_json(run/'plan.json', dict(config=dict(kind='v2_train')))
    checkpoint = run/'v2__test02__exp01__run001__step001000'
    checkpoint.mkdir()
    dashboard.atomic_json(checkpoint/'COMPLETE.json', dict(complete=True))
    dashboard.atomic_json(run/'v2_training.json', dict(passed=True,steps=1000))
    previous = manager.runs/'20261124-031205-a32b5587'
    previous.mkdir(parents=True)
    dashboard.atomic_json(previous/'plan.json', dict(config=dict(kind='v2_evaluate',
        checkpoint=str(checkpoint), protocol_sha256='same')))
    dashboard.atomic_json(previous/'paired_status.json', dict(status='failed', results=[]))
    from v2_lock_evaluation import digest
    # Test manager reads the real locked v003 protocol from the repository.
    protocol = manager.root/'Output/test_02/experiment_01/evidence/evaluation_protocol-v003.json'
    previous_plan = dict(config=dict(kind='v2_evaluate',checkpoint=str(checkpoint),
        protocol_sha256=digest(protocol)))
    dashboard.atomic_json(previous/'plan.json', previous_plan)
    plan = manager.prepare(dict(kind='v2_evaluate',gpu=0,training_job=run.name,
        resume_evaluation_job=previous.name))
    assert len(plan['stages']) == 1
    argv = plan['stages'][0]['argv']
    assert '--resume-status' in argv and argv[argv.index('--resume-status')+1].endswith('paired_status.json')
    assert plan['stages'][0]['report'] == 'paired_evaluation.json'


def test_v2_smoke_cleanup_requires_terminal_owned_job_and_archived_evidence(manager, tmp_path):
    manager.root = tmp_path
    run_id = '20261123-220000-1234abcd'
    directory = manager.runs/run_id
    directory.mkdir(parents=True)
    plan = dict(title='V2 test', created_at='2026-11-23', config=dict(kind='v2_training_smoke'))
    dashboard.atomic_json(directory/'plan.json', plan)
    dashboard.atomic_json(directory/'state.json', dict(status='running'))
    with pytest.raises(ValueError, match='已结束'):
        manager.delete(dict(run=run_id))
    dashboard.atomic_json(directory/'state.json', dict(status='complete'))
    with pytest.raises(ValueError, match='归档'):
        manager.delete(dict(run=run_id))
    evidence = tmp_path/'Output/test_02/experiment_01/evidence'/f'4090-{run_id}.json'
    evidence.parent.mkdir(parents=True)
    dashboard.atomic_json(evidence, dict(job=dict(id=run_id), log='finished', plan=plan))
    assert manager.delete(dict(run=run_id))['status'] == 'deleted'
    assert not directory.exists() and evidence.exists() and not manager.snapshot()['jobs']


def fake_plan(manager, monkeypatch, *, passed=True, wait=False):
    script = ('import json,sys,time; from pathlib import Path; '
              'print(json.dumps(dict(step=1,loss=0.5)),flush=True); ')
    script += ('time.sleep(30); ' if wait else '')
    script += f'Path(sys.argv[1]).write_text(json.dumps(dict(passed={passed})))'
    stage = dict(name='fixture', argv=[sys.executable, '-u', '-c', script, '{run}/check.json'],
                 report='check.json', steps=1)
    plan = dict(title='test', config=dict(kind='action', gpu=0, notes='test'),
                stages=[stage], ready=True, required_free_mb=0)
    monkeypatch.setattr(manager, 'prepare', lambda request: plan.copy())


def wait_for(manager, run_id, predicate):
    deadline = time.monotonic()+10
    while time.monotonic() < deadline:
        value = manager.detail(dict(run=run_id))
        if predicate(value['job']):
            return value
        time.sleep(.025)
    pytest.fail('worker timed out: '+str(value))


def test_reports_and_completed_progress_survive_new_manager(manager, monkeypatch):
    fake_plan(manager, monkeypatch)
    value = manager.start({})
    finished = wait_for(manager, value['job']['id'], lambda j: j['status']=='complete')
    assert finished['job']['completed'] == 1
    assert finished['job']['step'] is None  # Do not double count a completed stage.
    restored = dashboard.BenchmarkManager(ROOT)
    restored.output, restored.runs = manager.output, manager.runs
    same = restored.detail(dict(run=value['job']['id']))
    assert same['reports'][0]['data']['passed'] is True
    assert '"loss": 0.5' in same['log']
    assert len(same['plan']['source_digest']) == 64


def test_exit_zero_with_failed_report_is_failed(manager, monkeypatch):
    fake_plan(manager, monkeypatch, passed=False)
    value = manager.start({})
    finished = wait_for(manager, value['job']['id'], lambda j: j['status']=='failed')
    assert finished['job']['returncode'] == 0
    assert finished['job']['completed'] == 0


def test_stop_only_own_worker_and_blocks_overlapping_checks(manager, monkeypatch):
    fake_plan(manager, monkeypatch, wait=True)
    value = manager.start({})
    run_id = value['job']['id']
    try:
        wait_for(manager, run_id, lambda j: j.get('step')==1)
        with pytest.raises(ValueError, match='正在运行'):
            manager.start({})
        assert manager.stop(dict(run=run_id))['status'] == 'stopping'
        finished = wait_for(manager, run_id, lambda j: j['status']=='stopped')
        assert not finished['reports']
    finally:
        if manager.detail(dict(run=run_id))['job']['status'] in ('starting', 'running', 'stopping'):
            manager.stop(dict(run=run_id))


def test_prior_reports_read_only_and_image_paths_scoped(manager):
    dashboard.atomic_json(manager.output/'libero_smoke.json', dict(passed=True, suite='libero_spatial', tasks=[]))
    value = manager.detail(dict(run='saved-libero_smoke'))
    assert value['job']['legacy'] and value['job']['kind']=='libero'
    with pytest.raises(ValueError):
        manager.stop(dict(run='saved-libero_smoke'))
    run_id = '20261112-120000-1234abcd'
    directory = manager.runs/run_id
    directory.mkdir(parents=True)
    dashboard.atomic_json(directory/'plan.json', {})
    frames = directory/'task-0-frames'
    frames.mkdir()
    (frames/'start-agentview.png').write_bytes(b'png')
    assert manager.image_path(dict(run=run_id, file='task-0-frames/start-agentview.png')).read_bytes()==b'png'
    for name in ('../../plan.json', 'task-0-frames/../../plan.json', 'plan.json'):
        with pytest.raises(ValueError):
            manager.image_path(dict(run=run_id, file=name))


def test_http_assets_options_preview_and_errors(manager):
    class Handler(viewer.Handler):
        benchmark_manager = manager
    server = viewer.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        code = urlopen(url+'/benchmarks.js').read().decode()
        template = json.JSONDecoder().raw_decode(code[len('window.__benchmarkHTML = '):])[0]
        assert template == (ROOT/'evaluation/scripts/benchmark_workbench.html').read_text()
        assert len(json.load(urlopen(url+'/api/benchmarks/options'))['suites']) == 5
        request = Request(url+'/api/benchmarks/preview', data=b'{"kind":"libero","tasks":[0]}',
                          headers={'Content-Type': 'application/json'})
        assert json.load(urlopen(request))['config']['kind']=='libero'
        request = Request(url+'/api/benchmarks/preview', data=b'{"kind":"v2_training_smoke","steps":3}',
                          headers={'Content-Type': 'application/json'})
        assert json.load(urlopen(request))['config']['labels'] == 'artificial test values, NOT physical truth'
        request = Request(url+'/api/benchmarks/start', data=b'{"kind":"unknown"}',
                          headers={'Content-Type': 'application/json'})
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code==400 and json.load(error.value)['error']
        with pytest.raises(HTTPError) as error:
            urlopen(url+'/api/benchmarks/image?run=../../etc&file=passwd')
        assert error.value.code==404
    finally:
        server.shutdown();server.server_close();thread.join()


def test_metric_renderer_does_not_turn_environment_smoke_into_policy_success():
    node = shutil.which('node') or '/home/1024201092WYH/.nvm/versions/node/v22.23.1/bin/node'
    if not Path(node).exists():
        pytest.skip('Node required')
    script = """
const assert = require('node:assert/strict');
const {benchmarkMetrics} = require('./evaluation/scripts/benchmark_workbench.js');
const html = benchmarkMetrics([{data:{suite:'libero_goal', initial_state_index:0, seed:42,
  tasks:[{task:0,language:'<script>alert(1)</script>',steps:20,success:true,image_shapes:{camera:[128,128,3]}}]}}]);
assert.ok(html.includes('不统计策略成功率'));
assert.ok(!html.includes('100%'));
assert.ok(!html.includes('<script>'));
assert.ok(html.includes('&lt;script&gt;'));
const smoke = benchmarkMetrics([{data:{category:'engineering_smoke',labels:'TEST',steps:3,
  peak_cuda_gib:12,frozen_action_head_unchanged:true,reload_max_abs:{loss:0}}}]);
assert.ok(smoke.includes('人工测试值'));
assert.ok(smoke.includes('重载最大差异'));
"""
    result = subprocess.run([node], input=script, text=True, cwd=ROOT, capture_output=True)
    assert result.returncode==0, result.stderr


@pytest.fixture
def libero_options(manager, tmp_path, monkeypatch):
    info=manager.options()
    base=tmp_path/'base';base.mkdir();(base/'config.json').write_text('{}')
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir();(checkpoint/'config.json').write_text('{}')
    saved=dict(path=str(checkpoint),resumable=True,suite='libero_spatial',tasks=[0],arm='memory',
               fine_tuning_mode='action_full',group_size=16,seed=42,learning_rate=1e-5,step=300)
    info['libero'].update(base_checkpoint=str(base),checkpoints=[saved],
                         datasets=[dict(suite='libero_spatial',task=0,ready=True,episodes=50)])
    monkeypatch.setattr(manager,'options',lambda:info)
    return info


def test_libero_dataset_does_not_require_gpu(manager, libero_options):
    libero_options['gpus']=[]
    plan=manager.prepare(dict(kind='libero_dataset',suite='libero_spatial',tasks=[2,0]))
    assert plan['ready'] and plan['required_free_mb']==0 and plan['config']['tasks']==[0,2]
    assert plan['stages'][0]['argv'][2].endswith('.prepare')


def test_libero_training_requires_own_data_and_resume_contract(manager, libero_options):
    with pytest.raises(ValueError,match='准备示教'):
        manager.prepare(dict(kind='libero_train',tasks=[1]))
    plan=manager.prepare(dict(kind='libero_train',tasks=[0],fine_tuning_mode='lora',arm='baseline'))
    assert plan['config']['checkpoint']==libero_options['libero']['base_checkpoint']
    assert plan['required_free_mb']==16000
    saved=libero_options['libero']['checkpoints'][0]
    payload=dict(kind='libero_train',tasks=[0],checkpoint=saved['path'],steps=301)
    assert manager.prepare(payload)['config']['steps']==301
    for extra in [dict(steps=300),dict(arm='baseline'),dict(group_size=4),dict(seed=7),dict(learning_rate=1e-4)]:
        with pytest.raises(ValueError):manager.prepare(dict(payload,**extra))


def test_libero_evaluation_rejects_b1k_checkpoint_and_keeps_protocol(manager, libero_options, monkeypatch):
    monkeypatch.setattr(shutil,'which',lambda name:'/usr/bin/ffmpeg')
    with pytest.raises(ValueError,match='先训练'):
        manager.prepare(dict(kind='libero_eval',checkpoint=libero_options['libero']['base_checkpoint']))
    checkpoint=libero_options['libero']['checkpoints'][0]['path']
    plan=manager.prepare(dict(kind='libero_eval',checkpoint=checkpoint,max_steps=0,trials=50))
    assert plan['config']['max_steps']==0 and plan['config']['trials']==50
    assert plan['stages'][0]['argv'][2].endswith('.evaluate')
    for extra in [dict(max_steps=221),dict(max_steps=-1),dict(trials=51),dict(trials=0)]:
        with pytest.raises(ValueError):manager.prepare(dict(kind='libero_eval',checkpoint=checkpoint,**extra))


def test_libero_video_and_frames_are_scoped_to_one_job(manager):
    run_id='20261112-120000-1234abcd';directory=manager.runs/run_id
    media=directory/'task-0-episode-1';media.mkdir(parents=True)
    dashboard.atomic_json(directory/'plan.json',{})
    (media/'rollout.mp4').write_bytes(b'video');(media/'end-agentview.png').write_bytes(b'png')
    assert manager.video_path(dict(run=run_id,file='task-0-episode-1/rollout.mp4'))==media/'rollout.mp4'
    assert manager.image_path(dict(run=run_id,file='task-0-episode-1/end-agentview.png'))==media/'end-agentview.png'
    for name in ('../rollout.mp4','task-0-episode-1/../../rollout.mp4','plan.json'):
        with pytest.raises(ValueError):manager.video_path(dict(run=run_id,file=name))
