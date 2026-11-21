"""Device routing and source snapshots; never start GPU work in unit checks."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import shutil
import subprocess
import threading
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'train'),str(ROOT/'evaluation/scripts')]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_source_manifest_includes_all_versions_eval_and_exact_vendor(tmp_path):
    from dashboard import TrainingManager
    paths=['src/v1_groot_memoryvla/a.py','src/v2_groot_memoryvla/b.py','src/v3_bodymemory/c.py',
           'evaluation/scripts/entry.py','scripts/entry.py','train/tool.py','configs/setting.json']
    for name in paths:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('source')
    vendor=tmp_path.parent/(tmp_path.name+'_vendor');vendor.mkdir();(vendor/'simulator.py').write_text('same physics')
    (tmp_path/'configs/workbench_sources.json').write_text(json.dumps({'vendor/OmniGibson/omnigibson':str(vendor)}))
    (tmp_path/'scripts/alias.py').symlink_to('entry.py')
    outputs=tmp_path/'evaluation/V_1/memory/run/results.json';outputs.parent.mkdir(parents=True);outputs.write_text('results')
    env=tmp_path/'src/v1_groot_memoryvla/.venv/lib.py';env.parent.mkdir();env.write_text('large')
    manager=TrainingManager(tmp_path)
    try:
        manifest=manager.manifest()
        assert set(paths)<=manifest['files'].keys()
        assert 'vendor/OmniGibson/omnigibson/simulator.py' in manifest['files']
        assert manifest['files']['scripts/alias.py']==manifest['files']['scripts/entry.py']
        assert not any('.venv' in name or name.endswith('results.json') for name in manifest['files'])
        (vendor/'simulator.py').write_text('changed on 4090')
        assert manifest['digest']!=manager.manifest()['digest']
    finally:manager.close()


def test_eval_device_rejects_nonexistent_gpu_before_ssh(tmp_path):
    from groot_device_eval import DeviceEvaluationManager
    manager=object.__new__(DeviceEvaluationManager)
    for request in [dict(mode='dual1'), dict(mode='single2', gpu_index=1),
                    dict(task40_collect=True, sim_gpu=1)]:
        with pytest.raises(ValueError,match='GPU0'):manager._remote('start',request)


def test_sync_preserves_launcher_links_and_materializes_vendor(tmp_path,monkeypatch):
    from dashboard import TrainingManager
    root=tmp_path/'source';live=tmp_path/'live'
    (root/'scripts').mkdir(parents=True)
    (root/'evaluation/scripts').mkdir(parents=True)
    (root/'configs').mkdir()
    (root/'evaluation/scripts/tasks.py').write_text('VALUE="canonical"\n')
    (root/'evaluation/scripts/entry.py').write_text('from tasks import VALUE\nprint(VALUE)\n')
    (root/'scripts/entry.py').symlink_to('../evaluation/scripts/entry.py')
    external=tmp_path/'external';external.mkdir()
    (external/'model.py').write_text('source')
    (external/'alias.py').symlink_to('model.py')
    icon=tmp_path/'icon.png';icon.write_bytes(b'icon')
    (root/'configs/workbench_sources.json').write_text(json.dumps({'vendor/code':str(external),'vendor/docs/icon.png':str(icon)}))
    manager=TrainingManager(root);manager.live=str(live)
    real_run=subprocess.run
    def local_run(argv,**kwargs):
        if argv[0]=='rsync':
            argv=list(argv);option=argv.index('-e');del argv[option:option+2]
            argv[-1]=argv[-1].split(':',1)[1]
        return real_run(argv,**kwargs)
    def local_ssh(command,code=None,timeout=30):
        return real_run(command,shell=True,input=code,capture_output=True,text=True,check=True,timeout=timeout).stdout
    monkeypatch.setattr(subprocess,'run',local_run)
    monkeypatch.setattr(manager,'_ssh',local_ssh)
    try:
        manager._sync(force=True)
        assert (live/'scripts/entry.py').is_symlink()
        assert not (live/'vendor/code/alias.py').is_symlink()
        assert (live/'vendor/docs/icon.png').read_bytes()==b'icon'
        snapshot=tmp_path/'snapshot';shutil.copytree(live,snapshot,symlinks=True)
        result=real_run([sys.executable,str(snapshot/'scripts/entry.py')],capture_output=True,text=True,check=True)
        assert result.stdout.strip()=='canonical'
    finally:manager.close()


def test_http_selects_device_before_options_and_launch():
    viewer=load('device_routing_viewer','evaluation/scripts/view_results.py')
    calls=[]
    remote=SimpleNamespace(options=lambda:dict(gpus=[dict(index=0)],ready=True),
                           prepare=lambda value:calls.append(value) or dict(plan=value))
    class Handler(viewer.Handler):
        launch_manager=SimpleNamespace()
        evaluation_devices={'a6000':remote}
    server=viewer.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        base=f'http://127.0.0.1:{server.server_port}'
        assert json.load(urlopen(base+'/api/launch/options?device=a6000'))['gpus']==[dict(index=0)]
        request=Request(base+'/api/launch/preview',data=json.dumps(dict(device='a6000',gpu_index=0)).encode(),headers={'Content-Type':'application/json'})
        assert json.load(urlopen(request))['plan']==dict(gpu_index=0)
        assert calls==[dict(gpu_index=0)]
    finally:server.shutdown();server.server_close();thread.join()


def test_remote_evaluation_paths_are_device_local_and_preserve_action_protocol(tmp_path,monkeypatch):
    import groot_web_launch
    monkeypatch.setenv('GROOT_EVAL_ROOT',str(tmp_path/'results'))
    monkeypatch.setenv('GROOT_CACHE_ROOT',str(tmp_path/'cache'))
    monkeypatch.setenv('GROOT_DEVICE_ID','a6000')
    manager=groot_web_launch.LaunchManager(ROOT)
    result=manager.prepare(dict(mode='single1',gpu_index=0,task=0,preset='fast-debug',debug_max_steps=16,seed=0,record_video=False))
    plan=result['plan']
    assert Path(plan['output']).is_relative_to(tmp_path/'results')
    assert plan['device']=='a6000' and plan['action_chunk']==8 and plan['base_seed']==0
    assert plan['instances']=={'0':[0]} and plan['runner_max_episode_steps']==16
    command=plan['commands'][0]
    assert command[command.index('--appdata-path')+1].startswith(str(tmp_path/'cache'))


def test_optional_inference_uses_device_paths_and_ip_not_ssh_username():
    from groot_inference_launch import InferenceManager
    device=json.loads((ROOT/'configs/workbench_devices.example.json').read_text())['a6000']
    manager=InferenceManager(ROOT,device['host'],device)
    assert manager.bind_host=='192.0.2.30'
    assert manager.default_python==device['paths']['python']
    assert manager.default_source.startswith('/data/projects/memory_vla/')
    argv=manager.argv(dict(checkpoint='/data/test/checkpoint',gpu_index=0,port=5559))
    assert argv[argv.index('--host')+1]=='192.0.2.30'
    assert argv[argv.index('--python-path')+1]==device['paths']['python']
    assert 'DEVICE_PATHS=' in manager.remote


def test_task40_remote_rows_localize_under_exp09(tmp_path, monkeypatch):
    from groot_device_eval import DeviceEvaluationManager

    manager = object.__new__(DeviceEvaluationManager)
    manager.root = tmp_path/'local'
    manager.project = str(tmp_path/'remote')
    manager.device_id = 'a6000'
    manager.device = {'evaluation': {'environment': {}}}
    manager.host = 'mock-a6000'
    relative = Path('Output/test_01/experiment_09')
    remote = Path(manager.project)/relative/'runs/123abc456def'
    local = manager.root/relative/'runs'/remote.name
    row = dict(id='0123456789ab', kind='task40_collect', folder=None, output=str(remote),
               output_exists=False, state='running', artifacts=[], progress={'stage': 'initializing'},
               sim_log_tail='', plan={'output': str(remote)})
    calls = []
    real_run = subprocess.run

    def sync(argv, **kwargs):
        calls.append(argv)
        argv = list(argv)
        index = argv.index('-e')
        del argv[index:index+2]
        argv[-2] = argv[-2].split(':', 1)[1]
        return real_run(argv, **kwargs)

    monkeypatch.delenv('GROOT_TASK40_EXPERIMENT_ROOT', raising=False)
    monkeypatch.setattr(subprocess, 'run', sync)
    pending = manager._localize(row)
    assert pending['progress'] == {'stage': 'initializing'} and not calls

    (remote/'logs').mkdir(parents=True)
    (remote/'logs/simulator.log').write_text('running')
    (remote/'appdata').mkdir()
    (remote/'appdata/runtime-cache').write_text('cache')
    (remote/'job.json').write_text('{}')
    (remote/'results.json').write_text('{}')
    row.update(output_exists=True, state='complete', result_ready=True,
               artifacts=[{'path': 'job.json', 'url': str(remote/'job.json')}],
               progress={'stage': 'finished'}, sim_log_tail='running')
    result = manager._localize(row)
    assert result['output'] == result['plan']['output'] == str(local)
    assert result['device'] == 'a6000' and result['remote_output'] == str(remote)
    assert result['folder'] is None and result['result_ready']
    assert result['progress'] == row['progress'] and result['sim_log_tail'] == 'running'
    assert {item['url'] for item in result['artifacts']} == {
        '/api/launch/artifact?run=123abc456def&path=job.json',
        '/api/launch/artifact?run=123abc456def&path=results.json',
        '/api/launch/artifact?run=123abc456def&path=logs%2Fsimulator.log',
    }
    assert not (local/'appdata').exists()


def test_task40_remote_request_uses_collection_gpu_field(monkeypatch):
    from groot_device_eval import DeviceEvaluationManager

    manager = object.__new__(DeviceEvaluationManager)
    manager.root = ROOT
    manager.project = '/remote/memory_vla'
    manager.host = 'mock-a6000'
    manager.device = {'evaluation': {'environment': {'GROOT_DEVICE_ID': 'a6000'}}}
    identity = {'task40_diagnostics': True, 'port': 5690}

    def resolve(host, port):
        assert (host, port) == ('172.28.62.17', 5690)
        return identity

    manager.policy_resolver = resolve
    dependencies = {'digest': 'test-digest'}

    def ssh(argv, *, input, **kwargs):
        assert argv[-2:] == ['mock-a6000', 'python3 -']
        print_call = ast.parse(input.splitlines()[-1]).body[0].value
        dispatch_call = print_call.args[0].args[0]
        request = ast.literal_eval(dispatch_call.args[0])
        assert request['action'] == 'prepare'
        assert request['payload'] == {'task40_collect': True, 'sim_gpu': 0, 'policy_port': 5690}
        assert request['policy_identity'] == identity
        assert request['task40_dependencies'] == dependencies
        return subprocess.CompletedProcess(argv, 0, stdout='{"plan": {}}')

    monkeypatch.setattr(subprocess, 'run', ssh)
    assert manager._remote('prepare', {'task40_collect': True, 'policy_port': 5690}, dependencies) == {'plan': {}}


def test_task40_dependency_sync_and_dispatch(tmp_path, monkeypatch):
    from groot_device_eval import DeviceEvaluationManager, TASK40_ANCHOR, TASK40_RUNNER, TASK40_WEIGHTS
    import groot_device_eval_remote as remote
    import groot_web_launch

    root = tmp_path/'local'
    project = tmp_path/'remote'
    live = project/'deployments/web_live'
    live.mkdir(parents=True)
    (live/'web_source_manifest.json').write_text(json.dumps({'digest': 'source-test'}))
    (live/'Output').mkdir()
    (live/'Output/unrelated-output').write_text('not source')
    (project/'setup').mkdir()
    readiness = project/'setup/simulation_ready.json'
    readiness.write_text(json.dumps({'task_ids': [40]}))

    expected = [TASK40_RUNNER/'paired_sim_diagnostic.py']
    expected += [TASK40_ANCHOR/name for name in (
        'start.json', 'anchor.json', 'anchor_live_observation.npz', 'sim_checkpoint.pt',
        'prefix/0008/decision.json')]
    expected += [TASK40_WEIGHTS/name for name in (
        'config.json', 'embodiment_id.json', 'processor_config.json', 'statistics.json')]
    unrelated = [TASK40_RUNNER/'another_runner.py', TASK40_ANCHOR/'prefix/0008/observation.npz',
                 TASK40_WEIGHTS/'model.safetensors']
    for relative in expected + unrelated:
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative.as_posix())

    manager = object.__new__(DeviceEvaluationManager)
    manager.root = root
    manager.project = str(project)
    manager.host = 'mock-a6000'
    real_run = subprocess.run
    manager.training = SimpleNamespace(_ssh=lambda command: real_run(command, shell=True, check=True))

    def sync(argv, **kwargs):
        argv = list(argv)
        index = argv.index('-e')
        del argv[index:index+2]
        argv[-1] = argv[-1].split(':', 1)[1]
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, 'run', sync)
    dependencies = manager._sync_task40_dependencies()
    staged = project/'setup/task40_dependencies'/dependencies['digest']
    assert {path.relative_to(staged) for path in staged.rglob('*') if path.is_file()} == set(expected)

    experiment_root = project/'Output/test_01/experiment_09'
    output = experiment_root/'runs/abcdef012345'
    calls = []

    class TestLaunchManager:
        def __init__(self, source, log_root=None):
            self.source = Path(source)
            self.jobs = []

        def prepare(self, payload):
            calls.append(('prepare', self.source))
            assert payload == {'task40_collect': True, 'sim_gpu': 0, 'policy_port': 5690}
            assert self.policy_resolver('172.28.62.17', 5690)['task40_diagnostics']
            assert Path(os.environ['GROOT_TASK40_EXPERIMENT_ROOT']) == experiment_root
            assert Path(os.environ['GROOT_TASK40_RUNNER']) == staged/TASK40_RUNNER
            assert Path(os.environ['GROOT_TASK40_ANCHOR']) == staged/TASK40_ANCHOR
            assert Path(os.environ['GROOT_TASK40_WEIGHTS']) == staged/TASK40_WEIGHTS
            assert Path(os.environ['GROOT_REPO']) == self.source/'vendor/Isaac-GR00T'
            assert os.environ['B1K_PYTHON'] == '/envs/b1k/bin/python'
            assert os.environ['BEHAVIOR_DATA'] == '/datasets/simulation'
            return {'plan': {'kind': 'task40_collect', 'output': str(output)}}

        def start(self, payload):
            plan = self.prepare(payload)['plan']
            self.jobs.append({'plan': plan})
            return dict(id='987654321abc', pid=os.getpid(), kind=plan['kind'], gpus=['0'],
                        output=plan['output'], folder=None, log=str(experiment_root/'logs/test.log'))

        def _status(self, job):
            assert job['process'].poll() is None
            return dict(job, process=None, gpus=['0'], log_path=None,
                        artifacts=[], progress={'stage': 'collect'}, sim_log_tail='simulator')

        def stop_restore(self, payload):
            assert self.jobs[0]['process'].poll() is None
            return {'job': payload['job'], 'state': 'stopping'}

    monkeypatch.setattr(remote.os, 'environ', os.environ.copy())
    monkeypatch.setattr(groot_web_launch, 'LaunchManager', TestLaunchManager)
    request = dict(project=str(project),
                   environment=dict(GROOT_EVAL_ROOT=str(project/'evaluation'),
                                    B1K_PYTHON='/envs/b1k/bin/python', BEHAVIOR_DATA='/datasets/simulation'),
                   task40_dependencies=dependencies, policy_identity={'task40_diagnostics': True},
                   payload={'task40_collect': True, 'sim_gpu': 0, 'policy_port': 5690})

    assert remote.dispatch(dict(request, action='prepare'))['plan']['output'] == str(output)
    result = remote.dispatch(dict(request, action='start'))
    source = project/'deployments/eval_source-test'
    assert calls == [('prepare', live), ('prepare', source)]
    assert result['source'] == str(source) and result['source_digest'] == 'source-test'
    assert result['task40_dependency_digest'] == dependencies['digest']
    assert not (source/'Output').exists() and not result['output_exists']
    output.mkdir(parents=True)
    [status] = remote.dispatch(dict(request, action='snapshot', payload={}))
    assert status['output_exists'] and status['progress'] == {'stage': 'collect'}
    assert status['sim_log_tail'] == 'simulator' and status['plan'] == result['plan']
    assert remote.dispatch(dict(request, action='stop_restore', payload={'job': result['id']}))['state'] == 'stopping'

    (staged/TASK40_RUNNER/'paired_sim_diagnostic.py').write_text('altered')
    with pytest.raises(ValueError, match='校验失败'):
        remote.dispatch(dict(request, action='prepare'))
    readiness.write_text(json.dumps({'task_ids': [21]}))
    with pytest.raises(ValueError, match='Task40 仿真环境尚未验收'):
        remote.dispatch(dict(request, action='start'))
