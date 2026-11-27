"""Validate remote policy startup without loading models or using a GPU."""
import json
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evaluation/scripts'))
from groot_inference_launch import InferenceManager, STATUS, REMOTE
import groot_inference_remote as remote_control


def payload(**updates):
    return dict(checkpoint='/data/model checkpoint',gpu_index=2,port=5597,**updates)


@pytest.mark.parametrize('changes',[
    {'checkpoint':'relative/path'}, {'gpu_index':True}, {'port':0},
    {'source_dir':'/data/bad\npath'}, {'seed':-1}, {'command':'touch bad'},
])
def test_invalid_inference_parameters_are_rejected(changes):
    request=payload();request.update(changes)
    with pytest.raises(ValueError):
        InferenceManager(ROOT).argv(request)


def test_remote_launcher_uses_canonical_script_and_quotes_paths(monkeypatch):
    manager=InferenceManager(ROOT)
    calls=[]
    def remote(command,code):
        calls.append((shlex.split(command),code))
        return dict(checkpoint='/data/model checkpoint',pid=11,port=5597,state='loading')
    monkeypatch.setattr(manager,'_remote',remote)
    manager.prepare(payload());manager.start(payload())
    assert '--check-only' in calls[0][0]
    assert '--check-only' not in calls[1][0]
    assert '/data/model checkpoint' in calls[0][0]
    assert calls[0][1]==(ROOT/'evaluation/scripts/run_groot_server.sh').read_text()
    assert manager.jobs[0]['host']=='172.28.62.17'


def test_history_controls_are_forwarded_and_validated():
    manager=InferenceManager(ROOT)
    args=manager.argv(payload(history_mode='other_episode'))
    assert args[args.index('--history-mode')+1]=='other_episode'
    with pytest.raises(ValueError):
        manager.argv(payload(history_mode='random'))


def test_task40_experiment_10_service_archives_outside_exp09():
    manager = InferenceManager(ROOT)
    args = manager.argv(payload(task40_diagnostics=True, experiment='experiment_10'))
    assert '--task40-diagnostics' in args
    assert args[args.index('--log-dir') + 1].endswith('/experiment_10/policy_services')
    args = manager.argv(payload(task40_diagnostics=True, experiment='experiment_11'))
    assert args[args.index('--log-dir') + 1].endswith('/experiment_11/policy_services')
    with pytest.raises(ValueError, match='Task40'):
        manager.argv(payload(task40_diagnostics=True, experiment='experiment_09'))
    with pytest.raises(ValueError, match='Task40'):
        manager.argv(payload(experiment='experiment_10'))


def local_args(tmp_path,port):
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    (checkpoint/'config.json').write_text('{}')
    (checkpoint/'model.safetensors').write_bytes(b'fake weights')
    source=tmp_path/'source/src/v1_groot_memoryvla';source.mkdir(parents=True)
    (source/'serve.py').write_text('')
    args=['bash',str(ROOT/'evaluation/scripts/run_groot_server.sh'),'--json',
          '--checkpoint',str(checkpoint),'--source-dir',str(tmp_path/'source'),
          '--python-path','/usr/bin/python3','--gpu-index','2','--host','127.0.0.1',
          '--port',str(port),'--log-dir',str(tmp_path/'logs')]
    return args


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0))
        return probe.getsockname()[1]


def test_check_only_checks_real_files_and_busy_port_without_creating_logs(tmp_path):
    port=free_port();args=local_args(tmp_path,port)
    result=subprocess.run(args+['--check-only'],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['state']=='checked'
    assert not (tmp_path/'logs').exists()
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1',port));occupied.listen()
        result=subprocess.run(args+['--check-only'],text=True,capture_output=True)
    assert result.returncode!=0 and '端口不可用' in result.stderr


def test_training_snapshot_can_use_shared_task40_archive(tmp_path):
    args = local_args(tmp_path, free_port())
    project = tmp_path / 'project'
    archive = project / 'Output/test_01/experiment_07/scripts/policy_server_diagnostic.py'
    archive.parent.mkdir(parents=True)
    archive.write_text('')
    instrumentation = tmp_path / 'source/evaluation/scripts/task40_memory_instrumentation.py'
    instrumentation.parent.mkdir(parents=True)
    instrumentation.write_text('')
    script = Path(args[1]).read_text().replace(
        "PROJECT = Path('/nvme3/projects/memory_vla')", f"PROJECT = Path({str(project)!r})", 1)
    result = subprocess.run(['bash', '-s', '--', *args[2:], '--task40-diagnostics', '--check-only'],
                            input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    command = shlex.split(json.loads(result.stdout)['command'])
    assert str(instrumentation.parent) in command
    assert str(archive) in command
    assert command[command.index('--anchor') + 1] == str(project / 'Output/test_01/experiment_01/checkpoints')


@pytest.mark.parametrize('indexed', [False, True])
def test_incomplete_checkpoint_cannot_start_policy(tmp_path, indexed):
    args=local_args(tmp_path,free_port())
    checkpoint=tmp_path/'checkpoint'
    (checkpoint/'model.safetensors').unlink()
    if indexed:
        (checkpoint/'model.safetensors.index.json').write_text(json.dumps({
            'weight_map': {'layer.weight': 'model-00001-of-00002.safetensors'}}))
    result=subprocess.run(args+['--check-only'],text=True,capture_output=True)
    assert result.returncode!=0
    assert '权重尚未保存完整' in result.stderr


def test_loading_process_blocks_duplicate_port_and_status_detects_exit(tmp_path):
    port=free_port();args=local_args(tmp_path,port)
    checkpoint=str(tmp_path/'checkpoint')
    child=subprocess.Popen(['/usr/bin/python3','-c','import time;time.sleep(20)',
        '-m','v1_groot_memoryvla.serve','--model-path',checkpoint,'--port',str(port)],
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        result=subprocess.run(args+['--check-only'],text=True,capture_output=True)
        assert result.returncode!=0 and '可能正在加载' in result.stderr
        job=dict(pid=child.pid,port=port,checkpoint=checkpoint,log=str(tmp_path/'missing.log'),state='loading')
        code='JOBS = '+repr([job])+'\n'+STATUS
        snapshot=subprocess.check_output(['/usr/bin/python3','-c',code],text=True)
        assert json.loads(snapshot)['jobs'][0]['state']=='loading'
        child.terminate();child.wait(timeout=3)
        snapshot=subprocess.check_output(['/usr/bin/python3','-c',code],text=True)
        assert json.loads(snapshot)['jobs'][0]['state']=='failed'
        job['state']='stopped'
        snapshot=subprocess.check_output(['/usr/bin/python3','-c','JOBS = '+repr([job])+'\n'+STATUS],text=True)
        assert json.loads(snapshot)['jobs'][0]['state']=='stopped'
    finally:
        if child.poll() is None:
            child.terminate();child.wait(timeout=3)


def test_actual_background_submission_with_fake_python_only(tmp_path):
    args=local_args(tmp_path,free_port())
    fake=tmp_path/'python-fake'
    fake.write_text('#!/bin/sh\nexec /usr/bin/python3 -c "import time; print(123,flush=True); time.sleep(20)" "$@"\n')
    fake.chmod(0o755)
    args[args.index('--python-path')+1]=str(fake)
    result=subprocess.run(args,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    job=json.loads(result.stdout)
    try:
        assert job['state']=='loading' and job['pid']>0
        assert Path(job['log']).is_file()
        assert 'v1_groot_memoryvla.serve' in job['command']
    finally:
        import os
        os.kill(job['pid'],signal.SIGTERM)


def test_live_source_is_frozen_before_policy_process_launch(tmp_path):
    args=local_args(tmp_path,free_port())
    live=tmp_path/'source';live.rename(tmp_path/'web_live');live=tmp_path/'web_live'
    (live/'web_source_manifest.json').write_text(json.dumps(dict(digest='test-digest')))
    args[args.index('--source-dir')+1]=str(live)
    fake=tmp_path/'python-fake'
    fake.write_text('#!/bin/sh\nexec /usr/bin/python3 -c "import time;time.sleep(20)"\n')
    fake.chmod(0o755);args[args.index('--python-path')+1]=str(fake)
    result=subprocess.run(args,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    job=json.loads(result.stdout)
    try:
        frozen=Path(job['source_dir'])
        assert frozen!=live and frozen.name=='inference_test-digest'
        assert (frozen/'src/v1_groot_memoryvla/serve.py').is_file()
    finally:
        import os
        os.kill(job['pid'],signal.SIGTERM)


def test_http_routes_to_inference_manager_without_touching_evaluation(monkeypatch):
    import threading
    from urllib.request import Request,urlopen
    import view_results
    manager=InferenceManager(ROOT)
    calls=[]
    def remote(command,code):
        calls.append(command)
        if "dispatch('stop'," in code:
            return dict(pid=15,port=5597,checkpoint='/data/model checkpoint',start_ticks=123,state='stopping')
        if command=='python3 -':
            return dict(checkpoints=[],sources=[],gpus=[])
        return dict(pid=15,port=5597,state='checked' if '--check-only' in command else 'loading')
    monkeypatch.setattr(manager,'_remote',remote)
    class Handler(view_results.Handler):
        inference_manager=manager
        launch_manager=None
    server=view_results.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    url=f'http://127.0.0.1:{server.server_port}/api/inference/'
    try:
        with urlopen(url+'options',timeout=3) as response:
            assert json.load(response)['host']=='172.28.62.17'
        request=Request(url+'preview',data=json.dumps(payload()).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=3) as response:
            assert json.load(response)['state']=='checked'
        assert not manager.jobs
        request=Request(url+'start',data=json.dumps(payload()).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=3) as response:
            assert response.status==202 and json.load(response)['pid']==15
        assert len(manager.jobs)==1
        request=Request(url+'stop',data=json.dumps(dict(pid=15,port=5597,checkpoint='/data/model checkpoint',start_ticks=123)).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=3) as response:
            assert response.status==202 and json.load(response)['state']=='stopping'
    finally:
        server.shutdown();server.server_close();worker.join(timeout=3)


def test_checkpoint_version_and_source_match_saved_training_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_control,'TRAINING',tmp_path)
    root=tmp_path/'V_2_mem_task22'
    checkpoint=root/'memory_v1/checkpoints/step-003000'
    config=root/'config'
    config.mkdir(parents=True)
    (config/'web_launch.json').write_text(json.dumps(dict(version='V_2',source='/snapshot/V_2')))
    info=remote_control.checkpoint_info(str(checkpoint))
    assert info['version']=='V_2' and info['source_dir']=='/snapshot/V_2'
    assert info['label']=='V_2_memory_task22_train01 / step-003000'
    legacy=tmp_path/'old_run'
    (legacy/'config').mkdir(parents=True)
    (legacy/'config/data_split.json').write_text('{"task":21}')
    (legacy/'config/remote_locations.json').write_text('{"baseline_controller_source":"/baseline","memory_source":"/memory"}')
    info=remote_control.checkpoint_info(str(legacy/'baseline/checkpoints/step-001000'))
    assert info['version']=='V_1' and info['run']=='V_1_baseline_task21_train01' and info['source_dir']=='/baseline'
    info=remote_control.checkpoint_info('/outside/custom/checkpoints/step-001000')
    assert info['version']=='unversioned'
    run = 'v1__test01__exp12__run001'
    root = tmp_path/run
    (root/'config').mkdir(parents=True)
    (root/'config/web_launch.json').write_text(json.dumps(dict(source='/snapshot/exp12', version='V_1')))
    info = remote_control.checkpoint_info(str(root/'memory_v1/checkpoints'/f'{run}__step002000'))
    assert (info['version'], info['run'], info['training_id'], info['source_dir']) == (
        'V_1', run, run, '/snapshot/exp12')


@pytest.mark.parametrize('run',[
    'v1__test01__exp13__run001',
    'v2_1__test02__exp01__run001',
    'v2.1__test01__exp14__run001',
])
def test_checkpoint_catalog_preserves_run_id_and_canonical_version(tmp_path,monkeypatch,run):
    monkeypatch.setattr(remote_control,'TRAINING',tmp_path)
    root=tmp_path/run
    checkpoint=root/('memory_v1' if run.startswith('v1__') else 'run')/'checkpoints'/f'{run}__step002000'
    checkpoint.mkdir(parents=True)
    info=remote_control.checkpoint_info(str(checkpoint))
    assert info['run']==info['training_id']==run
    assert info['version']==('V_1' if run.startswith('v1__') else 'V_2_1')
    assert info['label']==run+' / '+checkpoint.name


def test_discovered_policy_stop_checks_identity_and_only_signals_selected_process(tmp_path):
    def stop_process(request):
        # Production SSH uses the A100 system Python. The simulation env's
        # Python was built without pidfd support, so use system Python here too.
        code=REMOTE
        result=subprocess.run(['/usr/bin/python3','-c',code+"\ndispatch('stop', "+repr(request)+")"],
                              text=True,capture_output=True)
        if result.returncode:
            raise ValueError(result.stderr)
        return json.loads(result.stdout)
    checkpoint=str(tmp_path/'fake-policy')
    child=subprocess.Popen(['/usr/bin/python3','-c','import time;time.sleep(30)',
        '-m','v1_groot_memoryvla.serve','--model-path',checkpoint,'--port','5599'],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    unrelated=subprocess.Popen(['/usr/bin/python3','-c','import time;time.sleep(30)'])
    try:
        row=remote_control.process_info(child.pid)
        assert row['checkpoint']==checkpoint and row['start_ticks']>0
        records=remote_control.snapshot([])['jobs']
        assert any(job['pid']==child.pid for job in records)
        assert all(job['pid']!=unrelated.pid for job in records)
        target={key:row[key] for key in ('pid','port','checkpoint','start_ticks')}
        with pytest.raises(ValueError,match='身份'):
            stop_process(dict(target,port=5598))
        with pytest.raises(ValueError,match='身份'):
            stop_process(dict(target,start_ticks=row['start_ticks']+1))
        assert child.poll() is None
        assert stop_process(target)['state']=='stopping'
        assert child.wait(timeout=3)==-signal.SIGTERM
        assert unrelated.poll() is None
        assert remote_control.snapshot([dict(row,state='stopping')])['jobs'][0]['state']=='stopped'
    finally:
        for process in [child,unrelated]:
            if process.poll() is None:
                process.terminate();process.wait(timeout=3)


@pytest.mark.parametrize('stop_request',[{},dict(pid=True,port=5556,checkpoint='/a',start_ticks=1),
    dict(pid=1,port=5556,checkpoint='/a'),dict(pid=1,port=5556,checkpoint='a',start_ticks=1)])
def test_stop_rejects_incomplete_process_identity(stop_request):
    with pytest.raises(ValueError):
        InferenceManager(ROOT).stop(stop_request)


def test_resolver_uses_actual_ready_process_and_canonical_checkpoint(monkeypatch):
    manager=InferenceManager(ROOT)
    job=dict(pid=12,port=5560,state='ready',checkpoint='/old/alias',checkpoint_resolved='/real/step-003000',
        training_id='V_2_memory_task22_train02',method='memory',source_digest='frozen-code')
    monkeypatch.setattr(manager,'snapshot',lambda:dict(jobs=[job]))
    result=manager.resolve(manager.host,5560)
    assert result['binding']=='observed_process' and result['checkpoint']=='/real/step-003000'
    assert result['training_id']==job['training_id'] and result['source_digest']=='frozen-code'
    assert manager.resolve('other-host',5560) is None
    with pytest.raises(ValueError,match='就绪'):
        manager.resolve(manager.host,5559)
