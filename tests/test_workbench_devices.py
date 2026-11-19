"""Device routing and source snapshots; never start GPU work in unit checks."""
import importlib.util
import json
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
    for request in [dict(mode='dual1'),dict(mode='single2',gpu_index=1)]:
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
    device=json.loads((ROOT/'configs/workbench_devices.json').read_text())['a6000']
    manager=InferenceManager(ROOT,device['host'],device)
    assert manager.bind_host=='100.100.2.30'
    assert manager.default_python==device['paths']['python']
    assert manager.default_source.startswith('/data/projects/memory_vla/')
    argv=manager.argv(dict(checkpoint='/data/test/checkpoint',gpu_index=0,port=5559))
    assert argv[argv.index('--host')+1]=='100.100.2.30'
    assert argv[argv.index('--python-path')+1]==device['paths']['python']
    assert 'DEVICE_PATHS=' in manager.remote
