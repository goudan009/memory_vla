"""Existing-workbench V2.1 launch/cleanup contracts without GPU execution."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
from v21_workbench import Backend, controller, read, write


@pytest.fixture
def backend(tmp_path):
    project=tmp_path/'project';training=tmp_path/'training';live=project/'deployments/web_live'
    python=tmp_path/'python';python.touch()
    base=tmp_path/'generic'
    write(base/'config.json',{'model_type':'gr00t_n1d6'})
    (base/'model.safetensors').touch()
    dataset=tmp_path/'dataset';(dataset/'meta').mkdir(parents=True)
    (dataset/'meta/episodes.jsonl').write_text(''.join(json.dumps({'episode_index':e,'length':256})+'\n' for e in range(200)))
    labels=tmp_path/'labels.parquet';labels.touch()
    remote=dict(TRAINING=training,PROJECT=project,LIVE=live,PYTHON=python,BASE_CHECKPOINT=base,
                gpu_info=lambda:[{'index':g} for g in range(4)],compute_pids=lambda g:[],
                live_run_processes=lambda r:[],process_start=lambda p:None,tail=lambda p:'')
    b=Backend(remote)
    b.request=dict(version='v2_1',task=40,run_id='v2_1__test01__exp14__run001__preflight001',
                   stage='stage1',dataset=str(dataset),labels=str(labels),checkpoint=str(base),
                   preflight_only=True,gpus=[0,1,2,3])
    return b


def fixture_job(b, status='stopped', preflight=False):
    request=dict(b.request,preflight_only=preflight,
                 run_id='v2_1__test01__exp14__run001'+('__preflight001' if preflight else ''))
    if not preflight:
        gate=b.training.parent/'evidence.json';write(gate,{'passed':True})
        request['accept_preflight']=str(gate)
    plan=b.prepare(request)
    root=b.training/plan['run_name']
    plan.update(source=str(b.live),source_manifest={'digest':'source'})
    write(root/'config/web_launch.json',plan)
    write(root/plan['arm']/'status.json',{'status':status,'step':3})
    return root,plan


def test_preview_is_readonly_and_preserves_global_recipe(backend):
    b=backend;plan=b.prepare(b.request)
    assert not b.training.exists()
    assert plan['global_windows_per_step']==2 and plan['global_decisions']==64
    assert plan['train_episodes']==180 and plan['validation_episodes']==20
    assert plan['steps']==6000 and plan['learning_rate']==1e-5
    argv=b.command(plan,b.live,b.training/plan['run_name'])
    assert '--nproc-per-node=4' in argv and argv[argv.index('--output')+1].endswith('/run')
    assert argv[argv.index('--checkpoint-prefix')+1]==plan['run_name']
    assert '--preflight-only' in argv
    for change in ({'gpus':[0,1]},{'global_windows_per_step':8},{'learning_rate':2e-5},{'steps':3}):
        with pytest.raises(ValueError):b.prepare(dict(b.request,**change))


def test_generic_source_and_formal_preflight_required(backend):
    b=backend
    alternate=b.training.parent/'behavior';write(alternate/'config.json',{})
    (alternate/'model.safetensors').touch()
    with pytest.raises(ValueError,match='通用'):
        b.prepare(dict(b.request,checkpoint=str(alternate)))
    with pytest.raises(ValueError,match='preflight.json'):
        b.prepare(dict(b.request,run_id='v2_1__test01__exp14__run001',preflight_only=False))


def test_queue_accepts_busy_resource_without_stopping_anything(backend):
    backend.r['compute_pids']=lambda g:[17]
    with pytest.raises(ValueError,match='现有进程'):backend.prepare(backend.request)
    plan=backend.prepare(dict(backend.request,enqueue=True))
    assert plan['existing_pids']==[17] and plan['enqueue']


def test_pause_and_resume_preserve_budget_and_source(backend):
    b=backend;root,plan=fixture_job(b,status='training')
    control=b.stop({'run':root.name,'step':4})
    assert control['stop_after_step']==4 and read(root/'run/stop.json')==control
    write(root/'baseline/status.json',{'status':'stopped','step':4})
    checkpoint=root/'run/checkpoints/v2_1__test01__exp14__run001__step000004'
    write(checkpoint/'COMPLETE.json',{'step':4});(checkpoint/'training_state.pt').touch()
    request=dict(run=root.name,preview=True)
    preview=b.resume(request)
    assert preview['resume_step']==4 and preview['steps']==6000 and preview['source']==str(b.live)
    assert (root/'run/stop.json').exists()  # Preview does not mutate.
    with pytest.raises(ValueError,match='原预算'):b.resume(dict(request,steps=9000))
    launched=[]
    b.launch=lambda *a,**k:launched.append((a,k)) or {'resumed':True}
    assert b.resume(dict(run=root.name))['resumed']
    assert not (root/'run/stop.json').exists()
    argv=launched[0][0][3]
    assert argv[argv.index('--resume')+1]==str(checkpoint)


def test_preflight_purge_requires_evidence_and_no_consumers(backend):
    b=backend;root,plan=fixture_job(b,preflight=True)
    source=b.project/'deployments'/root.name
    source.mkdir(parents=True)
    launch=read(root/'config/web_launch.json');launch['source']=str(source)
    write(root/'config/web_launch.json',launch)
    request=dict(run=root.name,delete_files=True,expected_output=str(root),purge_preflight=True)
    with pytest.raises(ValueError,match='归档'):b.delete(request)
    evidence=b.training.parent/'permanent-evidence.json';write(evidence,{'job':root.name,'passed':True})
    request['evidence_path']=str(evidence)
    consumer=b.training/'consumer/config/web_launch.json'
    write(consumer,{'checkpoint':str(root/'run/checkpoints/step-000003')})
    with pytest.raises(ValueError,match='引用'):b.delete(request)
    consumer.unlink()
    result=b.delete(request)
    assert result['purged'] and not root.exists() and not source.exists() and evidence.exists()
    assert read(b.training/'deletions'/(root.name+'.json'))['evidence_path']==str(evidence)


def test_formal_output_cannot_use_preflight_purge(backend):
    b=backend;root,_=fixture_job(b)
    evidence=b.training.parent/'evidence.json'
    with pytest.raises(ValueError,match='仅限'):
        b.delete(dict(run=root.name,delete_files=True,expected_output=str(root),purge_preflight=True,evidence_path=str(evidence)))
    assert root.exists()


def test_delete_formal_training_output_permanently_preserves_source(backend):
    b=backend; root,launch=fixture_job(b)
    source=Path(launch['source']);write(source/'keep.json',{'scientific_source':True})
    sibling=b.training/'other_run';write(sibling/'keep.json',{'other':True})
    result=b.delete(dict(run=root.name,delete_files=True,expected_output=str(root)))
    assert result['status']=='deleted' and not result['files_preserved']
    assert not root.exists() and not (b.training/'.trash').exists()
    assert 'trash_path' not in result
    assert (source/'keep.json').is_file() and (sibling/'keep.json').is_file()


def test_rpc_dispatch_coexists_with_legacy_dashboard(backend,monkeypatch):
    spec=importlib.util.spec_from_file_location('v21_remote_test',ROOT/'train/dashboard_remote.py')
    remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
    monkeypatch.setattr(remote,'v21_backend',lambda:backend)
    plan=remote.prepare(backend.request)
    assert plan['version']=='V_2_1'
    root,_=fixture_job(backend,preflight=True)
    assert remote.detail({'run':root.name})['config']['category']=='engineering_preflight'
    assert remote.display_name(root,'baseline',40)==root.name


def test_legacy_run_remains_readable_but_new_requests_use_v2_1(backend):
    b=backend
    legacy='v2.1__test01__exp14__run007'
    root=b.training/legacy
    write(root/'config/web_launch.json',dict(version='v2.1',arm='baseline',method='baseline'))
    write(root/'baseline/status.json',dict(status='stopped',step=0))
    assert b.root({'run':legacy})[0]==root
    with pytest.raises(ValueError,match='新训练标识'):
        b.prepare(dict(b.request,run_id=legacy))


@pytest.mark.parametrize('version',['V_2_1','V2_1','v2_1','v2.1'])
def test_version_aliases_use_one_canonical_workbench_group(backend,version):
    plan=backend.prepare(dict(backend.request,version=version))
    assert plan['version']=='V_2_1'
    root,launch=fixture_job(backend)
    write(root/'config/web_launch.json',dict(launch,version=version))
    assert backend.owns({'run':root.name})
    assert backend.root({'run':root.name})[0]==root
    job=backend.jobs()[0]
    assert job['version']==job['model_version']=='V_2_1'
    assert job['run']==root.name


def test_start_uses_snapshot_existing_queue_and_no_launch_on_preview(backend,monkeypatch):
    b=backend
    write(b.live/'web_source_manifest.json',{'digest':'source','files':{}})
    (b.live/'train').mkdir();(b.live/'train/v21_workbench.py').touch()
    calls=[]
    monkeypatch.setattr('v21_workbench.subprocess.Popen',lambda argv,**kwargs:calls.append((argv,kwargs)) or SimpleNamespace(pid=987654321))
    request=dict(b.request,enqueue=True)
    b.prepare(request);assert not calls
    job=b.start(request,'source')
    root=b.training/request['run_id']
    assert job['status']=='queued' and len(calls)==1
    assert calls[0][0][2].endswith('/train/web_training_queue.py')
    queued=read(b.training/'queues'/root.name/'config.json')
    assert queued['gpus']==[0,1,2,3] and queued['output']==str(root)
    controller_config=read(root/'config/v21_controller.json')
    assert controller_config['env']['PYTHONPATH']==str(Path(job['source'])/'src')
    assert not set(controller_config['env'])-{'PYTHONPATH','PYTHONNOUSERSITE','PYTHONDONTWRITEBYTECODE','HF_HUB_OFFLINE',
                'TRANSFORMERS_OFFLINE','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','TOKENIZERS_PARALLELISM'}
    with pytest.raises(ValueError,match='已存在'):b.start(request,'source')


def test_controller_reports_actual_native_progress_and_failure(tmp_path):
    root=tmp_path/'job'
    output=root/'run';output.mkdir(parents=True)
    # A tiny CPU fixture child exercises the controller boundary; no model.
    code='import json,pathlib; p=pathlib.Path('+repr(str(output))+'); (p/"metrics.jsonl").write_text(json.dumps({"step":4,"fm":1.0})+"\\n"); (p/"status.json").write_text(json.dumps({"status":"stopped","step":4}))'
    config=dict(root=str(root),arm='baseline',steps=6000,preflight_only=False,source=str(tmp_path),
                env={},argv=[sys.executable,'-c',code])
    assert controller(config)==0
    assert read(root/'baseline/status.json')['status']=='stopped'
    assert read(root/'baseline/status.json')['step']==4
    config['argv']=[sys.executable,'-c','raise SystemExit(7)']
    assert controller(config)==7
    assert read(root/'status.json')['status']=='failed'
