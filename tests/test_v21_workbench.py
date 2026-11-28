"""Existing-workbench V2.1 launch/cleanup contracts without GPU execution."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
from v21_workbench import Backend, controller, read, write, dashboard_metrics, metric_summary


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


def test_current_stage1_retrain_cannot_fall_back_to_sequential_recipe(backend):
    # Synthetic namespace for a test only; never creates an actual run.
    req=dict(backend.request,run_id='v2_1__test02__exp01__run999__preflight001')
    with pytest.raises(ValueError,match='禁止回退旧顺序训练配方'):
        backend.prepare(req)
    with pytest.raises(ValueError,match='不支持的原生Stage1参数'):
        backend.prepare(dict(req,recipe='native_random_v2'))
    assert not backend.training.exists()
    # Historical task remains available with its own unchanged recipe.
    assert backend.prepare(backend.request)['learning_rate']==1e-5



def native_request(b):
    base=Path(b.request['checkpoint']);data=Path(b.request['dataset'])
    write(base/'processor_config.json',{})
    for n in ('info','modality'):write(data/'meta'/f'{n}.json',{})
    split=b.training.parent/'split.json';write(split,dict(task=40,seed=42,train=list(range(20,200)),validation=list(range(20))))
    official=b.training.parent/'upstream'
    for n in ('experiment','trainer'):
        file=official/'gr00t/experiment'/f'{n}.py';file.parent.mkdir(parents=True,exist_ok=True);file.touch()
    b.r['gpu_info']=lambda:[dict(index=i,name='NVIDIA A100') for i in range(4)]
    req=dict(b.request)
    req.pop('labels')
    return dict(req,recipe='native_random_v2',run_id='v2_1__test02__exp01__run999__preflight001',
                split=str(split),official_root=str(official))


def test_native_preview_and_command_are_workbench_owned(backend):
    b=backend;req=native_request(b)
    plan=b.prepare(req)
    assert not b.training.exists()
    assert plan['learning_rate']==1e-4 and plan['steps']==6000
    assert plan['accumulation']==8 and plan['global_decisions']==64
    assert plan['preflight_steps']==4
    root=b.training/req['run_id']
    argv=b.command(plan,b.live,root)
    assert argv[argv.index('--job-root')+1]==str(root)
    assert 'stage1_native_train.py' in argv[-3]
    assert plan['output']==str(root)
    assert plan['checkpoint_root']==str(root/'run/checkpoints')
    assert 'v2_1_groot_memoryvla.train' not in argv
    with pytest.raises(ValueError,match='已锁定参数'):
        b.prepare(dict(req,learning_rate=1e-5))
    with pytest.raises(ValueError,match='四个不同'):
        b.prepare(dict(req,gpus=[0,1]))
    with pytest.raises(ValueError,match='accept_preflight'):
        b.prepare(dict(req,run_id=req['run_id'].split('__preflight')[0],preflight_only=False))


def test_native_pause_resume_preserves_identity(backend):
    b=backend;plan=b.prepare(native_request(b))
    root=b.training/plan['run_id']
    plan.update(source=str(b.live),source_manifest={'digest':'source'})
    write(root/'config/web_launch.json',plan);write(root/'baseline/status.json',{'status':'paused','step':2})
    checkpoint=root/'run/checkpoints'/(root.name+'__step000002')
    write(checkpoint/'COMPLETE.json',dict(step=2,run_id=root.name,recipe='native_random_v2',
          per_rank_sample_cursor=32,files={'training_state.pt':'fixture'}))
    (checkpoint/'training_state.pt').touch()
    assert b.checkpoints(root)==[dict(step=2,path=str(checkpoint))]
    preview=b.resume(dict(run=root.name,preview=True))
    assert preview['resume_step']==2 and preview['steps']==6000
    argv=b.command(plan,b.live,root,checkpoint)
    assert argv[argv.index('--resume')+1]==str(checkpoint)
    wrong=checkpoint.with_name('checkpoint-2')
    checkpoint.rename(wrong)
    assert not b.checkpoints(root)


def test_native_detail_reports_action_loss_without_state_weight(backend):
    b=backend;plan=b.prepare(native_request(b))
    root=b.training/plan['run_id']
    plan.update(source=str(b.live),source_manifest={'digest':'source'})
    write(root/'config/web_launch.json',plan)
    write(root/'baseline/status.json',dict(status='training',step=1))
    (root/'run').mkdir()
    (root/'run/metrics.jsonl').write_text(json.dumps(dict(step=1,fm=.25,state=0.))+'\n')
    detail=b.detail(dict(run=root.name))
    assert detail['curve']==[[1,.25,1]]
    assert detail['loss_series']==detail['curve']


def test_native_dashboard_metrics_match_cards_timing_and_validation(backend):
    b=backend;plan=b.prepare(native_request(b))
    root=b.training/plan['run_id']
    plan.update(source=str(b.live),source_manifest={'digest':'source'},preflight_only=False,preflight_steps=None)
    write(root/'config/web_launch.json',plan)
    rows=[dict(step=10,fm=.2,state=0.,grad_norm=.1,lr=1e-4,seen_samples=640,
               elapsed_seconds=30.,gpu_peak_allocated=32*2**30),
          dict(step=20,fm=.1,state=0.,grad_norm=.05,lr=9e-5,seen_samples=1280,
               elapsed_seconds=50.,gpu_peak_allocated=33*2**30)]
    write(root/'baseline/status.json',dict(status='training',step=22,last=rows[-1]))
    (root/'run').mkdir()
    journal=root/'run/metrics.jsonl'
    journal.write_text(''.join(json.dumps(row)+'\n' for row in rows)+'{"step":')
    write(root/'run/validation-000000.json',dict(step=0,fm=.3,samples=100))
    before=journal.read_bytes()
    job=b.jobs()[0];detail=b.detail(dict(run=root.name))
    assert job['last']['loss']==.1 and job['last']['fm']==.1
    assert job['last']['gradient_norm']==.05 and job['last']['peak_allocated_gib']==33
    assert job['last']['seconds']==2. and job['last']['decisions_per_second_end_to_end']==32.
    timing=detail['timing']
    assert timing['elapsed_seconds']==50. and timing['last_step_seconds']==2.
    assert timing['recent_mean_step_seconds']==2.5 and timing['recent_samples']==20
    assert timing['eta_seconds']==pytest.approx((6000-22)*2.5)
    assert detail['validation'][0]['validation_loss']==.3
    assert detail['epochs']==[] and detail['curve_metadata']['has_epochs'] is False
    assert detail['curve_metadata']['steps_label']=='记录点 Loss'
    assert detail['state']['last']['loss']==.1 and journal.read_bytes()==before


def test_legacy_dashboard_metrics_keep_weighted_loss_and_epoch_evidence(backend):
    b=backend;root,launch=fixture_job(b,status='training')
    rows=[dict(step=i,fm=.2,state=2.,seconds=2.,decisions=64,epoch=0 if i<3 else 1,
               grad_norm=.1,peak_allocated_gib=30.) for i in (1,2,3)]
    (root/'run').mkdir()
    (root/'run/metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    write(root/'baseline/status.json',dict(status='training',step=3,last=rows[-1]))
    detail=b.detail(dict(run=root.name));job=b.jobs()[0]
    assert job['last']['loss']==pytest.approx(.22)
    assert job['last']['decisions_per_second_end_to_end']==32.
    assert detail['loss_series']==[[i,pytest.approx(.22),1] for i in (1,2,3)]
    assert detail['curve']==[[3,pytest.approx(.22),3]]
    assert detail['epochs'][0]['complete'] is True and detail['epochs'][1]['complete'] is False
    assert detail['timing']['elapsed_seconds']==6. and detail['timing']['eta_seconds']==(6000-3)*2.
    write(root/'baseline/status.json',dict(status='completed',step=6000,last=rows[-1]))
    assert b.detail(dict(run=root.name))['timing']['eta_seconds']==0


def test_missing_metrics_do_not_become_zero_loss_or_invented_timing(backend):
    b=backend;root,_=fixture_job(b)
    (root/'run').mkdir()
    (root/'run/metrics.jsonl').write_text('{"step":1}\n{"step":2,"fm":NaN}\n')
    detail=b.detail(dict(run=root.name))
    assert detail['loss_series']==[] and detail['curve']==[]
    assert detail['timing']['elapsed_seconds'] is None
    assert b.jobs()[0]['last'] is None


def test_native_timing_retains_elapsed_segments_after_resume_and_stop_budget():
    launch=dict(recipe='native_random_v2',preflight_only=False,steps=6000)
    rows=dashboard_metrics([dict(step=10,fm=.2,seen_samples=640,elapsed_seconds=30.),
                            dict(step=20,fm=.1,seen_samples=1280,elapsed_seconds=20.)],launch)
    summary=metric_summary(rows,launch,dict(status='training',step=22),dict(stop_after_step=100))
    assert summary['timing']['elapsed_seconds']==50.
    assert summary['timing']['eta_seconds']==78*2.5
    assert rows[-1]['seconds']==2. and rows[-1]['decisions_per_second_end_to_end']==32.
    assert metric_summary(rows,launch,dict(status='paused',step=22),{})['timing']['eta_seconds'] is None
