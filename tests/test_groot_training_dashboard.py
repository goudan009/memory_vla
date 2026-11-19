"""Remote launch/snapshot contracts without starting GPU training."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def remote(tmp_path, monkeypatch):
    m = load('training_remote_test', ROOT/'train/dashboard_remote.py')
    m.PROJECT = tmp_path/'project'
    m.TRAINING = tmp_path/'training'
    m.LIVE = m.PROJECT/'deployments/web_live'
    m.PYTHON = tmp_path/'python'
    m.PYTHON.touch()
    dataset = tmp_path/'data'
    m.atomic_json(dataset/'meta/info.json', {})
    checkpoint = tmp_path/'official'
    m.atomic_json(checkpoint/'config.json', {})
    split = tmp_path/'split.json'
    m.atomic_json(split, dict(task=0,train=list(range(180)),validation=list(range(180,200))))
    monkeypatch.setattr(m, 'gpu_info', lambda:[dict(index=i) for i in range(4)])
    monkeypatch.setattr(m, 'compute_pids', lambda gpus:[])
    monkeypatch.setattr(m, 'catalog', lambda:dict(datasets=[dict(task=0,dataset=str(dataset),split=str(split))]))
    m.request = dict(task=0,task_name='turning_on_radio',dataset=str(dataset),split=str(split),checkpoint=str(checkpoint),
        gpus=[0,2,3],validation_gpu=1,arm='baseline',steps=3000,global_windows_per_step=3,microbatch=4,
        seed=42,learning_rate=1e-5,share_existing=False)
    return m


def test_preview_keeps_batch_freeze_and_busy_gpu_permission(remote, monkeypatch):
    plan = remote.prepare(remote.request)
    assert plan['run_name']=='V_1_baseline_task0_trainNN'
    assert remote.prepare(dict(remote.request,version='V_2',arm='memory_v1'))['run_name']=='V_2_memory_task0_trainNN'
    with pytest.raises(ValueError,match='版本'):
        remote.prepare(dict(remote.request,version='../V_2'))
    assert plan['global_decisions']==96 and plan['frozen_eagle'] and not plan['lora']
    assert plan['existing_pids']==[] and plan['checkpoint_interval']==1000
    with pytest.raises(ValueError,match='整除'):
        remote.prepare(dict(remote.request,gpus=[0,1]))
    monkeypatch.setattr(remote,'compute_pids',lambda gpus:[123])
    with pytest.raises(ValueError,match='现有进程'):
        remote.prepare(remote.request)
    assert remote.prepare(dict(remote.request,share_existing=True))['existing_pids']==[123]
    assert not remote.TRAINING.exists()


def test_memory_training_settings_reach_worker_and_reject_architecture_swap(remote):
    request = dict(remote.request, arm='memory_v1')
    plan = remote.prepare(request)
    assert (plan['memory_conditioning'], plan['memory_length'], plan['image_augmentation']) == ('residual',64,True)
    assert plan['memory_cognitive_qk_norm'] == 'layernorm'
    custom = remote.prepare(dict(request,memory_length=256,image_augmentation=False))
    argv = remote.command(custom,remote.LIVE,remote.TRAINING/'new')
    assert argv[argv.index('--memory-length')+1]=='256'
    assert argv[argv.index('--memory-conditioning')+1]=='residual'
    assert '--no-image-augmentation' in argv
    assert argv[argv.index('--memory-cognitive-qk-norm')+1] == 'layernorm'
    assert argv[argv.index('--memory-residual-readout')+1] == 'delta'
    for values in [dict(memory_length=0),dict(memory_conditioning='other'),dict(image_augmentation='false')]:
        with pytest.raises(ValueError):
            remote.prepare(dict(request,**values))
    remote.atomic_json(Path(request['checkpoint'])/'config.json',dict(memory_enabled=True))
    assert remote.prepare(request)['memory_conditioning']=='residual'
    assert remote.prepare(dict(request,memory_conditioning='residual'))['memory_conditioning']=='residual'
    assert remote.prepare(dict(request,memory_conditioning='residual'))['memory_cognitive_qk_norm']=='none'
    with pytest.raises(ValueError, match='原始GR00T'):
        remote.prepare(dict(request,memory_conditioning='residual',memory_cognitive_qk_norm='layernorm'))
    remote.atomic_json(Path(request['checkpoint'])/'config.json',dict(memory_enabled=True,memory_conditioning='direct'))
    assert remote.prepare(request)['memory_conditioning']=='direct'
    with pytest.raises(ValueError,match='原始GR00T'):
        remote.prepare(dict(request,memory_conditioning='residual'))


def test_value_only_residual_readout_reaches_canonical_worker(remote):
    request = dict(remote.request, arm='memory_v1', memory_residual_readout='value')
    plan = remote.prepare(request)
    argv = remote.command(plan, remote.LIVE, remote.TRAINING/'new')
    assert plan['memory_residual_readout'] == 'value'
    assert argv[argv.index('--memory-residual-readout')+1] == 'value'
    with pytest.raises(ValueError, match='Residual readout'):
        remote.prepare(dict(request, memory_residual_readout='unknown'))
    with pytest.raises(ValueError, match='Residual Memory'):
        remote.prepare(dict(remote.request, memory_conditioning='direct', memory_residual_readout='value'))
    remote.atomic_json(Path(request['checkpoint'])/'config.json', dict(memory_enabled=True,
        memory_conditioning='residual', memory_residual_readout='delta'))
    with pytest.raises(ValueError, match='Residual readout与权重不同'):
        remote.prepare(request)


def test_cognitive_projection_scope_and_optimizer_epsilon_reach_worker(remote):
    request = dict(remote.request, arm='memory_v1', memory_residual_readout='value',
        memory_cognitive_residual_projection=True, memory_cognitive_residual_scope='text',
        optimizer_eps=1e-12)
    plan = remote.prepare(request)
    argv = remote.command(plan, remote.LIVE, remote.TRAINING/'new')
    assert '--memory-cognitive-residual-projection' in argv
    assert argv[argv.index('--memory-cognitive-residual-scope')+1] == 'text'
    assert float(argv[argv.index('--optimizer-eps')+1]) == 1e-12
    for values in (dict(optimizer_eps=0), dict(optimizer_eps=float('nan')),
                   dict(memory_cognitive_residual_projection='true'),
                   dict(memory_cognitive_residual_scope='all')):
        with pytest.raises(ValueError):
            remote.prepare(dict(request, **values))
    remote.atomic_json(Path(request['checkpoint'])/'config.json', dict(memory_enabled=True,
        memory_conditioning='residual', memory_residual_readout='value',
        memory_cognitive_residual_projection=True, memory_cognitive_residual_scope='text'))
    saved = remote.prepare(dict(remote.request, arm='memory_v1'))
    assert saved['memory_cognitive_residual_projection'] is True
    assert saved['memory_cognitive_residual_scope'] == 'text'
    with pytest.raises(ValueError, match='原始GR00T'):
        remote.prepare(dict(request, memory_cognitive_residual_projection=False))


def test_memory_options_roundtrip_through_public_launcher_worker():
    import argparse
    from v1_groot_memoryvla.experiment.memory_training_options import add_memory_training_options, memory_training_argv
    parser = argparse.ArgumentParser()
    add_memory_training_options(parser)
    args = parser.parse_args(['--optimizer-eps', '1e-12', '--memory-cognitive-residual-projection',
                              '--memory-cognitive-residual-scope', 'text'])
    worker = parser.parse_args(memory_training_argv(args))
    assert vars(worker) == vars(args)


def test_recommended_validation_budget_reaches_launcher(remote):
    plan = remote.prepare(dict(remote.request, fine_tuning_mode='action_full',
        validation_interval=3000, disable_early_stopping=True))
    argv = remote.command(plan, remote.LIVE, remote.TRAINING/'new')
    assert argv[argv.index('--validation-interval')+1] == '3000'
    assert '--disable-early-stopping' in argv
    legacy = remote.prepare(remote.request)
    assert legacy['validation_interval'] == 1000 and not legacy['disable_early_stopping']
    assert '--disable-early-stopping' not in remote.command(legacy, remote.LIVE, remote.TRAINING/'old')
    for values in [dict(validation_interval=500), dict(validation_interval=True),
                   dict(disable_early_stopping='false'), dict(fine_tuning_mode='unknown')]:
        with pytest.raises(ValueError):
            remote.prepare(dict(remote.request, **values))


@pytest.mark.parametrize('mode', ['action_full', 'full', 'custom', 'lora'])
def test_finetuning_options_reach_real_worker_command(remote, mode):
    plan = remote.prepare(dict(remote.request, fine_tuning_mode=mode, freeze_vision=True,
        lora_targets='llm,action_dit', lora_rank=8, lora_alpha=16., lora_dropout=.05,
        llm_last_n_layers=2, gradient_checkpointing=True, weight_decay=.03, max_grad_norm=.5))
    argv = remote.command(plan, remote.LIVE, remote.TRAINING/'new')
    assert argv[argv.index('--fine-tuning-mode')+1] == mode
    assert '--freeze-vision' in argv and '--gradient-checkpointing' in argv
    assert argv[argv.index('--lora-rank')+1] == '8'
    assert argv[argv.index('--weight-decay')+1] == '0.03'
    assert argv[argv.index('--max-grad-norm')+1] == '0.5'
    assert plan['lora'] == (mode == 'lora')
    assert plan['frozen_eagle'] == (mode in ('action_full', 'custom'))


@pytest.mark.parametrize('values', [dict(freeze_llm='false'), dict(lora_rank=True),
    dict(lora_dropout=-.1), dict(weight_decay=-1), dict(max_grad_norm=0),
    dict(fine_tuning_mode='lora', freeze_llm=True, freeze_action_dit=True)])
def test_finetuning_web_rejects_invalid_settings(remote, values):
    with pytest.raises(ValueError):
        remote.prepare(dict(remote.request, **values))


def test_recommendation_profile_counts_episode_tails_and_registered_split(remote):
    source = dict(dataset=remote.request['dataset'], split=remote.request['split'])
    assert remote.dataset_training_profile(source) is None
    path = Path(source['dataset'])/'meta/episodes.jsonl'
    path.write_text(''.join(json.dumps(dict(episode_index=e,length=257 if e<180 else 1))+'\n' for e in range(200)))
    profile = remote.dataset_training_profile(source)
    assert profile == dict(total_episodes=200, total_frames=180*257+20,
        total_windows=380, train_windows=360, split_seed=42, validation_percent=10.)
    checkpoint = Path(remote.request['checkpoint'])
    remote.atomic_json(checkpoint/'config.json', dict(memory_enabled=True, memory_length=256))
    saved = remote.checkpoint_training_profile(checkpoint)
    assert saved['kind'] == 'trained' and saved['memory_length'] == 256
    assert saved['memory_conditioning'] == 'residual'


def test_start_uses_immutable_source_and_canonical_entry(remote, monkeypatch):
    source_file = remote.LIVE/'train/run_groot_train.sh'
    source_file.parent.mkdir(parents=True)
    source_file.write_text('version-one')
    remote.atomic_json(remote.LIVE/'web_source_manifest.json',dict(digest='sha-one'))
    remote.request_digest = 'sha-one'
    launched=[]
    monkeypatch.setattr(remote.subprocess,'Popen',lambda argv,**kw:launched.append(argv) or SimpleNamespace(pid=12345))
    result=remote.start(remote.request)
    assert result['run']=='V_1_baseline_task0_train01'
    output=Path(result['output'])
    source=Path(result['source'])
    assert launched[0][:2]==['bash',str(source/'train/run_groot_train.sh')]
    assert '--async-validation-gpu' in launched[0]
    assert '--resume-step' not in launched[0]
    source_file.write_text('version-two')
    assert (source/'train/run_groot_train.sh').read_text()=='version-one'
    published=remote.read_json(output/'config/data_split.json')
    original=remote.read_json(Path(remote.request['split']))
    assert published['train']==original['train'] and published['validation']==original['validation']
    assert remote.read_json(output/'baseline/validation_queue/step-000000.json')['checkpoint']==remote.request['checkpoint']
    assert remote.jobs()['jobs'][0]['run']==result['run']
    assert remote.detail(dict(run=result['run'],arm='baseline'))['curve']==[]
    repeated=remote.start(remote.request)
    assert repeated['run']=='V_1_baseline_task0_train02'
    assert Path(repeated['source'])!=source
    assert (source/'train/run_groot_train.sh').read_text()=='version-one'


def test_training_notes_saved_with_run_and_listed_without_changing_command(remote,monkeypatch):
    (remote.LIVE/'train').mkdir(parents=True)
    (remote.LIVE/'train/run_groot_train.sh').write_text('saved launcher')
    remote.atomic_json(remote.LIVE/'web_source_manifest.json',dict(digest='notes-source'))
    remote.request_digest='notes-source'
    monkeypatch.setattr(remote.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(pid=99999999))
    notes='Task0 首轮\n只修改记忆容量 <对比>'
    plan=remote.prepare(dict(remote.request,notes='  '+notes+'  '))
    assert plan['notes']==notes
    assert '--notes' not in remote.command(plan,remote.LIVE,remote.TRAINING/'new')
    job=remote.start(dict(remote.request,notes=notes));root=Path(job['output'])
    assert remote.read_json(root/'config/web_launch.json')['notes']==notes
    assert remote.jobs()['jobs'][0]['notes']==notes
    assert remote.detail(dict(run=job['run'],arm='baseline'))['notes']==notes
    assert remote.detail(dict(run=job['run'],arm='baseline'))['training_parameters']['values']['learning_rate']==1e-5
    for bad in [None,{},'x'*2001]:
        with pytest.raises(ValueError,match='备注'):
            remote.prepare(dict(remote.request,notes=bad))


def test_recorded_training_parameters_prefer_worker_and_keep_missing_unknown(remote):
    root=make_job(remote)
    remote.atomic_json(root/'config/web_launch.json',dict(learning_rate=9e-4,microbatch=32,
        global_windows_per_step=6,steps=5000,validation_enabled=True,validation_gpu=3,validation_interval=1000))
    remote.atomic_json(root/'config/data_split.json',dict(tasks=[0,34,40],train=[1,2],validation=[3]))
    worker=dict(learning_rate=1e-5,microbatch=4,global_windows_per_step=4,decisions_per_window=32,
        steps=5000,seed=42,no_validation=True,image_augmentation=False,memory_length=64,eval_interval=3000)
    remote.atomic_json(root/'baseline/config/ddp.json',worker)
    remote.atomic_json(root/'baseline/status.json',dict(status='training',step=2001,total_steps=8000))
    detail=remote.detail(dict(run=root.name,arm='baseline'))
    p=detail['training_parameters']['values']
    assert p['learning_rate']==1e-5 and p['microbatch']==4
    assert p['global_decisions_max']==128 and p['gpus']=='1'
    assert p['steps']==8000 and p['original_steps']==5000
    assert p['validation_enabled'] is False and p['image_augmentation'] is False
    assert p['validation_interval']==3000
    assert p['train_episodes']==2 and p['validation_episodes']==1 and p['tasks']==[0,34,40]
    assert 'checkpoint_interval' not in p and 'frozen_eagle' not in p
    assert detail['notes']==''


def test_recorded_training_parameters_fall_back_to_historical_worker_argv(remote):
    root=make_job(remote)
    plan=remote.read_json(root/'config/ddp_launch.json')
    plan['commands']['baseline']['argv']=['python','--learning-rate','0.00002','--microbatch','2','--seed','13']
    remote.atomic_json(root/'config/ddp_launch.json',plan)
    info=remote.detail(dict(run=root.name,arm='baseline'))['training_parameters']
    assert info['values']['learning_rate']==2e-5 and info['values']['microbatch']==2
    assert info['values']['seed']==13 and info['values']['validation_enabled'] is None


def make_job(remote):
    root=remote.TRAINING/'task0_test'
    remote.atomic_json(root/'config/ddp_launch.json',dict(steps_each=5000,commands={'baseline':{'CUDA_VISIBLE_DEVICES':'1'}}))
    remote.atomic_json(root/'config/data_split.json',dict(task=0))
    remote.atomic_json(root/'config/run_config.json',dict(source='/snapshot/version1'))
    remote.atomic_json(root/'baseline/status.json',dict(status='training',step=1000,total_steps=5000))
    remote.atomic_json(root/'baseline/config/ddp.json',dict(steps_per_epoch=2))
    metrics=root/'baseline/metrics/metrics.jsonl'
    metrics.parent.mkdir(parents=True)
    records=[dict(kind='train',step=1,epoch=0,loss=1.,global_decisions=32),
             dict(kind='eval',step=0,validation_loss=9.,action_mae={'normal':2.}),
             dict(kind='train',step=2,epoch=0,loss=3.,global_decisions=96),
             dict(kind='train',step=3,epoch=1,loss=2.,global_decisions=32)]
    metrics.write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    checkpoint=root/'baseline/checkpoints/step-001000'
    remote.atomic_json(checkpoint/'config.json',{})
    remote.atomic_json(root/'baseline/validation_queue/step-001000.json',dict(step=1000,checkpoint=str(checkpoint)))
    remote.atomic_json(root/'baseline/checkpoints/step-002000/config.json',{})
    return root


def resumable_job(remote, monkeypatch, validation=False):
    (remote.LIVE/'train').mkdir(parents=True)
    (remote.LIVE/'train/run_groot_train.sh').write_text('saved launcher')
    remote.atomic_json(remote.LIVE/'web_source_manifest.json',dict(digest='saved-source'))
    remote.request_digest='saved-source'
    monkeypatch.setattr(remote.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(pid=99999999))
    job=remote.start(dict(remote.request,validation_enabled=validation))
    root=Path(job['output']); source=Path(job['source'])
    (source/'src/v1_groot_memoryvla/experiment').mkdir(parents=True)
    (source/'src/v1_groot_memoryvla/experiment/task21_ddp.py').write_text('original worker')
    config=dict(dataset=remote.request['dataset'],checkpoint=remote.request['checkpoint'],arm='baseline',
        steps=3000,world_size=3,seed=42,global_windows_per_step=3,microbatch=4,learning_rate=1e-5,
        no_validation=not validation)
    remote.atomic_json(root/'baseline/config/ddp.json',config)
    worker=['python','-m','torch.distributed.run','--module','v1_groot_memoryvla.experiment.task21_ddp',
        '--output',str(root),'--steps','3000','--arm','baseline']+(['--no-validation'] if not validation else ['--async-validation'])
    remote.atomic_json(root/'config/ddp_launch.json',dict(steps_each=3000,
        commands={'baseline':dict(CUDA_VISIBLE_DEVICES='0,2,3',argv=worker)}))
    remote.atomic_json(root/'baseline/status.json',dict(status='completed',step=2000,training_complete=True))
    remote.atomic_json(root/'status.json',dict(status='completed'))
    remote.atomic_json(root/'baseline/validation_status.json',dict(status='completed'))
    checkpoint=root/'baseline/checkpoints/step-002000'
    remote.atomic_json(checkpoint/'config.json',{})
    (checkpoint/'model.safetensors').touch()
    remote.atomic_json(root/'baseline/checkpoints/latest.json',dict(step=2000,path=str(checkpoint),
        world_size=3,seed=42,global_windows_per_step=3))
    for name in ['latest_optimizer.pt',*[f'latest_rank-{r:03d}.pt' for r in range(3)]]:
        (root/'baseline/checkpoints'/name).touch()
    monkeypatch.setattr(remote,'live_run_processes',lambda root:[])
    monkeypatch.setattr(remote,'validate_resume',lambda *args:None)
    return root


def test_resume_reuses_snapshot_optimizer_seed_batch_and_output(remote,monkeypatch):
    root=resumable_job(remote,monkeypatch)
    remote.atomic_json(root/'metrics/stop_training.json',dict(stop_after_step=2000))
    row=remote.jobs()['jobs'][0]
    assert row['resume']['available'] and row['resume']['step']==2000
    launched=[]
    monkeypatch.setattr(remote.subprocess,'Popen',lambda argv,**kw:launched.append((argv,kw)) or SimpleNamespace(pid=99999998))
    before=(root/'baseline/config/ddp.json').read_bytes()
    result=remote.resume(dict(run=root.name,arm='baseline',steps=3000),controller_source='controller code')
    assert result['run']==root.name and result['resume_step']==2000
    saved=remote.read_json(Path(result['attempt'])/'request.json')
    assert saved['source']==str(remote.PROJECT/'deployments'/root.name)
    assert saved['worker_argv'][saved['worker_argv'].index('--resume-step')+1]=='2000'
    assert '--no-validation' in saved['worker_argv'] and not saved['validation_enabled']
    assert saved['gpus']==[0,2,3] and saved['steps']==3000
    assert (root/'baseline/config/ddp.json').read_bytes()==before
    assert not (root/'metrics/stop_training.json').exists()
    assert (Path(result['attempt'])/'previous/metrics/stop_training.json').exists()
    assert remote.read_json(root/'baseline/status.json')['status']=='resuming'
    with pytest.raises(ValueError,match='停止|结束|运行'):
        remote.resume(dict(run=root.name,arm='baseline',steps=3000),controller_source='controller code')


@pytest.mark.parametrize('case',['active','missing_rank','missing_weight','paired','busy','small_budget','running_validator'])
def test_resume_refuses_incomplete_or_active_run_before_mutation(remote,monkeypatch,case):
    root=resumable_job(remote,monkeypatch)
    request=dict(run=root.name,arm='baseline',steps=3000)
    if case=='active': monkeypatch.setattr(remote,'live_run_processes',lambda root:[123])
    if case=='missing_rank': (root/'baseline/checkpoints/latest_rank-001.pt').unlink()
    if case=='missing_weight': (root/'baseline/checkpoints/step-002000/model.safetensors').unlink()
    if case=='paired':
        plan=remote.read_json(root/'config/ddp_launch.json');plan['commands']['memory_v1']=plan['commands']['baseline']
        remote.atomic_json(root/'config/ddp_launch.json',plan)
    if case=='busy': monkeypatch.setattr(remote,'compute_pids',lambda g:[123])
    if case=='small_budget': request['steps']=2000
    if case=='running_validator': remote.atomic_json(root/'baseline/validation_status.json',dict(status='validating'))
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError): remote.resume(request,controller_source='controller code')
    assert before=={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_resume_archives_unsaved_tail_and_rolls_back_failed_spawn(remote,monkeypatch):
    root=resumable_job(remote,monkeypatch)
    metrics=root/'baseline/metrics/metrics.jsonl';metrics.parent.mkdir(parents=True)
    metrics.write_text(''.join(json.dumps(dict(kind='train',step=s))+'\n' for s in [1999,2000,2001]))
    remote.atomic_json(root/'metrics/stop_training.json',dict(stop_after_step=2000))
    def fail(*a,**kw): raise OSError('spawn failed')
    monkeypatch.setattr(remote.subprocess,'Popen',fail)
    with pytest.raises(OSError,match='spawn failed'):
        remote.resume(dict(run=root.name,arm='baseline',steps=3000),controller_source='controller code')
    assert [json.loads(s)['step'] for s in metrics.read_text().splitlines()]==[1999,2000,2001]
    assert remote.read_json(root/'baseline/status.json')['status']=='completed'
    assert (root/'metrics/stop_training.json').exists()
    monkeypatch.setattr(remote.subprocess,'Popen',lambda *a,**kw:SimpleNamespace(pid=99999998))
    result=remote.resume(dict(run=root.name,arm='baseline',steps=3000),controller_source='controller code')
    assert [json.loads(s)['step'] for s in metrics.read_text().splitlines()]==[1999,2000]
    assert (Path(result['attempt'])/'previous/baseline/metrics/metrics.jsonl').read_text().endswith('2001}\n')


def test_resume_keeps_async_validation_command_and_requires_explicit_gpu_sharing(remote,monkeypatch):
    root=resumable_job(remote,monkeypatch,validation=True)
    monkeypatch.setattr(remote,'compute_pids',lambda g:[123,456])
    result=remote.resume(dict(run=root.name,arm='baseline',steps=4000,share_existing=True),controller_source='controller code')
    spec=remote.read_json(Path(result['attempt'])/'request.json')
    assert spec['validation_enabled']
    assert '--async-validation-gpu' in spec['argv'] and '--no-validation' not in spec['argv']
    assert spec['argv'][spec['argv'].index('--allow-service-pids')+1:]==['123','456']
    assert spec['argv'][spec['argv'].index('--steps')+1]=='4000'
    origin=remote.detail(dict(run=root.name,arm='baseline'))['initial_weights']
    assert origin['mode']=='resume' and origin['resume_step']==2000
    assert origin['optimizer_restored'] is False  # submitted is not yet restored


@pytest.mark.parametrize('validation',[False,True])
def test_resume_reads_legacy_validation_setting_from_saved_worker_not_current_defaults(remote,monkeypatch,validation):
    root=resumable_job(remote,monkeypatch,validation=validation)
    launch=remote.read_json(root/'config/web_launch.json')
    launch.pop('validation_enabled');launch.pop('validation_gpu')
    remote.atomic_json(root/'config/web_launch.json',launch)
    info=remote.jobs()['jobs'][0]['resume']
    assert info['available'] and info['validation_enabled']==validation
    assert info['validation_gpu']==(1 if validation else None)


def test_resume_failed_controller_ignores_stale_status_but_never_live_ranks(remote,monkeypatch):
    root=resumable_job(remote,monkeypatch)
    remote.atomic_json(root/'status.json',dict(status='failed'))
    remote.atomic_json(root/'baseline/status.json',dict(status='training',step=2001))
    remote.atomic_json(root/'baseline/validation_status.json',dict(status='waiting_for_checkpoint'))
    assert remote.jobs()['jobs'][0]['resume']['available']
    monkeypatch.setattr(remote,'live_run_processes',lambda root:[123])
    with pytest.raises(ValueError,match='仍有训练或验证进程'):
        remote.resume(dict(run=root.name,arm='baseline',steps=3000),controller_source='controller code')


@pytest.mark.parametrize('success',[True,False])
def test_resume_controller_runs_saved_worker_on_cpu_and_tracks_exit(tmp_path,success):
    """Exercise the actual supervisor process without allocating any GPU."""
    root=tmp_path/'run';source=tmp_path/'snapshot';source.mkdir()
    remote=load('resume_controller_fixture',ROOT/'train/dashboard_remote.py')
    remote.atomic_json(root/'baseline/status.json',dict(status='resuming',step=2000))
    (root/'logs').mkdir();log=root/'logs/baseline_torchrun.log';log.write_text('old log\n')
    worker='''import json,os,sys
from pathlib import Path
root=Path(sys.argv[1])
assert os.environ['CUDA_VISIBLE_DEVICES']=='0,2'
assert os.environ['PYTHONPATH']==str(Path.cwd()/'src')
print('resumed from saved state',flush=True)
'''+("(root/'baseline/status.json').write_text(json.dumps(dict(status='completed',step=2001,training_complete=True)))" if success else 'raise SystemExit(7)')
    spec=dict(output=str(root),source=str(source),arm='baseline',gpus=[0,2],resume_step=2000,
        steps=3000,checkpoint='/saved/step-002000',validation_enabled=False,
        worker_argv=[sys.executable,'-c',worker,str(root)])
    remote.atomic_json(root/'config/resume.json',spec)
    result=subprocess.run([sys.executable,str(ROOT/'train/web_training_resume.py'),'--config',str(root/'config/resume.json')],
        text=True,capture_output=True,timeout=10)
    assert (result.returncode==0)==success
    assert remote.read_json(root/'status.json')['status']==('completed' if success else 'failed')
    assert log.read_text().startswith('old log\nresumed from saved state')
    if success:
        assert remote.read_json(root/'baseline/config/resume-step-002000.json')['optimizer_restored']
    else:
        assert remote.read_json(root/'baseline/status.json')['status']=='failed'
        assert not (root/'baseline/config/resume-step-002000.json').exists()


def test_progress_uses_training_steps_and_recorded_source(remote):
    make_job(remote)
    row=remote.jobs()['jobs'][0]
    assert row['step']==1000 and row['source']=='/snapshot/version1'
    assert row['name']=='V_1_baseline_task0_train01'


def test_detail_weighted_loss_and_only_completed_checkpoints(remote):
    make_job(remote)
    result=remote.detail(dict(run='task0_test',arm='baseline'))
    assert result['epochs'][0]['loss']==2.5 and result['epochs'][0]['complete']
    assert not result['epochs'][1]['complete']
    assert result['validation'][0]['step']==0
    assert len(result['checkpoints'])==1 and result['checkpoints'][0]['step']==1000


def test_initial_weights_use_recorded_worker_config_before_launch_selection(remote):
    root=make_job(remote)
    remote.atomic_json(root/'config/web_launch.json',dict(checkpoint=str(remote.OFFICIAL)))
    remote.atomic_json(root/'baseline/config/ddp.json',dict(checkpoint=str(remote.BASE_CHECKPOINT),resume_step=0))
    info=remote.detail(dict(run=root.name,arm='baseline'))['initial_weights']
    assert info['checkpoint']==str(remote.BASE_CHECKPOINT)
    assert info['label']=='官方 GR00T N1.6-3B · 通用预训练'
    assert info['mode']=='fresh' and info['optimizer_restored'] is False
    assert info['source_file']==str(root/'baseline/config/ddp.json')


def test_initial_weights_fall_back_to_recorded_launch_and_preserve_resume_provenance(remote):
    root=make_job(remote)
    remote.atomic_json(root/'config/ddp_launch.json',dict(commands={'baseline':dict(argv=[
        'python','--checkpoint',str(remote.OFFICIAL),'--resume-step','2000'])}))
    info=remote.detail(dict(run=root.name,arm='baseline'))['initial_weights']
    assert info['checkpoint']==str(remote.OFFICIAL)
    assert info['mode']=='resume' and info['resume_step']==2000
    assert info['label']=='官方 GR00T N1.6-BEHAVIOR1k · B1K 微调'
    assert info['source_file']==str(root/'config/ddp_launch.json')


def test_initial_weights_do_not_substitute_current_page_defaults_for_missing_history(remote):
    root=make_job(remote)
    request=dict(run=root.name,arm='baseline')
    info=remote.detail(request)['initial_weights']
    assert info['checkpoint'] is None and info['mode']=='unknown'
    remote.atomic_json(root/'config/web_launch.json',dict(checkpoint='/weights/custom/step-001000'))
    info=remote.detail(request)['initial_weights']
    assert info['checkpoint']=='/weights/custom/step-001000'
    assert info['label']=='项目权重 / 自定义权重'


def test_training_curve_keeps_every_optimizer_loss_separate_from_validation(remote):
    root=make_job(remote)
    result=remote.detail(dict(run=root.name,arm='baseline'))
    assert result['loss_series']==[[1,1.,1],[2,3.,1],[3,2.,1]]
    assert result['curve']==[[3,2.4,3]]


def test_training_curve_exponential_smoothing_preserves_raw_values():
    node=Path('/home/1024201092WYH/.nvm/versions/node/v22.23.1/bin/node')
    if not node.exists():
        pytest.skip('Node.js required for curve check')
    result=subprocess.run([str(node)],input='''
const assert=require('node:assert/strict');
const {smoothTrainingCurve}=require('./train/dashboard.js');
const raw=[[1,1,1],[2,9,1],[3,1,1]], copy=JSON.stringify(raw);
assert.deepEqual(smoothTrainingCurve(raw,0),raw);
assert.deepEqual(smoothTrainingCurve([],0.9),[]);
const curve=smoothTrainingCurve(raw,0.5);
assert.equal(curve[0][1],1);
assert.ok(Math.abs(curve[1][1]-19/3)<1e-12);
assert.ok(Math.abs(curve[2][1]-23/7)<1e-12);
assert.equal(JSON.stringify(raw),copy);
assert.deepEqual(smoothTrainingCurve([[10,2,1],[20,2,1]],0.99).map(p=>p[1]),[2,2]);
assert.deepEqual(curve.map(p=>p[0]),[1,2,3]);
''',text=True,capture_output=True,cwd=ROOT)
    assert result.returncode==0,result.stderr


def test_training_timing_counts_optimizer_loops_and_uses_recent_fifty_steps(remote):
    root=make_job(remote)
    rows=[dict(kind='train',step=i,epoch=0,loss=1.,global_decisions=128,seconds=float(i)) for i in range(1,61)]
    rows.insert(30,dict(kind='eval',step=30,seconds=10000,validation_loss=1.))
    (root/'baseline/metrics/metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    remote.atomic_json(root/'baseline/status.json',dict(status='training',step=60,total_steps=160,world_size=4))
    timing=remote.detail(dict(run=root.name,arm='baseline'))['timing']
    assert timing['scope']=='optimizer_loop'
    assert timing['elapsed_seconds']==1830 and timing['timed_steps']==60
    assert timing['mean_step_seconds']==30.5 and timing['last_step_seconds']==60
    assert timing['recent_samples']==50 and timing['recent_mean_step_seconds']==35.5
    assert timing['remaining_steps']==100 and timing['eta_seconds']==3550
    remote.atomic_json(root/'metrics/stop_training.json',dict(stop_after_step=100))
    assert remote.detail(dict(run=root.name,arm='baseline'))['timing']['eta_seconds']==1420


def test_training_timing_does_not_invent_missing_history_or_waiting_eta(remote):
    root=make_job(remote)
    request=dict(run=root.name,arm='baseline')
    timing=remote.detail(request)['timing']
    assert timing['elapsed_seconds'] is None and timing['eta_seconds'] is None
    row=dict(kind='train',step=1,epoch=0,loss=1.,seconds=2.)
    metrics=root/'baseline/metrics/metrics.jsonl'
    metrics.write_text(json.dumps(row)+'\n'+json.dumps(dict(row,step=2,seconds=float('nan')))+'\n')
    remote.atomic_json(root/'baseline/status.json',dict(status='queued',step=2,total_steps=1000))
    timing=remote.detail(request)['timing']
    assert timing['elapsed_seconds']==2. and timing['timed_steps']==1
    assert timing['last_step_seconds'] is None and timing['eta_seconds'] is None
    remote.atomic_json(root/'baseline/status.json',dict(status='completed',step=2,training_complete=True))
    timing=remote.detail(request)['timing']
    assert timing['remaining_steps']==0 and timing['eta_seconds']==0


def test_stop_writes_only_selected_run_control(remote):
    root=make_job(remote)
    result=remote.stop(dict(run='task0_test',arm='baseline',step=3000))
    assert result['stop_after_step']==3000
    assert remote.read_json(root/'metrics/stop_training.json')['stop_after_step']==3000
    with pytest.raises(ValueError):
        remote.stop(dict(run='task0_test',arm='baseline',step=1000))


def test_stop_at_next_checkpoint_is_calculated_from_latest_state(remote):
    root=make_job(remote)
    remote.atomic_json(root/'baseline/status.json',dict(status='training',step=2417,total_steps=5000))
    result=remote.stop(dict(run=root.name,arm='baseline',next_checkpoint=True))
    assert result['stop_after_step']==3000
    assert remote.read_json(root/'metrics/stop_training.json')['stop_after_step']==3000


def test_short_training_can_save_and_stop_before_periodic_checkpoint(remote):
    root=make_job(remote)
    plan=remote.read_json(root/'config/ddp_launch.json')
    plan['validation_enabled']=False
    remote.atomic_json(root/'config/ddp_launch.json',plan)
    remote.atomic_json(root/'baseline/status.json',dict(status='training',step=120,total_steps=5000))
    result=remote.stop(dict(run=root.name,arm='baseline',step=300))
    assert result['stop_after_step']==300
    for target in (120, 0, 5001, 300.5, True):
        with pytest.raises(ValueError):
            remote.stop(dict(run=root.name,arm='baseline',step=target))


def test_delete_training_record_preserves_checkpoints_and_other_runs(remote):
    root=make_job(remote)
    remote.atomic_json(root/'baseline/status.json',dict(status='completed',step=3000,training_complete=True))
    remote.atomic_json(root/'baseline/validation_status.json',dict(status='completed'))
    other=remote.TRAINING/'other_training'
    remote.atomic_json(other/'config/ddp_launch.json',dict(commands={'baseline':{}}))
    remote.atomic_json(other/'baseline/status.json',dict(status='training',step=100))
    before={str(path.relative_to(root)):path.read_bytes() for path in root.rglob('*') if path.is_file()}
    assert next(row for row in remote.jobs()['jobs'] if row['run']==root.name)['can_delete']
    result=remote.delete(dict(run=root.name,arm='baseline'))
    assert result['status']=='deleted' and result['files_preserved']
    assert all(row['run']!=root.name for row in remote.jobs()['jobs'])
    assert [row['run'] for row in remote.jobs()['jobs']]==['other_training']
    assert all((root/path).read_bytes()==data for path,data in before.items())
    assert remote.detail(dict(run=root.name,arm='baseline'))['checkpoints'][0]['step']==1000


@pytest.mark.parametrize('status,validation', [('training',{}),('queued',{}),('initializing',{}),
    ('saving_checkpoint',{}),('completed',{'status':'validating'})])
def test_delete_refuses_active_training_or_validation(remote,status,validation):
    root=make_job(remote)
    remote.atomic_json(root/'baseline/status.json',dict(status=status,step=1000))
    remote.atomic_json(root/'baseline/validation_status.json',validation)
    assert not remote.jobs()['jobs'][0]['can_delete']
    with pytest.raises(ValueError,match='停止|结束'):
        remote.delete(dict(run=root.name,arm='baseline'))
    assert len(remote.jobs()['jobs'])==1


def make_waiting_queue(remote):
    jobs=[]
    for arm in ['baseline','memory_v1']:
        run='pending_'+arm
        root=remote.TRAINING/run
        remote.atomic_json(root/'config/web_launch.json',dict(arm=arm,gpus=[0],steps=3000))
        remote.atomic_json(root/arm/'status.json',dict(status='queued',step=0,total_steps=3000))
        jobs.append(dict(run=run,arm=arm,output=str(root)))
    queue=remote.TRAINING/'queues/pending'
    remote.atomic_json(queue/'config.json',dict(jobs=jobs))
    remote.atomic_json(queue/'status.json',dict(status='waiting_for_gpus',jobs=jobs,pid=123))
    remote.atomic_json(queue/'process.json',dict(pid=123,argv=['python','-u',
        str(remote.PROJECT/'deployments/snapshot/train/joint_task_queue.py'),'--config',str(queue/'config.json')]))
    return queue,jobs


def test_pending_queue_exposes_explicit_paired_cancel_scope(remote):
    _, jobs=make_waiting_queue(remote)
    rows=remote.jobs()['jobs']
    assert len(rows)==2 and all(row['can_cancel'] and not row['can_stop'] for row in rows)
    assert rows[0]['queue']['affected_runs']==[job['run'] for job in jobs]


def register_tasks(remote,monkeypatch,tasks=(0,34,40)):
    sources=[]
    for task in tasks:
        root=remote.PROJECT/f'data_task{task}'
        ids=[task*10000+e for e in range(200)]
        remote.atomic_json(root/'preparation/data_split.json',dict(task=task,seed=42,train=ids[:180],validation=ids[180:]))
        remote.atomic_json(root/'meta/info.json',dict(fps=30,chunks_size=10000,features={},data_path='data/{episode_chunk}/x',video_path='videos/{episode_chunk}/x'))
        (root/'meta/episodes.jsonl').write_text(''.join(json.dumps(dict(episode_index=e,length=1000))+'\n' for e in ids))
        for name in ['modality.json','stats.json','relative_stats.json','tasks.jsonl']:
            (root/'meta'/name).write_text('{}')
        for kind in ['data','videos']:
            (root/kind/f'chunk-{task:04d}').mkdir(parents=True)
        sources.append(dict(task=task,dataset=str(root),split=str(root/'preparation/data_split.json')))
    monkeypatch.setattr(remote,'catalog',lambda:dict(datasets=sources))
    return sources


def test_joint_preview_custom_split_and_no_validation(remote,monkeypatch):
    register_tasks(remote,monkeypatch)
    request=dict(remote.request,tasks=[40,0,34],validation_percent=20)
    plan=remote.prepare(request)
    assert plan['tasks']==[0,34,40] and plan['task']=='0_34_40'
    assert plan['train_episodes']==480 and plan['validation_episodes']==120
    split=plan['split_value']
    assert not set(split['train']) & set(split['validation'])
    assert all(len(part['train'])==160 and len(part['validation'])==40 for part in split['per_task'].values())
    assert remote.prepare(request)['split_value']==split
    assert remote.prepare(dict(request,seed=43))['split_value']['validation']!=split['validation']
    no_validation=remote.prepare(dict(request,validation_enabled=False,validation_gpu=None))
    assert no_validation['train_episodes']==600 and not no_validation['split_value']['validation']
    argv=remote.command(no_validation,remote.LIVE,remote.TRAINING/'new')
    assert '--no-validation' in argv and '--async-validation-gpu' not in argv
    monkeypatch.setattr(remote,'compute_pids',lambda gpus:[123] if 3 in gpus else [])
    request.update(gpus=[0],global_windows_per_step=1,validation_gpu=3)
    remote.prepare(dict(request,validation_enabled=False))
    with pytest.raises(ValueError,match='现有进程'):
        remote.prepare(request)
    assert remote.prepare(dict(request,enqueue=True))['enqueue']


def test_joint_submission_links_data_and_queues_without_starting_training(remote,monkeypatch):
    sources=register_tasks(remote,monkeypatch)
    remote.atomic_json(remote.LIVE/'web_source_manifest.json',dict(digest='test'))
    remote.request_digest='test'
    launched=[]
    monkeypatch.setattr(remote.subprocess,'Popen',lambda argv,**kw:launched.append(argv) or SimpleNamespace(pid=12345))
    request=dict(remote.request,tasks=[0,34,40],enqueue=True,validation_enabled=False)
    job=remote.start(request)
    root=Path(job['output']);queue=remote.TRAINING/'queues'/job['run']
    assert job['run']=='V_1_baseline_task0_34_40_train01' and job['status']=='queued'
    assert launched[0][2].endswith('/train/web_training_queue.py')
    assert (root/'dataset/data/chunk-0034').is_symlink()
    assert remote.read_json(root/'dataset/meta/info.json')['total_episodes']==600
    assert remote.read_json(root/'config/data_split.json')['validation']==[]
    assert not (root/'baseline/validation_queue').exists()
    assert not (root/'status.json').exists()
    assert remote.read_json(queue/'config.json')['gpus']==[0,2,3]
    assert '--no-validation' in remote.read_json(queue/'config.json')['argv']
    assert remote.jobs()['jobs'][0]['can_cancel']
    assert len(sources)==3


def test_web_queue_waits_fifo_only_for_overlapping_gpus_and_dispatches_public_command(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'train'))
    queue=load('web_queue_test',ROOT/'train/web_training_queue.py')
    training=tmp_path/'training'
    first=training/'queues/z_first';second=training/'queues/a_second'
    output=training/'run'
    queue.atomic_json(first/'status.json',dict(status='queued',created_at='2026-01-01T00:00:00'))
    queue.atomic_json(first/'config.json',dict(gpus=[0]))
    queue.atomic_json(second/'status.json',dict(status='queued',created_at='2026-01-01T00:00:01',jobs=[]))
    queue.atomic_json(second/'process.json',dict(pid=os.getpid()))
    command=[sys.executable,'-c','pass']
    config=dict(gpus=[0],output=str(output),arm='baseline',argv=command,allowed_processes={})
    queue.atomic_json(second/'config.json',config)
    queue.atomic_json(output/'config/web_process.json',dict(status='queued'))
    monkeypatch.setattr(queue,'compute_pids',lambda gpus:[])
    assert queue.blockers(training,second,config)==([],[first.name])
    assert queue.blockers(training,first,dict(gpus=[0]))==([],[])
    assert queue.blockers(training,second,dict(config,gpus=[1]))==([],[])
    queue.atomic_json(first/'status.json',dict(status='completed'))
    launched=[]
    monkeypatch.setattr(queue.subprocess,'Popen',lambda argv,**kw:launched.append(argv) or SimpleNamespace(wait=lambda:0))
    queue.run(second/'config.json')
    assert launched==[command]
    assert queue.read_json(second/'status.json')['status']=='completed'
    assert queue.read_json(output/'config/web_process.json')['status']=='submitted'
    monkeypatch.setattr(queue,'compute_pids',lambda gpus:[123])
    monkeypatch.setattr(queue,'process_start',lambda pid:'new_identity')
    assert queue.blockers(training,second,dict(config,allowed_processes={'123':'old_identity'}))[0]==[123]
    assert queue.blockers(training,second,dict(config,allowed_processes={'123':'new_identity'}))[0]==[]


def test_web_queue_real_cpu_dispatch_waits_for_preceding_job(tmp_path):
    # Use a fake idle nvidia-smi; only the harmless CPU child below may run.
    tools=tmp_path/'bin';tools.mkdir()
    smi=tools/'nvidia-smi'
    smi.write_text('#!/bin/sh\ncase "$1" in --query-gpu=*) printf "0, fake-gpu\\n";; esac\n')
    smi.chmod(0o755)
    training=tmp_path/'training';queue=training/'queues/second';first=training/'queues/first'
    remote=load('queue_integration_remote',ROOT/'train/dashboard_remote.py')
    output=training/'current';marker=tmp_path/'launched'
    remote.atomic_json(first/'config.json',dict(gpus=[0]))
    remote.atomic_json(first/'status.json',dict(status='queued',created_at='2026-01-01T00:00:00'))
    remote.atomic_json(queue/'status.json',dict(status='queued',created_at='2026-01-01T00:00:01',jobs=[]))
    remote.atomic_json(queue/'config.json',dict(output=str(output),arm='baseline',gpus=[0],allowed_processes={},
        argv=[sys.executable,'-c',f'from pathlib import Path;Path({str(marker)!r}).touch()']))
    remote.atomic_json(output/'config/web_process.json',dict(status='queued'))
    child=subprocess.Popen([sys.executable,str(ROOT/'train/web_training_queue.py'),'--config',str(queue/'config.json')],
        env=dict(os.environ,PATH=str(tools)+os.pathsep+os.environ['PATH']),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        remote.atomic_json(queue/'process.json',dict(pid=child.pid))
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if remote.read_json(queue/'status.json').get('waiting_for_runs'):
                break
            time.sleep(.05)
        else:
            pytest.fail('Queue did not record its pending predecessor')
        assert not marker.exists()
        remote.atomic_json(first/'status.json',dict(status='completed'))
        stdout,stderr=child.communicate(timeout=10)
        assert child.returncode==0,stdout+stderr
        assert marker.is_file()
        assert remote.read_json(queue/'status.json')['status']=='completed'
    finally:
        if child.poll() is None:
            child.terminate();child.wait(timeout=3)


def test_no_validation_saved_models_are_available_without_validation_queue(remote):
    root=make_job(remote)
    remote.atomic_json(root/'baseline/checkpoints/latest.json',dict(step=2000))
    checkpoints=remote.detail(dict(run=root.name,arm='baseline'))['checkpoints']
    assert [c['step'] for c in checkpoints]==[2000,1000]


def test_cancel_waiting_queue_terminates_only_controller_and_keeps_records(remote,monkeypatch):
    queue,jobs=make_waiting_queue(remote)
    terminated=[]
    monkeypatch.setattr(remote,'terminate_waiting_controller',lambda path:terminated.append(path))
    result=remote.cancel(dict(run=jobs[0]['run'],arm=jobs[0]['arm']))
    assert result['status']=='cancelled' and terminated==[queue]
    assert result['affected_runs']==[job['run'] for job in jobs]
    assert remote.read_json(queue/'status.json')['status']=='cancelled'
    for job in jobs:
        root=remote.TRAINING/job['run']
        assert (root/'config/web_launch.json').exists()
        assert remote.read_json(root/job['arm']/'status.json')['status']=='cancelled'
        assert not (root/'metrics/stop_training.json').exists()
    assert all(not row['can_cancel'] and not row['can_stop'] for row in remote.jobs()['jobs'])


def test_cancel_refuses_started_job_and_stop_refuses_queued_job(remote,monkeypatch):
    queue,jobs=make_waiting_queue(remote)
    with pytest.raises(ValueError,match='排队'):
        remote.stop(dict(run=jobs[0]['run'],arm=jobs[0]['arm'],step=1000))
    remote.atomic_json(remote.TRAINING/jobs[0]['run']/jobs[0]['arm']/'status.json',dict(status='training',step=1))
    monkeypatch.setattr(remote,'terminate_waiting_controller',lambda path:pytest.fail('Cannot signal a running queue'))
    with pytest.raises(ValueError,match='开始'):
        remote.cancel(dict(run=jobs[0]['run'],arm=jobs[0]['arm']))
    assert remote.read_json(queue/'status.json')['status']=='waiting_for_gpus'


@pytest.mark.parametrize('launch_race',[False,True])
def test_controller_cancellation_with_real_processes_preserves_other_processes(remote,monkeypatch,launch_race):
    queue,jobs=make_waiting_queue(remote)
    record=remote.read_json(queue/'process.json')
    script=Path(record['argv'][2])
    script.parent.mkdir(parents=True)
    script.write_text('import time\ntime.sleep(60)\n')
    argv=[sys.executable,'-u',str(script),'--config',str(queue/'config.json')]
    controller=subprocess.Popen(argv)
    unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
    try:
        state=remote.read_json(queue/'status.json')
        remote.atomic_json(queue/'status.json',dict(state,pid=controller.pid))
        remote.atomic_json(queue/'process.json',dict(pid=controller.pid,argv=argv))
        if launch_race:
            def now_started(directory):
                raise ValueError('队列已经开始训练')
            monkeypatch.setattr(remote,'queue_pending',now_started)
            with pytest.raises(ValueError,match='开始训练'):
                remote.terminate_waiting_controller(queue)
            assert controller.poll() is None
            proc_state=Path(f'/proc/{controller.pid}/stat').read_text().rsplit(')',1)[1].split()[0]
            assert proc_state not in ('T','t')
        else:
            result=remote.cancel(dict(run=jobs[0]['run'],arm=jobs[0]['arm']))
            assert result['status']=='cancelled'
            assert controller.wait(timeout=3)==-15
        assert unrelated.poll() is None
    finally:
        for process in [controller,unrelated]:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=3)


def test_cancellation_refuses_reused_or_unrelated_pid(remote,monkeypatch):
    queue,_=make_waiting_queue(remote)
    record=remote.read_json(queue/'process.json')
    remote.atomic_json(queue/'process.json',dict(record,pid=os.getpid()))
    monkeypatch.setattr(remote.signal,'pidfd_send_signal',lambda *args:pytest.fail('Unrelated process must never be signalled'))
    with pytest.raises(ValueError,match='PID与记录不一致'):
        remote.terminate_waiting_controller(queue)


def test_running_job_has_save_stop_but_no_queue_cancel(remote):
    make_job(remote)
    row=remote.jobs()['jobs'][0]
    assert row['can_stop'] and not row['can_cancel']


def test_training_page_hides_cancelled_jobs_and_leaves_cancelled_detail():
    node=Path('/home/1024201092WYH/.nvm/versions/node/v22.23.1/bin/node')
    if not node.exists():
        pytest.skip('Node.js required for training UI behavior check')
    script=(ROOT/'train/dashboard.js').read_text().replace('window.trainingDashboard =', '''
    window.trainingTest = {
      refresh,
      setup() { chosen={run:'cancelled_pair',arm:'baseline'}; view='detail'; lastOptionsAt=Date.now(); },
      state() { return {jobs,chosen,view}; }
    };
    window.trainingDashboard =''')
    harness='''
const assert=require('node:assert/strict');
const elements={};
global.window={}; global.setInterval=()=>{};
global.document={getElementById(id) { return elements[id] ||= {
    textContent:'',innerHTML:'',hidden:false,scrollTop:0,
    querySelectorAll:()=>[],setAttribute:()=>{}
}; }};
global.fetch=async()=>({ok:true,json:async()=>({jobs:[
    {run:'cancelled_pair',arm:'baseline',status:'cancelled'},
    {run:'cancelled_memory',arm:'memory_v1',status:'cancelled'},
    {run:'still_training',arm:'baseline',status:'training',step:1000,max_steps:3000},
    {run:'completed_training',arm:'memory_v1',status:'completed',step:3000,max_steps:3000}
]})});
'''
    assertions='''
(async()=>{
  window.trainingTest.setup(); await window.trainingTest.refresh();
  const state=window.trainingTest.state();
  assert.deepEqual(state.jobs.map(j=>j.run),['still_training','completed_training']);
  assert.equal(document.getElementById('train-count').textContent,'2 条');
  assert.ok(!document.getElementById('train-jobs').innerHTML.includes('cancelled_pair'));
  assert.ok(!document.getElementById('train-jobs').innerHTML.includes('cancelled_memory'));
  assert.equal(state.view,'new');
  assert.equal(document.getElementById('train-monitor').hidden,true);
  assert.equal(document.getElementById('train-config').hidden,false);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    result=subprocess.run([str(node)],input=harness+script+assertions,text=True,capture_output=True)
    assert result.returncode==0,result.stderr


def test_training_form_submits_joint_tasks_queue_and_optional_validation():
    node=Path('/home/1024201092WYH/.nvm/versions/node/v22.23.1/bin/node')
    if not node.exists():
        pytest.skip('Node.js required for training form behavior check')
    script=(ROOT/'train/dashboard.js').read_text().replace('window.trainingDashboard =', '''
    options={datasets:[0,34,40].map(task=>({task,ready:true})),checkpoints:[],gpus:[]};
    window.trainingTest={parameters,datasetHint};
    window.trainingDashboard =''')
    harness='''
const assert=require('node:assert/strict'), elements={};
global.window={};global.setInterval=()=>{};
global.document={getElementById(id) {return elements[id] ||= {
  value:'',checked:false,selectedOptions:[],textContent:'',hidden:false,
  querySelectorAll:selector=>selector==='input:checked'?[{value:'0'},{value:'1'}]:[]
};}};
'''
    assertions='''
const el=id=>document.getElementById(id);
el('train-dataset').selectedOptions=[{value:'0'},{value:'1'},{value:'2'}];
for(const [key,value] of Object.entries({version:'V_1',checkpoint:'/base',arm:'baseline',steps:'3000',
    windows:'4',microbatch:'4',lr:'0.00001',seed:'42',method:'',
    'method-description':'','notes':'Task0/34/40 比较记忆容量','validation-gpu':'3','validation-percent':'20',
    'fine-tuning':'action_full','validation-interval':'3000'})) el('train-'+key).value=value;
el('train-enqueue').checked=true;
let request=window.trainingTest.parameters();window.trainingTest.datasetHint();
assert.deepEqual(request.tasks,[0,34,40]);assert.equal(request.enqueue,true);
assert.equal(request.notes,'Task0/34/40 比较记忆容量');
assert.equal(request.fine_tuning_mode,'action_full');assert.equal(request.validation_interval,3000);
assert.equal(request.disable_early_stopping,true);
assert.equal(request.validation_enabled,false);
assert.ok(!('validation_gpu' in request));assert.ok(!('validation_percent' in request));
assert.equal(el('train-validation-gpu').disabled,true);
assert.equal(el('train-validation-gpu-field').hidden,true);
assert.ok(el('train-split').textContent.includes('600 集训练 / 0 集验证'));
el('train-validation-enabled').checked=true;
request=window.trainingTest.parameters();window.trainingTest.datasetHint();
assert.equal(request.validation_percent,20);assert.equal(request.validation_gpu,3);
assert.equal(el('train-validation-gpu').disabled,false);
assert.equal(el('train-validation-gpu-field').hidden,false);
assert.ok(el('train-split').textContent.includes('480 集训练 / 120 集验证'));
el('train-fine-tuning').value='lora';
el('train-freeze-vision').value='true';
el('train-freeze-action-projector').value='true';
el('train-lora-targets').value='llm,action_dit';
el('train-lora-rank').value='8';el('train-lora-alpha').value='16';
el('train-lora-dropout').value='0.05';el('train-llm-last-n-layers').value='2';
el('train-gradient-checkpointing').checked=true;
el('train-weight-decay').value='0.03';el('train-max-grad-norm').value='0.5';
request=window.trainingTest.parameters();
assert.equal(request.fine_tuning_mode,'lora');assert.equal(request.freeze_vision,true);
assert.equal(request.freeze_action_projector,true);assert.ok(!('freeze_llm' in request));
assert.equal(request.lora_targets,'llm,action_dit');assert.equal(request.lora_rank,8);
assert.equal(request.lora_alpha,16);assert.equal(request.lora_dropout,.05);
assert.equal(request.llm_last_n_layers,2);assert.equal(request.gradient_checkpointing,true);
assert.equal(request.weight_decay,.03);assert.equal(request.max_grad_norm,.5);
// Manual budgets remain usable; automatic mode must not present a stale budget
// when the selected resources cannot produce a recommendation.
assert.equal(el('train-steps').value,'3000');
el('train-auto-recommend').checked=true;
window.trainingTest.datasetHint();
assert.equal(el('train-steps').value,'');
assert.ok(el('train-recommendation-summary').textContent.length>0);
assert.equal(el('train-recommendation-details').hidden,true);
'''
    result=subprocess.run([str(node)],input=harness+script+assertions,text=True,capture_output=True)
    assert result.returncode==0,result.stderr


def test_method_metadata_and_legacy_numbering_keep_retries_distinct(remote, monkeypatch):
    from run_identity import reserve_training, training_identity
    old=remote.TRAINING/'V_2_mem_task0_r3'
    remote.atomic_json(old/'config/web_launch.json',dict(version='V_2',arm='memory_v1',method='memory',task=0))
    remote.atomic_json(old/'config/data_split.json',dict(task=0))
    (old/'memory_v1').mkdir()
    run,number=reserve_training(remote.TRAINING,remote.PROJECT/'deployments','V_2',0,'memory')
    assert run=='V_2_memory_task0_train04' and number==4
    assert old.is_dir()
    assert reserve_training(remote.TRAINING,remote.PROJECT/'deployments','V_2',0,'memory')[0].endswith('train05')
    assert reserve_training(remote.TRAINING,remote.PROJECT/'deployments','V_2',0,'baseline')[0].endswith('train01')
    plan=remote.prepare(dict(remote.request,arm='memory_v1',method='ablation_no_perception',method_description='Disabled in frozen code'))
    assert plan['run_name']=='V_1_ablation_no_perception_task0_trainNN'
    with pytest.raises(ValueError,match='说明'):
        remote.prepare(dict(remote.request,arm='memory_v1',method='ablation_no_perception'))
    with pytest.raises(ValueError,match='不一致'):
        remote.prepare(dict(remote.request,method='memory'))


def test_manifest_excludes_environment_outputs_and_changes_with_source(tmp_path):
    m=load('training_dashboard_test',ROOT/'train/dashboard.py')
    code=tmp_path/'src/v1_groot_memoryvla/model.py'
    code.parent.mkdir(parents=True)
    code.write_text('a')
    env=code.parent/'.venv/site.py'
    env.parent.mkdir()
    env.write_text('large environment')
    (tmp_path/'outputs').mkdir()
    (tmp_path/'outputs/large').write_text('weights')
    manager=m.TrainingManager(tmp_path)
    try:
        first=manager.manifest()
        assert list(first['files'])==['src/v1_groot_memoryvla/model.py']
        code.write_text('b')
        assert manager.manifest()['digest']!=first['digest']
    finally: manager.close()


def test_http_training_page_and_preview_without_launching():
    sys.path.insert(0,str(ROOT/'evaluation/scripts'))
    viewer=load('training_dashboard_viewer',ROOT/'evaluation/scripts/view_results.py')
    class Handler(viewer.Handler):
        training_manager=SimpleNamespace(options=lambda:dict(gpus=[]),prepare=lambda value:dict(checked=value),
            prepare_dataset=lambda value:dict(preparing=value['task']),cancel=lambda value:dict(cancelled=value['run']),
            delete=lambda value:dict(deleted=value['run'],files_preserved=True),
            resume=lambda value:dict(resumed=value['run'],steps=value['steps']))
    server=viewer.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base='http://127.0.0.1:'+str(server.server_address[1])
    try:
        assert json.load(urlopen(base+'/api/training/options'))=={'gpus':[]}
        request=Request(base+'/api/training/preview',data=b'{"steps":3000}',headers={'Content-Type':'application/json'})
        assert json.load(urlopen(request))=={'checked':{'steps':3000}}
        request=Request(base+'/api/training/dataset',data=b'{"task":34}',headers={'Content-Type':'application/json'})
        assert json.load(urlopen(request))=={'preparing':34}
        request=Request(base+'/api/training/cancel',data=b'{"run":"pending_baseline","arm":"baseline"}',headers={'Content-Type':'application/json'})
        assert json.load(urlopen(request))=={'cancelled':'pending_baseline'}
        request=Request(base+'/api/training/delete',data=b'{"run":"finished_baseline","arm":"baseline"}',headers={'Content-Type':'application/json'})
        assert json.load(urlopen(request))=={'deleted':'finished_baseline','files_preserved':True}
        request=Request(base+'/api/training/resume',data=b'{"run":"finished_baseline","arm":"baseline","steps":20000}',headers={'Content-Type':'application/json'})
        assert json.load(urlopen(request))=={'resumed':'finished_baseline','steps':20000}
        assert b'train-form' in urlopen(base+'/training.html').read()
        script=urlopen(base+'/training.js').read().decode()
        prefix='window.__grootTrainingPageHTML = '
        assert script.startswith(prefix) and 'trainingDashboard' in script
        template=json.JSONDecoder().raw_decode(script[len(prefix):])[0]
        assert template==(ROOT/'train/dashboard.html').read_text()
        assert 'train-checkpoint-choice' in template
    finally:
        server.shutdown();server.server_close();thread.join()


def test_http_device_routing_keeps_same_run_names_on_separate_hosts():
    from urllib.error import HTTPError
    sys.path.insert(0,str(ROOT/'evaluation/scripts'))
    viewer=load('training_device_viewer',ROOT/'evaluation/scripts/view_results.py')
    def manager(label):
        return SimpleNamespace(options=lambda:dict(label=label),
            prepare=lambda value:dict(label=label,checked=value))
    class Handler(viewer.Handler):
        training_manager=manager('a100')
        training_devices={'a100':training_manager,'a6000':manager('a6000')}
    server=viewer.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_address[1]}'
    try:
        assert json.load(urlopen(base+'/api/training/options'))['label']=='a100'
        assert json.load(urlopen(base+'/api/training/options?device=a6000'))['label']=='a6000'
        request=Request(base+'/api/training/preview',data=b'{"device":"a6000","steps":1000}',
                        headers={'Content-Type':'application/json'})
        assert json.load(urlopen(request))==dict(label='a6000',checked=dict(steps=1000))
        with pytest.raises(HTTPError) as error:
            urlopen(base+'/api/training/options?device=unknown')
        assert error.value.code==400
    finally:
        server.shutdown();server.server_close();thread.join()


def test_optional_device_paths_sync_state_and_single_gpu_validation(remote,tmp_path,monkeypatch):
    module=load('training_optional_device',ROOT/'train/dashboard.py')
    device=dict(label='RTX 6000 Ada',single_gpu=True,paths=dict(project='/data/project',training='/data/training'))
    manager=module.TrainingManager(tmp_path,'user@optional','a6000',device)
    try:
        assert manager.live=='/data/project/deployments/web_live'
        assert manager.record==tmp_path/'outputs/shared/training_dashboard/a6000/sync_state.json'
        (tmp_path/'train').mkdir()
        (tmp_path/'train/dashboard_remote.py').write_text((ROOT/'train/dashboard_remote.py').read_text())
        code=[]
        monkeypatch.setattr(manager,'_ssh',lambda command,source,**kw:code.append(source) or '{}')
        manager._remote('catalog')
        assert "'device_id': 'a6000'" in code[0] and "'project': '/data/project'" in code[0]
    finally:manager.close()
    monkeypatch.setattr(remote,'_device',dict(single_gpu=True,device_id='a6000'))
    with pytest.raises(ValueError,match='只有一张卡'):
        remote.prepare(remote.request)
    plan=remote.prepare(dict(remote.request,gpus=[0],validation_enabled=False))
    assert plan['device_id']=='a6000' and plan['validation_gpu'] is None
    assert '--no-validation' in remote.command(plan,remote.LIVE,tmp_path/'new')


def test_all_tasks_visible_with_readiness_and_preparation_state(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'evaluation/scripts'))
    module=load('training_all_tasks',ROOT/'train/dashboard.py')
    manager=module.TrainingManager(tmp_path)
    monkeypatch.setattr(manager,'_remote',lambda *args,**kwargs:dict(datasets=[
        dict(task=34,dataset='/data/task34',split='/data/task34/split.json',train_episodes=180,validation_episodes=20)],gpus=[]))
    try:
        rows=manager.options()['datasets']
        assert len(rows)==50 and [r['task'] for r in rows]==list(range(50))
        assert rows[34]['ready'] and rows[34]['task_name']=='hanging_pictures'
        assert rows[34]['preparation']['status']=='completed'
        assert rows[34]['preparation']['verified_on']=='A100'
        assert not rows[40]['ready'] and not rows[40]['can_prepare']
        assert rows[40]['dataset'] is None
        with pytest.raises(ValueError):
            manager.prepare_dataset(dict(task=50))
    finally:manager.close()


def test_remote_catalog_discovers_prepared_dataset_before_any_training(remote,monkeypatch):
    module=load('training_discovery',ROOT/'train/dashboard_remote.py')
    module.DATASETS=remote.PROJECT/'datasets'
    module.TRAINING=remote.TRAINING
    module.OFFICIAL=remote.PROJECT/'official_missing'
    monkeypatch.setattr(module,'gpu_info',lambda:[])
    dataset=module.DATASETS/'b1k2025_groot_task40_v1_200'
    module.atomic_json(dataset/'preparation/ready.json',dict(status='completed',task=40,dataset=str(dataset),
        split=str(dataset/'preparation/data_split.json'),train_episodes=180,validation_episodes=20))
    rows=module.catalog()['datasets']
    assert len(rows)==1 and rows[0]['task']==40


def test_queued_job_shows_dependencies_and_gpu_wait_reason(remote):
    root=remote.TRAINING/'V_1_memory_task0_34_40_train01'
    remote.atomic_json(root/'config/data_split.json',dict(tasks=[0,34,40]))
    remote.atomic_json(root/'config/ddp_launch.json',dict(steps_each=18000,
        commands={'memory_v1':dict(CUDA_VISIBLE_DEVICES='2,3',argv=[])}))
    remote.atomic_json(root/'memory_v1/status.json',dict(status='queued',step=0))
    remote.atomic_json(remote.TRAINING/'queues/joint/status.json',dict(status='waiting_for_gpus',
        jobs=[dict(run=root.name)],unexpected_pids=[123],
        dependencies=[dict(task=40,arm='memory_v1',step=3000,complete=True)]))
    rows=remote.jobs()['jobs']
    assert rows[0]['status']=='waiting_for_gpus'
    assert rows[0]['queue']['unexpected_pids']==[123]
    assert rows[0]['task_id']=='0_34_40'


def test_task_queue_submits_four_matched_3000_step_runs_once(tmp_path,monkeypatch):
    module=load('training_task_queue',ROOT/'train/task_queue.py')
    source=tmp_path/'source'
    source.mkdir()
    (source/'web_source_manifest.json').write_text('{"digest":"fixed-source"}')
    previous=tmp_path/'previous'
    previous.joinpath('memory_v1/checkpoints/step-003000').mkdir(parents=True)
    (previous/'memory_v1/checkpoints/step-003000/config.json').write_text('{}')
    (previous/'memory_v1/status.json').write_text('{"training_complete":true}')
    (previous/'status.json').write_text('{"status":"completed"}')
    datasets=tmp_path/'datasets'
    calls=[]
    fake=SimpleNamespace(DATASETS=datasets,OFFICIAL=tmp_path/'official',compute_pids=lambda gpus:[11])
    for task in [34,40]:
        ready=datasets/f'b1k2025_groot_task{task}_v1_200/preparation/ready.json'
        ready.parent.mkdir(parents=True)
        ready.write_text(json.dumps(dict(dataset=f'/data/task{task}',split=f'/data/task{task}/split.json')))
    def start(request):
        calls.append(request)
        name=f"V_1_task{request['task']}_{request['arm']}"
        root=tmp_path/name
        (root/request['arm']).mkdir(parents=True)
        (root/'status.json').write_text('{"status":"completed"}')
        (root/request['arm']/'status.json').write_text('{"step":3000,"training_complete":true}')
        return dict(run=name,arm=request['arm'],output=str(root),pid=0)
    fake.start=start
    monkeypatch.setattr(module.importlib.util,'spec_from_file_location',lambda *args:SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None)))
    monkeypatch.setattr(module.importlib.util,'module_from_spec',lambda spec:fake)
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(source=str(source),after_run=str(previous),tasks=[34,40],gpus=[1,2,3],global_windows_per_step=3,
        validation_gpu=0,allowed_existing_pids=[11],task_names={'34':'hanging_pictures','40':'make_microwave_popcorn'})))
    module.run(config)
    assert [(p['task'],p['arm']) for p in calls]==[(34,'baseline'),(34,'memory_v1'),(40,'baseline'),(40,'memory_v1')]
    assert all(p['steps']==3000 and p['global_windows_per_step']==3 and p['seed']==42 for p in calls)
    assert fake.LIVE==source and fake.request_digest=='fixed-source'
    assert json.loads((tmp_path/'status.json').read_text())['status']=='completed'
    with pytest.raises(ValueError,match='already'):
        module.run(config)


def test_dashboard_reload_observes_evaluation_without_stopping_it(tmp_path):
    sys.path.insert(0,str(ROOT/'evaluation/scripts'))
    viewer=load('training_reload_viewer',ROOT/'evaluation/scripts/view_results.py')
    pid=os.getpid()
    start=Path(f'/proc/{pid}/stat').read_text().split()[21]
    job=dict(pid=pid,process_start=start,output=str(tmp_path),returncode=None)
    assert viewer.DetachedEvaluation(job).poll() is None
    (tmp_path/'results.json').write_text('{"status":"completed"}')
    assert viewer.DetachedEvaluation(dict(job,process_start='different-process')).poll()==0


def test_renaming_keeps_live_queue_paths_and_retires_aliases(tmp_path, monkeypatch):
    m=load('training_rename', ROOT/'train/migrate_run_names.py')
    training=tmp_path/'v1_groot_memoryvla/training'
    old=training/'V_1_task34_mem'
    old.joinpath('config').mkdir(parents=True)
    source='/deployments/V_1_task34_mem'
    (old/'config/ddp.json').write_text(json.dumps(dict(output=str(old),source=source)))
    queue=training/'queues/V_1_task34_40/status.json'
    m.save(queue,dict(status='training'))
    monkeypatch.setattr(m,'references',lambda p:[])
    record=tmp_path/'rename.json'
    state=m.migrate(training,record)
    new=training/'V_1_mem_task34'
    assert new.is_dir() and old.is_symlink() and old.resolve()==new
    assert state['aliases']['V_1_task34_mem']['queue']
    m.save(queue,dict(status='completed'))
    state=m.migrate(training,record)
    assert not old.exists() and not state['aliases']
    config=m.read(new/'config/ddp.json')
    assert config==dict(output=str(new),source=source)


def test_paired_run_split_preserves_weights_and_own_arm_paths(tmp_path,monkeypatch):
    m=load('training_pair_rename',ROOT/'train/migrate_run_names.py')
    training=tmp_path/'v1_groot_memoryvla/training'
    old=training/m.PAIR
    monkeypatch.setattr(m,'references',lambda p:[])
    for arm in ['baseline','memory_v1']:
        p=old/arm/'checkpoints/step-001000'
        p.mkdir(parents=True)
        (p/'model.safetensors').write_bytes(arm.encode())
        m.save(p/'latest.json',dict(path=str(p),source='/snapshot/original'))
    m.save(old/'config/ddp_launch.json',dict(commands=dict(baseline={},memory_v1={})))
    m.save(old/'status.json',dict(status='completed',selected_checkpoints={a:dict(path=str(old/a/'checkpoints/step-001000')) for a in ['baseline','memory_v1']}))
    m.migrate(training,tmp_path/'rename.json')
    for arm,name in [('baseline','V_1_bas_task21'),('memory_v1','V_1_mem_task21')]:
        root=training/name
        p=root/arm/'checkpoints/step-001000'
        assert (p/'model.safetensors').read_bytes()==arm.encode()
        assert m.read(p/'latest.json')==dict(path=str(p),source='/snapshot/original')
        assert list(m.read(root/'config/ddp_launch.json')['commands'])==[arm]
        assert list(m.read(root/'status.json')['selected_checkpoints'])==[arm]
    assert not old.exists()
