"""A100 operations for the web workbench; launches the canonical training CLI."""

import datetime
from collections import deque
import fcntl
import json
import math
import os
from pathlib import Path
import re
import random
import shlex
import shutil
import signal
import subprocess
import sys
import time

if '__file__' in globals():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_identity import NAMING_SCHEMA, method_name, training_base, training_identity, reserve_training

# This RPC is also executed as source over SSH, without importing the model stack.
import importlib.util
_options_root = Path(__file__).resolve().parents[1] if '__file__' in globals() and Path(__file__).is_file() else Path('/nvme3/projects/memory_vla/deployments/web_live')
_options_file = _options_root/'src/v1_groot_memoryvla/experiment/finetuning_options.py'
if not _options_file.is_file():
    _options_file = Path(globals().get('REQUEST', {}).get('paths', {}).get('project', '/nvme3/projects/memory_vla'))/'deployments/web_live/src/v1_groot_memoryvla/experiment/finetuning_options.py'
_fine_spec = importlib.util.spec_from_file_location('training_finetuning_options', _options_file)
_fine_module = importlib.util.module_from_spec(_fine_spec)
_fine_spec.loader.exec_module(_fine_module)
resolve_finetuning = _fine_module.resolve_finetuning
FINETUNING_COMPONENTS = _fine_module.COMPONENTS

PROJECT = Path('/nvme3/projects/memory_vla')
TRAINING = Path('/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training')
LIVE = PROJECT / 'deployments/web_live'
PYTHON = PROJECT / 'envs/n1d6-p0/bin/python'
OFFICIAL = Path('/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k')
BASE_CHECKPOINT = OFFICIAL.parent/'GR00T-N1.6-3B'
DATASETS = Path('/nvme3/data_wyh/datasets')
# Paths come from the server's device registry, never from browser form fields.
_device = globals().get('REQUEST', {})
_paths = _device.get('paths', {})
PROJECT = Path(_paths.get('project', PROJECT))
TRAINING = Path(_paths.get('training', TRAINING))
LIVE = PROJECT / 'deployments/web_live'
PYTHON = Path(_paths.get('python', PYTHON))
OFFICIAL = Path(_paths.get('official', OFFICIAL))
BASE_CHECKPOINT = Path(_paths.get('base_checkpoint', BASE_CHECKPOINT))
DATASETS = Path(_paths.get('datasets', DATASETS))
DEVICE_LABEL = _device.get('device_label', 'A100')
WAITING_QUEUE_STATES = {'waiting_for_four_runs', 'waiting_for_gpus', 'queued'}


def run_name(version, task, arm):
    return training_base(version, task, method_name(arm))


def display_name(root, arm, task):
    identity = training_identity(root, arm, task)
    return training_base(identity['version'],identity['task_id'],identity['method'])+'_train%02d'%identity['train_number'] if identity['task_id'] is not None else root.name


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def gpu_info():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=8)
    rows = []
    for line in result.stdout.splitlines():
        index, name, total, used, utilization = map(str.strip, line.split(','))
        rows.append(dict(index=int(index), name=name, total_mb=int(total), used_mb=int(used), utilization=int(utilization)))
    return rows


def compute_pids(gpus):
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    owners = {u.strip(): int(i) for i, u in (line.split(',') for line in result.splitlines())}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    return sorted({int(pid) for line in apps.splitlines() if line.strip()
                   for gpu, pid in [line.split(',')] if owners[gpu.strip()] in gpus})


def dataset_training_profile(source):
    """Count real recurrent windows, including each episode's short tail."""
    episodes = Path(source['dataset'])/'meta/episodes.jsonl'
    if not episodes.is_file():
        return None
    rows = [json.loads(line) for line in episodes.read_text().splitlines() if line.strip()]
    lengths = {row['episode_index']: row['length'] for row in rows}
    split = read_json(Path(source['split']), {})
    ids = split.get('train', []) + split.get('validation', [])
    if not ids or any(e not in lengths or lengths[e] <= 0 for e in ids):
        return None
    return dict(total_episodes=len(ids), total_frames=sum(lengths[e] for e in ids),
        total_windows=sum(math.ceil(lengths[e]/256) for e in ids),
        train_windows=sum(math.ceil(lengths[e]/256) for e in split['train']),
        split_seed=split.get('seed', 42), validation_percent=100*len(split['validation'])/len(ids))


def checkpoint_training_profile(path):
    config = read_json(path/'config.json', {})
    kind = 'base' if path == BASE_CHECKPOINT else 'behavior' if path == OFFICIAL else 'trained'
    return dict(kind=kind, model_type=config.get('model_type'),
        memory_enabled=config.get('memory_enabled', False),
        memory_conditioning=config.get('memory_conditioning', 'residual'),
        memory_residual_readout=config.get('memory_residual_readout', 'delta'),
        memory_cognitive_residual_projection=config.get('memory_cognitive_residual_projection', False),
        memory_cognitive_residual_scope=config.get('memory_cognitive_residual_scope', 'eos'),
        memory_cognitive_qk_norm=config.get('memory_cognitive_qk_norm', 'none'),
        memory_length=config.get('memory_length', 16))


def catalog():
    datasets = {}
    for marker in sorted(DATASETS.glob('b1k2025_groot_task*_v1_200/preparation/ready.json')):
        data=read_json(marker)
        if data.get('status')=='completed' and Path(data['dataset']).is_dir():
            datasets[data['task']]=dict(task=data['task'],dataset=data['dataset'],split=data['split'],
                train_episodes=data['train_episodes'],validation_episodes=data['validation_episodes'])
    for file in sorted(TRAINING.glob('*/config/data_split.json')):
        if file.parents[1].is_symlink():
            continue
        split = read_json(file)
        if 'task' not in split:
            continue
        root = file.parents[1]
        plan = read_json(root / 'config/ddp_launch.json', {})
        for cmd in plan.get('commands', {}).values():
            argv = cmd['argv']
            dataset = argv[argv.index('--dataset')+1]
            if Path(dataset).is_dir() and split['task'] not in datasets:
                datasets[split['task']] = dict(task=split['task'], dataset=dataset, split=str(file),
                    train_episodes=len(split['train']), validation_episodes=len(split['validation']))
    checkpoints = []
    for path, label in [(BASE_CHECKPOINT, '官方 GR00T N1.6-3B · 通用预训练起点'),
                        (OFFICIAL, '官方 GR00T N1.6-BEHAVIOR1k · 已经 B1K 微调')]:
        if (path/'config.json').is_file():
            index=read_json(path/'model.safetensors.index.json',{})
            files=set(index.get('weight_map',{}).values())
            if files and all((path/name).is_file() for name in files):
                checkpoints.append(dict(path=str(path),label=label,training_profile=checkpoint_training_profile(path)))
    for file in sorted(TRAINING.glob('*/*/checkpoints/step-*/config.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        root=file.parents[3]
        if root.is_symlink():
            continue
        arm=file.parents[2].name
        task=read_json(root/'config/data_split.json',{}).get('task')
        checkpoints.append(dict(path=str(file.parent), label=display_name(root,arm,task)+' / '+file.parent.name,
            training_profile=checkpoint_training_profile(file.parent)))
    for source in datasets.values():
        source['training_profile'] = dataset_training_profile(source)
    return dict(datasets=[datasets[key] for key in sorted(datasets)], checkpoints=checkpoints, gpus=gpu_info(),
                python_path=str(PYTHON), live_source=str(LIVE))


def _path(value, label):
    if not isinstance(value, str) or not value.startswith('/') or any(ord(c) in (0, 10, 13) for c in value):
        raise ValueError(label + '必须为目标设备上的绝对路径')
    return Path(value)


def prepare(request):
    allowed = {'task', 'tasks', 'task_name', 'dataset', 'split', 'checkpoint', 'gpus', 'validation_gpu',
               'arm', 'steps', 'global_windows_per_step', 'microbatch', 'seed', 'learning_rate', 'share_existing', 'version',
               'method', 'method_description', 'validation_enabled', 'validation_percent', 'enqueue',
               'memory_length', 'memory_conditioning', 'memory_residual_readout', 'memory_cognitive_qk_norm', 'image_augmentation', 'fine_tuning_mode',
               'validation_interval', 'disable_early_stopping', 'notes', 'lora_targets', 'lora_rank', 'lora_alpha',
               'lora_dropout', 'llm_last_n_layers', 'gradient_checkpointing', 'weight_decay', 'max_grad_norm',
               'optimizer_eps', 'memory_cognitive_residual_projection', 'memory_cognitive_residual_scope',
               *('freeze_' + name for name in FINETUNING_COMPONENTS)}
    if not isinstance(request, dict) or set(request)-allowed:
        raise ValueError('不支持的训练参数')
    fine_tuning = resolve_finetuning(request)
    weight_decay, max_grad_norm = request.get('weight_decay', .01), request.get('max_grad_norm', 1.)
    if type(weight_decay) not in (int,float) or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('Weight decay须为非负有限数')
    if type(max_grad_norm) not in (int,float) or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError('梯度裁剪阈值须为正有限数')
    optimizer_eps = request.get('optimizer_eps', 1e-8)
    if type(optimizer_eps) not in (int,float) or not math.isfinite(optimizer_eps) or optimizer_eps <= 0:
        raise ValueError('AdamW epsilon须为正有限数')
    validation_interval = request.get('validation_interval', 1000)
    if type(validation_interval) is not int or validation_interval < 1000 or validation_interval % 1000:
        raise ValueError('验证间隔须为1000的正整数倍')
    disable_early_stopping = request.get('disable_early_stopping', False)
    if type(disable_early_stopping) is not bool:
        raise ValueError('自动早停必须是开关')
    for key in ['steps', 'global_windows_per_step', 'microbatch', 'seed']:
        if type(request.get(key)) is not int:
            raise ValueError(key + '必须为整数')
    gpus = request.get('gpus')
    if not isinstance(gpus, list) or not gpus or any(type(i) is not int for i in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError('请选择不重复的训练GPU')
    for key in ['validation_enabled','enqueue']:
        if type(request.get(key, True if key=='validation_enabled' else False)) is not bool:
            raise ValueError(key+'必须为开关')
    validation_enabled=request.get('validation_enabled',True)
    if _device.get('single_gpu') and validation_enabled:
        raise ValueError('该设备只有一张卡，请关闭独立 Episode 验证；训练后再单独评估权重')
    validation_gpu=request.get('validation_gpu') if validation_enabled else None
    if validation_enabled and type(validation_gpu) is not int:
        raise ValueError('开启验证时请选择验证GPU')
    percent=request.get('validation_percent',10) if validation_enabled else 0
    if type(percent) is not int or not (1<=percent<=50 if validation_enabled else percent==0):
        raise ValueError('验证比例应为1–50之间的整数百分比')
    selected=set(gpus+([validation_gpu] if validation_enabled else []))
    if not selected <= {g['index'] for g in gpu_info()}:
        raise ValueError('所选GPU不存在')
    tasks=request.get('tasks',[request.get('task')])
    if not isinstance(tasks,list) or not tasks or any(type(t) is not int or not 0<=t<50 for t in tasks) or len(set(tasks))!=len(tasks):
        raise ValueError('请选择不重复的任务编号0–49')
    tasks=sorted(tasks)
    task_label=tasks[0] if len(tasks)==1 else '_'.join(map(str,tasks))
    if request.get('arm') not in ['baseline', 'memory_v1']:
        raise ValueError('请选择Baseline或MemoryVLA v1')
    version=request.get('version','V_1')
    method=method_name(request['arm'],request.get('method'))
    if (request['arm']=='baseline' and method!='baseline') or (request['arm']=='memory_v1' and method not in ('memory',) and not method.startswith('ablation_')):
        raise ValueError('方法名称与实际模型类型不一致')
    description=request.get('method_description','')
    notes=request.get('notes','')
    if not isinstance(notes,str) or len(notes)>2000:
        raise ValueError('训练备注须为不超过2000字的文本')
    if not isinstance(description,str) or len(description)>2000:
        raise ValueError('方法说明格式不正确')
    if method.startswith('ablation_') and not description.strip():
        raise ValueError('消融需要填写实际代码或配置改动说明；名称不会自动关闭模块')
    name_base=training_base(version,task_label,method)
    if request['steps'] < 1000 or request['steps'] % 1000:
        raise ValueError('当前保存周期为1000，请填写1000的整数倍步数')
    if not 1 <= request['global_windows_per_step'] <= 180 or request['global_windows_per_step'] % len(gpus):
        raise ValueError('全局窗口数必须能被训练卡数整除，且在1–180之间')
    if not 1 <= request['microbatch'] <= 32 or not 0 <= request['seed'] < 2**32:
        raise ValueError('microbatch或seed超出范围')
    lr = request.get('learning_rate')
    if type(lr) not in [int, float] or not math.isfinite(lr) or not 0 < lr <= .01:
        raise ValueError('学习率必须在0到0.01之间')
    if type(request.get('share_existing', False)) is not bool:
        raise ValueError('共卡选项必须是开关')
    memory_length = request.get('memory_length', 64)
    memory_conditioning = request.get('memory_conditioning', 'residual')
    memory_residual_readout = request.get('memory_residual_readout', 'delta')
    image_augmentation = request.get('image_augmentation', True)
    if type(memory_length) is not int or memory_length < 1:
        raise ValueError('记忆容量必须为正整数')
    if memory_conditioning not in ('direct', 'residual'):
        raise ValueError('记忆接入方式必须为direct或residual')
    if memory_residual_readout not in ('delta', 'value'):
        raise ValueError('Residual readout必须为delta或value')
    if memory_conditioning != 'residual' and memory_residual_readout != 'delta':
        raise ValueError('value-only readout仅适用于Residual Memory')
    if type(image_augmentation) is not bool:
        raise ValueError('图像增强必须是开关')
    name = request.get('task_name', 'task_' + str(task_label))
    if not re.fullmatch(r'[A-Za-z0-9_]{1,100}', name):
        raise ValueError('任务名格式不正确')
    checkpoint = _path(request.get('checkpoint'), '权重')
    known = catalog()['datasets']
    sources=[]
    splits={}
    train,validation,mapping=[],[],{}
    count=2*percent
    if request['global_windows_per_step']>200-count:
        raise ValueError('全局窗口数不能超过每个任务的训练Episode数，请减少窗口数或验证比例')
    for task in tasks:
        source=next((d for d in known if d['task']==task),None)
        if source is None:
            raise ValueError(f'Task{task}数据尚未在{DEVICE_LABEL}就绪，请先准备数据')
        source=dict(source)
        original=read_json(Path(source['split']))
        if 'tasks' not in request and (request.get('dataset')!=source['dataset'] or read_json(_path(request.get('split'),'数据划分'))!=original):
            raise ValueError('请选择已登记的数据集和对应Episode划分')
        ids=original['train']+original['validation']
        if original.get('task')!=task or len(ids)!=200 or len(set(ids))!=200 or set(original['train']) & set(original['validation']):
            raise ValueError(f'Task{task}需要完整的200条独立Episode')
        if not (Path(source['dataset'])/'meta/info.json').is_file():
            raise ValueError(f'Task{task}数据不存在')
        if any(e in mapping for e in ids):
            raise ValueError('多任务Episode编号重复，不能直接合并')
        if count==len(original['validation']) and request['seed']==original.get('seed',42):
            selected_train,selected_validation=original['train'],original['validation']
        else:
            ids=sorted(ids)
            random.Random(request['seed']+task).shuffle(ids)
            selected_train,selected_validation=sorted(ids[count:]),sorted(ids[:count])
        train.extend(selected_train);validation.extend(selected_validation)
        mapping.update({e:task for e in ids})
        splits[str(task)]=dict(task=task,seed=request['seed'],train=selected_train,validation=selected_validation)
        sources.append(source)
    split=dict(seed=request['seed'],train=train,validation=validation,
        train_episodes_per_task=200-count,validation_episodes_per_task=count,
        validation_enabled=validation_enabled,episode_tasks=mapping,per_task=splits)
    split.update(task=tasks[0]) if len(tasks)==1 else split.update(tasks=tasks,task_sampling='equal task cycle')
    if not (checkpoint/'config.json').is_file() or not PYTHON.is_file():
        raise ValueError('数据集、checkpoint或目标设备 Python 环境不存在')
    saved_config = read_json(checkpoint/'config.json')
    if 'memory_conditioning' not in request and saved_config.get('memory_enabled'):
        memory_conditioning = saved_config.get('memory_conditioning', 'residual')
        if memory_conditioning not in ('direct', 'residual'):
            raise ValueError('权重保存的记忆接入方式无效')
    if 'memory_residual_readout' not in request and saved_config.get('memory_enabled'):
        memory_residual_readout = saved_config.get('memory_residual_readout', 'delta')
    cognitive_options = {}
    for key, default, valid in (
            ('memory_cognitive_residual_projection', False, lambda v: type(v) is bool),
            ('memory_cognitive_residual_scope', 'eos', lambda v: v in ('eos', 'text'))):
        saved = saved_config.get(key, default)
        value = request.get(key, saved if saved_config.get('memory_enabled') else default)
        if not valid(value):
            raise ValueError(key+'参数无效')
        if request['arm']=='memory_v1' and saved_config.get('memory_enabled') and value != saved:
            raise ValueError(key+'与权重不同；本轮请从原始GR00T权重开始')
        cognitive_options[key] = value
    if memory_conditioning != 'residual' and (cognitive_options['memory_cognitive_residual_projection']
            or cognitive_options['memory_cognitive_residual_scope'] != 'eos'):
        raise ValueError('认知残差设置仅适用于Residual Memory')
    # Preserve old trained checkpoints. Fresh GR00T runs get the stable retrieval.
    saved_qk_norm = saved_config.get('memory_cognitive_qk_norm', 'none')
    memory_cognitive_qk_norm = request.get('memory_cognitive_qk_norm',
        saved_qk_norm if saved_config.get('memory_enabled') else 'layernorm')
    if memory_cognitive_qk_norm not in ('none', 'layernorm'):
        raise ValueError('认知检索归一化必须为none或layernorm')
    if (request['arm']=='memory_v1' and saved_config.get('memory_enabled')
            and memory_cognitive_qk_norm != saved_qk_norm):
        raise ValueError('认知检索归一化与权重不同；本轮请从原始GR00T权重开始')
    if (request['arm']=='memory_v1' and saved_config.get('memory_enabled')
            and saved_config.get('memory_conditioning', 'residual') != memory_conditioning):
        raise ValueError('记忆接入方式与权重不同；本轮请从原始GR00T权重开始')
    if (request['arm']=='memory_v1' and saved_config.get('memory_enabled')
            and saved_config.get('memory_residual_readout', 'delta') != memory_residual_readout):
        raise ValueError('Residual readout与权重不同；本轮请从原始GR00T权重开始')
    existing = compute_pids(selected)
    if existing and not request.get('share_existing') and not request.get('enqueue'):
        raise ValueError('所选GPU有现有进程，请换卡，或显式勾选共卡；不会停止现有进程')
    return dict(version=version, method=method, method_description=description, notes=notes.strip(), naming_schema=NAMING_SCHEMA,
        device_id=_device.get('device_id','a100'), device_label=DEVICE_LABEL,
        run_name=name_base+'_trainNN', task=task_label, tasks=tasks, task_name=name, arm=request['arm'],
        datasets=sources,dataset=sources[0]['dataset'] if len(tasks)==1 else None,
        checkpoint=str(checkpoint), split=sources[0]['split'] if len(tasks)==1 else None, split_value=split, gpus=gpus,
        validation_gpu=validation_gpu, validation_enabled=validation_enabled, validation_percent=percent,
        train_episodes=len(train),validation_episodes=len(validation),enqueue=request.get('enqueue',False),
        share_existing=request.get('share_existing',False),steps=request['steps'],
        global_windows_per_step=request['global_windows_per_step'], global_decisions=32*request['global_windows_per_step'],
        microbatch=request['microbatch'], seed=request['seed'], learning_rate=lr, existing_pids=existing,
        memory_length=memory_length, memory_conditioning=memory_conditioning,
        memory_residual_readout=memory_residual_readout, image_augmentation=image_augmentation,
        memory_cognitive_qk_norm=memory_cognitive_qk_norm,
        **cognitive_options, optimizer_eps=optimizer_eps,
        **fine_tuning, weight_decay=weight_decay, max_grad_norm=max_grad_norm, validation_interval=validation_interval,
        disable_early_stopping=disable_early_stopping,
        frozen_eagle=all(fine_tuning['freeze_' + k] for k in ('llm', 'vision', 'vision_projector')),
        lora=fine_tuning['fine_tuning_mode']=='lora', checkpoint_interval=1000, asynchronous_validation=validation_enabled,
        initialization='New optimizer from selected weights; not an optimizer-state resume')


def command(plan, source, output):
    result = ['bash', str(source/'train/run_groot_train.sh'), '--source-dir', str(source),
        '--python-path', str(PYTHON)]
    for key in ['dataset', 'checkpoint', 'split', 'arm', 'steps', 'global_windows_per_step', 'microbatch', 'seed', 'learning_rate']:
        value = output/'config/data_split.json' if key=='split' else plan[key]
        result += ['--'+key.replace('_','-'), str(value)]
    result += ['--output', str(output), '--gpus', ','.join(map(str,plan['gpus']))]
    for key, value in resolve_finetuning(plan).items():
        flag = key.replace('_', '-')
        result += [('--' if value else '--no-') + flag] if type(value) is bool else ['--' + flag, str(value)]
    result += ['--weight-decay', str(plan.get('weight_decay', .01)), '--max-grad-norm', str(plan.get('max_grad_norm', 1.))]
    result += ['--optimizer-eps', str(plan.get('optimizer_eps', 1e-8)),
               '--memory-cognitive-residual-scope', plan.get('memory_cognitive_residual_scope', 'eos'),
               '--memory-cognitive-residual-projection' if plan.get('memory_cognitive_residual_projection', False)
               else '--no-memory-cognitive-residual-projection']
    result += ['--memory-length', str(plan.get('memory_length', 64)),
               '--memory-conditioning', plan.get('memory_conditioning', 'residual'),
               '--memory-residual-readout', plan.get('memory_residual_readout', 'delta'),
               '--memory-cognitive-qk-norm', plan.get('memory_cognitive_qk_norm', 'layernorm'),
               '--image-augmentation' if plan.get('image_augmentation', True) else '--no-image-augmentation']
    if plan['validation_enabled']:
        result += ['--async-validation-gpu', str(plan['validation_gpu']),
                   '--validation-interval', str(plan.get('validation_interval', 1000))]
    else:
        result.append('--no-validation')
    if plan.get('disable_early_stopping', False):
        result.append('--disable-early-stopping')
    if plan['existing_pids'] and plan['share_existing']:
        result += ['--allow-service-pids', *map(str,plan['existing_pids'])]
    return result


def materialize_dataset(plan, output):
    """Merge metadata and link existing videos/parquets; never copy large data."""
    if len(plan['datasets'])==1:
        return plan['datasets'][0]['dataset']
    target=output/'dataset'
    reference=Path(plan['datasets'][0]['dataset'])
    info=read_json(reference/'meta/info.json')
    rows=[]
    for source in plan['datasets']:
        root=Path(source['dataset'])
        for name in ['modality.json','stats.json','relative_stats.json','tasks.jsonl']:
            if (root/'meta'/name).read_bytes()!=(reference/'meta'/name).read_bytes():
                raise ValueError('多任务数据的归一化或模态配置不同：'+name)
        current=read_json(root/'meta/info.json')
        if any(current.get(field)!=info.get(field) for field in ['features','fps','chunks_size','data_path','video_path']):
            raise ValueError('多任务数据的路径模板或特征配置不同')
        rows.extend(json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines() if line)
    if {r['episode_index'] for r in rows}!=set(plan['split_value']['episode_tasks']) or len(rows)!=200*len(plan['tasks']):
        raise ValueError('多任务Episode元数据不完整')
    (target/'meta').mkdir(parents=True)
    for name in ['modality.json','stats.json','relative_stats.json','tasks.jsonl']:
        shutil.copyfile(reference/'meta'/name,target/'meta'/name)
    for source in plan['datasets']:
        root=Path(source['dataset'])
        for kind in ['data','videos']:
            for chunk in (root/kind).iterdir():
                dest=target/kind/chunk.name
                if dest.exists():
                    raise ValueError('多任务数据chunk路径重叠')
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.symlink_to(chunk.resolve(),target_is_directory=True)
    info.update(total_episodes=len(rows),total_frames=sum(r['length'] for r in rows),
        total_videos=len(rows)*3,total_chunks=len(list((target/'data').iterdir())),
        splits=dict(train_episode_ids=plan['split_value']['train'],validation_episode_ids=plan['split_value']['validation']))
    atomic_json(target/'meta/info.json',info)
    (target/'meta/episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return str(target)


def start(request):
    plan = prepare(request)
    manifest = read_json(LIVE/'web_source_manifest.json')
    if not manifest or manifest['digest'] != request_digest:
        raise ValueError('同步源码版本不匹配，请重新同步')
    run_id, number = reserve_training(TRAINING, PROJECT/'deployments',plan['version'],plan['task'],plan['method'])
    plan.update(training_id=run_id,train_number=number,run_name=run_id)
    output = TRAINING/run_id
    source = PROJECT/'deployments'/run_id
    shutil.copytree(LIVE, source, symlinks=True)
    plan['dataset']=materialize_dataset(plan,output)
    argv = command(plan, source, output)
    atomic_json(output/'config/data_split.json', plan.pop('split_value'))
    atomic_json(output/'config/web_launch.json', dict(plan, source=str(source), source_manifest=manifest, argv=argv))
    atomic_json(output/'config/source_manifest.json', manifest)
    (output/plan['arm']).mkdir(exist_ok=True)
    if plan['validation_enabled']:
        atomic_json(output/(plan['arm']+'/validation_queue/step-000000.json'), dict(step=0, checkpoint=plan['checkpoint']))
    (output/'logs').mkdir(exist_ok=True)
    status='queued' if plan['enqueue'] else 'submitted'
    if plan['enqueue']:
        queue=TRAINING/'queues'/run_id
        atomic_json(queue/'config.json',dict(output=str(output),arm=plan['arm'],argv=argv,
            gpus=plan['gpus']+([plan['validation_gpu']] if plan['validation_enabled'] else []),
            allowed_processes={str(pid):process_start(pid) for pid in plan['existing_pids']} if plan['share_existing'] else {}))
        atomic_json(queue/'status.json',dict(status='queued',jobs=[dict(run=run_id,arm=plan['arm'])],
            created_at=datetime.datetime.now().isoformat()))
        atomic_json(output/plan['arm']/'status.json',dict(status='queued',step=0,total_steps=plan['steps']))
        argv=[str(PYTHON),'-u',str(source/'train/web_training_queue.py'),'--config',str(queue/'config.json')]
    with (output/'logs/controller.log').open('ab') as log:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    job = dict(run=run_id, training_id=run_id, version=plan['version'],method=plan['method'],train_number=number,
               arm=plan['arm'], pid=proc.pid, output=str(output), source=str(source),
               source_digest=manifest['digest'], command=shlex.join(argv), status=status)
    atomic_json(output/'config/web_process.json', job)
    if plan['enqueue']:
        # Queue cancellation identifies this supervisor separately from training.
        state=read_json(queue/'status.json')
        if state.get('status')=='queued':
            atomic_json(queue/'status.json',dict(state,pid=proc.pid))
        atomic_json(queue/'process.json',dict(pid=proc.pid,argv=argv))
    return job


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:
        return None


def tail(path, size=16000):
    if not path.exists():
        return ''
    with path.open('rb') as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell()-size))
        return handle.read().decode(errors='replace')


def job_root(request):
    run = request.get('run','')
    arm = request.get('arm','')
    if not isinstance(run,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',run) or arm not in ['baseline','memory_v1']:
        raise ValueError('训练运行名称不正确')
    root = TRAINING/run
    if not ((root/'config/ddp_launch.json').is_file() or (root/'config/web_launch.json').is_file()) or not (root/arm).is_dir():
        raise ValueError('训练运行不存在')
    return root, arm


def recorded_plan(root):
    plan = read_json(root/'config/ddp_launch.json')
    if plan:
        return plan
    launch = read_json(root/'config/web_launch.json', {})
    return dict(steps_each=launch.get('steps'), commands={launch['arm']:
        dict(CUDA_VISIBLE_DEVICES=','.join(map(str,launch['gpus'])))}) if launch else {}


def live_run_processes(root):
    """Find this run's actual workers/controllers, including orphaned ranks."""
    result=[]
    for file in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv=file.read_bytes().decode().strip('\0').split('\0')
            if '--output' in argv and argv[argv.index('--output')+1]==str(root):
                result.append(int(file.parent.name))
        except (OSError, UnicodeError, IndexError):
            continue
    process=read_json(root/'config/web_process.json',{})
    pid=process.get('pid')
    if pid and process_start(pid) is not None:
        try:
            argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
            if any(str(root) in arg for arg in argv) and pid not in result:
                result.append(pid)
        except (OSError, UnicodeError):
            pass
    return result


def resume_info(root, arm):
    """Lightweight availability check; full state validation runs on submission."""
    terminal={'complete','completed','stopped','failed'}
    state=read_json(root/arm/'status.json',{})
    plan=recorded_plan(root)
    launch=read_json(root/'config/web_launch.json',{})
    config=read_json(root/arm/'config/ddp.json',{})
    latest=read_json(root/arm/'checkpoints/latest.json',{})
    failed=read_json(root/'status.json',{}).get('status')=='failed'
    info=dict(available=False,step=latest.get('step'),reason='')
    if state.get('status') not in terminal and not failed:
        info['reason']='请先等待训练停止或结束'
    elif len(plan.get('commands',{}))!=1:
        info['reason']='配对运行暂不支持单独续跑，请使用原配对命令'
    elif not failed and read_json(root/arm/'validation_status.json',{}).get('status') not in terminal | {None,'disabled'}:
        info['reason']='请等待独立验证退出后续跑'
    elif not launch.get('argv') or not launch.get('source') or not plan.get('commands',{}).get(arm,{}).get('argv'):
        info['reason']='历史运行缺少完整网页启动命令，请使用原 CLI 续跑'
    elif not config or not latest.get('step'):
        info['reason']='尚无可恢复的完整训练断点'
    else:
        validation_enabled=launch.get('validation_enabled',not config.get('no_validation',False))
        launch_argv=launch['argv']
        validation_gpu=(int(launch_argv[launch_argv.index('--async-validation-gpu')+1])
                        if '--async-validation-gpu' in launch_argv else None)
        checkpoint=Path(latest.get('path',''))
        index=read_json(checkpoint/'model.safetensors.index.json',{})
        weights=set(index.get('weight_map',{}).values()) or {'model.safetensors'}
        files=[root/arm/'checkpoints/latest_optimizer.pt',
               *[root/arm/f'checkpoints/latest_rank-{r:03d}.pt' for r in range(config.get('world_size',0))],
               checkpoint/'config.json',*[checkpoint/name for name in weights],
               Path(launch['source'])/'src/v1_groot_memoryvla/experiment/task21_ddp.py',
               Path(launch['source'])/'train/run_groot_train.sh']
        if validation_enabled and validation_gpu is None:
            info['reason']='原运行采用同步验证，请使用原 CLI 处理续跑'
        elif not config.get('world_size') or any(not p.is_file() for p in files):
            info['reason']='模型、优化器、各 rank 状态或原源码快照不完整'
        elif any(latest.get(k)!=config.get(k) for k in ['world_size','seed','global_windows_per_step']):
            info['reason']='断点与原训练配置不匹配'
        else:
            previous=read_json(root/'config/web_resume_latest.json',{})
            minimum=1000*math.ceil(max(config['steps'],latest['step']+1)/1000)
            info.update(available=True,checkpoint=str(checkpoint),source=launch['source'],
                gpus=launch['gpus'],seed=config['seed'],min_steps=minimum,
                default_steps=max(minimum,previous.get('steps',config['steps'])),
                validation_enabled=validation_enabled,validation_gpu=validation_gpu)
    return info


def validate_resume(root, arm, info, steps):
    # Import only the saved training implementation. mmap avoids copying optimizer
    # tensors; no model construction, CUDA context or GPU allocation is needed.
    code='''import argparse,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]+'/src')
from v1_groot_memoryvla.experiment.task21_ddp import resume_manifest
import torch
root=Path(sys.argv[2]); arm=sys.argv[3]
config=json.loads((root/arm/'config/ddp.json').read_text())
config.update(steps=int(sys.argv[4]),resume_step=int(sys.argv[5]))
args=argparse.Namespace(**config)
manifest=resume_manifest(args,root/arm,config['world_size'])
for rank in range(config['world_size']):
    state=torch.load(root/arm/f'checkpoints/latest_rank-{rank:03d}.pt',map_location='cpu',weights_only=True,mmap=True)
    assert all(state[k]==manifest[k] for k in ('step','epoch','position')), 'rank checkpoint mismatch'
optimizer=torch.load(root/arm/'checkpoints/latest_optimizer.pt',map_location='cpu',weights_only=True,mmap=True)
assert optimizer['state'], 'Missing optimizer state'
assert all(0<int(value['step'])<=manifest['step'] for value in optimizer['state'].values()), 'Optimizer step mismatch'
'''
    result=subprocess.run([str(PYTHON),'-c',code,info['source'],str(root),arm,str(steps),str(info['step'])],
        capture_output=True,text=True,timeout=60,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
        PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
    if result.returncode:
        raise ValueError('断点校验失败，未启动训练：'+(result.stderr or result.stdout)[-1600:])


def replace_option(argv, key, value=None, multiple=False):
    argv=list(argv)
    if key in argv:
        start=argv.index(key);end=start+1
        while end<len(argv) and not argv[end].startswith('--') and (multiple or end==start+1):
            end+=1
        del argv[start:end]
    if value is not None:
        argv += [key,str(value)]
    return argv


def resume(request, controller_source):
    root,arm=job_root(request)
    # Serialize independently of the HTTP process so retries cannot submit twice.
    with (TRAINING/'.gpu_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        info=resume_info(root,arm)
        if not info['available']:
            raise ValueError(info['reason'])
        if live_run_processes(root):
            raise ValueError('原运行仍有训练或验证进程，请等待退出后续跑')
        steps=request.get('steps')
        if type(steps) is not int or steps%1000 or steps<info['min_steps']:
            raise ValueError(f"总步数须为1000的整数倍，且至少为 {info['min_steps']}；填写总步数而非新增步数")
        share=request.get('share_existing',False)
        if type(share) is not bool:
            raise ValueError('共卡选项必须是开关')
        launch=read_json(root/'config/web_launch.json')
        selected=info['gpus']+([info['validation_gpu']] if info['validation_enabled'] else [])
        occupied=compute_pids(selected)
        if occupied and not share:
            raise ValueError('原训练GPU有现有进程，请等待释放，或显式勾选允许共卡；不会停止现有进程')
        validate_resume(root,arm,info,steps)
        if not controller_source:
            raise ValueError('续跑控制器缺失，请刷新网页后重试')
        worker=recorded_plan(root)['commands'][arm]['argv']
        worker=replace_option(replace_option(worker,'--steps',steps),'--resume-step',info['step'])
        worker[0]=str(PYTHON)
        argv=replace_option(launch['argv'],'--allow-service-pids',multiple=True)
        argv=replace_option(replace_option(argv,'--steps',steps),'--resume-step',info['step'])
        if occupied: argv += ['--allow-service-pids',*map(str,occupied)]
        parent=root/'config/web_resumes';parent.mkdir(exist_ok=True)
        number=max([int(p.name.split('-')[-1]) for p in parent.glob('resume-*') if p.is_dir()]+[0])+1
        attempt=parent/f'resume-{number:03d}';attempt.mkdir()
        spec=dict(run=root.name,arm=arm,source=info['source'],output=str(root),gpus=info['gpus'],
            steps=steps,resume_step=info['step'],checkpoint=info['checkpoint'],argv=argv,worker_argv=worker,
            validation_enabled=info['validation_enabled'],existing_pids=occupied,
            requested_at=datetime.datetime.now().isoformat(),attempt=str(attempt))
        atomic_json(attempt/'request.json',spec)
        (attempt/'controller.py').write_text(controller_source)
        # Preserve the old stop request, status and any uncheckpointed metrics.
        # Replay begins at the checkpoint, so failed-tail rows cannot stay on the curve.
        changed={}
        def backup(file):
            if file in changed: return
            changed[file]=file.read_bytes() if file.exists() else None
            if file.exists():
                saved=attempt/'previous'/file.relative_to(root)
                saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,saved)
        try:
            for file in (root/arm).glob('**/metrics/metrics.jsonl'):
                rows=file.read_text().splitlines(keepends=True);keep=[]
                for line in rows:
                    try: row=json.loads(line)
                    except ValueError: continue
                    if row.get('step',0)<=info['step']: keep.append(line if line.endswith('\n') else line+'\n')
                if ''.join(keep)!= ''.join(rows):
                    backup(file);temporary=file.with_suffix('.pending');temporary.write_text(''.join(keep));temporary.replace(file)
            stop_file=root/'metrics/stop_training.json'
            if stop_file.exists(): backup(stop_file);stop_file.unlink()
            for file in [root/'status.json',root/arm/'status.json']:
                backup(file)
                atomic_json(file,dict(status='resuming',step=info['step'],total_steps=steps,training_complete=False))
            for file in [root/'config/web_process.json',root/'config/web_resume_latest.json']:
                backup(file)
            with (root/'logs/controller.log').open('ab') as log:
                proc=subprocess.Popen([str(PYTHON),'-u',str(attempt/'controller.py'),'--config',str(attempt/'request.json')],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except Exception:
            for file,content in changed.items():
                if content is None: file.unlink(missing_ok=True)
                else: file.write_bytes(content)
            raise
        job=dict(read_json(root/'config/web_process.json',{}),run=root.name,arm=arm,pid=proc.pid,
            status='resuming',source=info['source'],output=str(root),resume_step=info['step'],steps=steps,attempt=str(attempt))
        atomic_json(root/'config/web_process.json',job)
        atomic_json(root/'config/web_resume_latest.json',spec)
        return job


def queue_pending(directory):
    state = read_json(directory/'status.json', {})
    if state.get('status') not in WAITING_QUEUE_STATES or not state.get('jobs'):
        raise ValueError('队列已经开始训练或已经结束，不能取消排队；请刷新状态')
    for entry in state['jobs']:
        root, arm = job_root(entry)
        arm_state = read_json(root/arm/'status.json', {})
        process=read_json(root/'config/web_process.json',{})
        if (arm_state.get('status') != 'queued' or arm_state.get('step', 0) != 0
                or entry.get('pid') or process and process.get('status')!='queued'):
            raise ValueError('队列中已有任务开始启动，不能取消排队；请使用保存后停止')
    return state


def terminate_waiting_controller(directory):
    """Freeze and recheck the waiting controller before terminating just that PID."""
    record = read_json(directory/'process.json', {})
    pid, argv = record.get('pid'), record.get('argv', [])
    if (type(pid) is not int or pid <= 1 or '--config' not in argv
            or argv[argv.index('--config')+1:] != [str(directory/'config.json')]
            or not any(str(arg).startswith(str(PROJECT/'deployments')+'/')
                       and str(arg).endswith(('/train/joint_task_queue.py','/train/web_training_queue.py')) for arg in argv)):
        raise ValueError('排队进程记录不完整，未发送停止信号')
    proc = Path('/proc')/str(pid)
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        # A queue whose controller has already exited can still be withdrawn.
        queue_pending(directory)
        return
    paused = False
    try:
        if proc.stat().st_uid != os.getuid() or proc.joinpath('cmdline').read_bytes().split(b'\0')[:-1] != [str(a).encode() for a in argv]:
            raise ValueError('排队PID与记录不一致，未发送停止信号')
        if read_json(directory/'status.json', {}).get('pid') != pid:
            raise ValueError('排队进程已经变化，请刷新后重试')
        signal.pidfd_send_signal(fd, signal.SIGSTOP)
        paused = True
        deadline = time.monotonic()+2
        while True:
            state = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[0]
            if state in ('T', 't'):
                break
            if time.monotonic() >= deadline:
                raise ValueError('无法确认排队进程已暂停，请重试')
            time.sleep(.01)
        queue_pending(directory)
        if proc.joinpath(f'task/{pid}/children').read_text().strip():
            raise ValueError('队列正在执行子进程，本次未取消；请稍后重试')
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        if paused:
            try:
                signal.pidfd_send_signal(fd, signal.SIGCONT)
            except ProcessLookupError:
                pass
        os.close(fd)


def cancel(request):
    root, arm = job_root(request)
    for directory in sorted((TRAINING/'queues').glob('*')):
        if directory.is_dir() and any(entry.get('run') == root.name and entry.get('arm') == arm
                for entry in read_json(directory/'status.json', {}).get('jobs', [])):
            break
    else:
        raise ValueError('该任务没有可撤销的训练队列')
    # The same lock serializes dispatch against cancellation for web queues.
    with (TRAINING/'.gpu_queue.lock').open('a') as dispatch_lock, (directory/'cancel.lock').open('a') as lock:
        fcntl.flock(dispatch_lock,fcntl.LOCK_EX)
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_json(directory/'status.json', {})
        if state.get('status') == 'cancelled':
            return state['cancellation']
        state = queue_pending(directory)
        terminate_waiting_controller(directory)
        control = dict(status='cancelled', affected_runs=[entry['run'] for entry in state['jobs']],
            requested_at=datetime.datetime.now().isoformat(),
            reason='User cancelled the pending training queue from the dashboard')
        atomic_json(directory/'cancellation.json', control)
        for entry in state['jobs']:
            job, job_arm = job_root(entry)
            arm_state = read_json(job/job_arm/'status.json', {})
            atomic_json(job/job_arm/'status.json', dict(arm_state, status='cancelled', cancellation=control))
        atomic_json(directory/'status.json', dict(state, status='cancelled', cancellation=control))
        return control


def jobs():
    result = []
    queued = {}
    for path in TRAINING.glob('queues/*/status.json'):
        queue = read_json(path, {})
        for entry in queue.get('jobs', []):
            if entry.get('run'):
                queued[entry['run']] = dict(name=path.parent.name, status=queue.get('status'),
                    dependencies=queue.get('dependencies', []), unexpected_pids=queue.get('unexpected_pids', []),
                    waiting_for_runs=queue.get('waiting_for_runs',[]),
                    error=queue.get('error'), path=str(path),
                    affected_runs=[job['run'] for job in queue.get('jobs', []) if job.get('run')],
                    cancellable=queue.get('status') in WAITING_QUEUE_STATES and (path.parent/'process.json').is_file())
    roots = {p.parents[1] for name in ['ddp_launch.json', 'web_launch.json'] for p in TRAINING.glob('*/config/'+name)}
    for root in sorted(roots, key=lambda p:p.name, reverse=True):
        if root.is_symlink():
            continue
        plan = recorded_plan(root)
        split = read_json(root/'config/data_split.json',{})
        aggregate = read_json(root/'status.json',{})
        stop = read_json(root/'metrics/stop_training.json',{})
        locations = read_json(root/'config/remote_locations.json',{})
        for arm, cmd in plan.get('commands',{}).items():
            if (root/'config'/f'dashboard_deleted_{arm}.json').is_file():
                continue
            state = read_json(root/arm/'status.json',{})
            last = None
            for line in reversed(tail(root/arm/'metrics/metrics.jsonl', 100000).splitlines()):
                try:
                    record = json.loads(line)
                    if record.get('kind')=='train':
                        last = {k:record.get(k) for k in ['step','loss','seconds','gradient_norm','peak_allocated_gib',
                            'decisions_per_second_end_to_end']}
                        break
                except ValueError:
                    continue
            status=state.get('status',aggregate.get('status','submitted'))
            queue = queued.get(root.name)
            if status == 'queued' and queue:
                status = queue.get('status', status)
            process=read_json(root/'config/web_process.json',{})
            if aggregate.get('status')=='failed':
                status='failed'
            elif not state and process.get('pid'):
                try: os.kill(process['pid'],0)
                except ProcessLookupError: status='failed'
            validation = read_json(root/arm/'validation_status.json',{})
            terminal = {'complete','completed','stopped','failed','cancelled'}
            can_delete = status in terminal and validation.get('status') in terminal | {None, 'disabled'}
            result.append(dict(run=root.name, name=display_name(root,arm,split.get('task')), arm=arm, task=split.get('task'), status=status,
                **training_identity(root,arm,split.get('task')),
                step=state.get('step', (last or {}).get('step',0)), max_steps=stop.get('stop_after_step',state.get('total_steps',plan.get('steps_each'))),
                gpus=cmd.get('CUDA_VISIBLE_DEVICES'), training_complete=state.get('training_complete',False),
                validation=validation, last=last,
                source=(read_json(root/'config/web_process.json',{}).get('source') or
                        read_json(root/'config/extension_5000.json',{}).get('source') or
                        read_json(root/'config/run_config.json',{}).get('source') or
                        locations.get('memory_source' if arm=='memory_v1' else 'baseline_controller_source')),
                output=str(root), stop_reason=stop.get('reason'), queue=queue,
                stop_after_step=stop.get('stop_after_step'), can_delete=can_delete,
                resume=resume_info(root,arm),
                notes=read_json(root/'config/web_launch.json',{}).get('notes',''),
                delete_block_reason='' if can_delete else '请先停止训练或取消排队，并等待独立验证结束。',
                validation_enabled=read_json(root/'config/web_launch.json',{}).get('validation_enabled',True),
                can_cancel=bool(queue and queue['cancellable'] and state.get('status') == 'queued'),
                can_stop=not state.get('training_complete',False) and len(plan.get('commands',{}))==1
                    and status in ('training','running')))
    return dict(jobs=result)


def initial_weights(root, arm, config):
    """Use this run's saved configuration, never the current form selection."""
    worker_file = root/arm/'config/ddp.json'
    launch_file = root/'config/web_launch.json'
    launch = read_json(launch_file, {})
    argv = recorded_plan(root).get('commands', {}).get(arm, {}).get('argv', [])
    def argument(name):
        return argv[argv.index(name)+1] if name in argv and argv.index(name)+1 < len(argv) else None
    checkpoint = config.get('checkpoint') or launch.get('checkpoint') or argument('--checkpoint')
    source = str(worker_file if config.get('checkpoint') else launch_file if launch.get('checkpoint') else
                 root/'config/ddp_launch.json') if checkpoint else None
    resume_step = int(argument('--resume-step') or config.get('resume_step') or 0)
    resume_file = None
    resume_checkpoint = None
    for file in sorted((root/arm/'config').glob('resume-*.json')):
        match = re.fullmatch(r'resume-(?:step-)?(\d+)\.json', file.name)
        if match and int(match[1]) >= resume_step:
            resume_step = int(match[1])
            resume_file = str(file)
            manifest = read_json(file, {}).get('checkpoint')
            if isinstance(manifest, dict):
                resume_checkpoint = manifest.get('path')
    label = ('官方 GR00T N1.6-3B · 通用预训练' if checkpoint == str(BASE_CHECKPOINT) else
             '官方 GR00T N1.6-BEHAVIOR1k · B1K 微调' if checkpoint == str(OFFICIAL) else
             '项目权重 / 自定义权重' if checkpoint else '未记录')
    mode = 'resume' if resume_step else 'fresh' if checkpoint else 'unknown'
    restored=mode=='resume' if mode!='unknown' else None
    pending=read_json(root/'config/web_resume_latest.json',{})
    if pending.get('arm')==arm and pending.get('resume_step',0)>=resume_step:
        resume_step=pending['resume_step'];mode='resume'
        resume_checkpoint=pending['checkpoint']
        resume_file=str(root/'config/web_resume_latest.json')
        restored=read_json(root/arm/f'config/resume-step-{resume_step:06d}.json',{}).get('optimizer_restored',False)
    return dict(checkpoint=checkpoint, label=label, source_file=source, mode=mode,
        resume_step=resume_step, resume_checkpoint=resume_checkpoint, resume_record=resume_file,
        optimizer_restored=restored)


def training_parameters(root, arm, config):
    """Read the selected run's recorded values; never use current form defaults."""
    launch=read_json(root/'config/web_launch.json',{})
    plan=recorded_plan(root)
    cmd=plan.get('commands',{}).get(arm,{})
    argv=cmd.get('argv',[]) or launch.get('argv',[])
    values={};sources=[]
    if launch: sources.append(str(root/'config/web_launch.json'))
    if argv: sources.append(str(root/'config/ddp_launch.json') if cmd.get('argv') else str(root/'config/web_launch.json'))
    if config: sources.append(str(root/arm/'config/ddp.json'))
    numeric={'steps':int,'global_windows_per_step':int,'microbatch':int,'seed':int,'learning_rate':float,
             'memory_length':int,'checkpoint_interval':int,'lora_rank':int,'lora_alpha':float,'lora_dropout':float,
             'llm_last_n_layers':int,'weight_decay':float,'max_grad_norm':float,'optimizer_eps':float}
    fields=[*numeric,'memory_conditioning','memory_residual_readout','memory_cognitive_qk_norm','dataset','checkpoint','training_protocol','frozen_eagle',
            'memory_cognitive_residual_projection','memory_cognitive_residual_scope',
            'image_augmentation','disable_early_stopping','fine_tuning_mode','validation_interval','lora_targets',
            'gradient_checkpointing', *('freeze_' + name for name in FINETUNING_COMPONENTS)]
    for key in fields:
        if key in config: values[key]=config[key]
        elif key in launch: values[key]=launch[key]
        else:
            flag='--'+key.replace('_','-')
            if flag in argv and argv.index(flag)+1<len(argv):
                raw=argv[argv.index(flag)+1]
                if raw.startswith('--'): continue
                try: values[key]=numeric.get(key,str)(raw)
                except (ValueError,TypeError): pass
    values['gpus']=cmd.get('CUDA_VISIBLE_DEVICES') or (','.join(map(str,launch['gpus'])) if 'gpus' in launch else None)
    values['validation_enabled']=(not config['no_validation'] if 'no_validation' in config else
        launch.get('validation_enabled',plan.get('validation_enabled')))
    values['validation_gpu']=launch.get('validation_gpu',plan.get('async_validation_gpu'))
    values['validation_interval']=config.get('eval_interval',values.get('validation_interval'))
    windows=values.get('global_windows_per_step')
    decisions=config.get('global_decisions_per_step',launch.get('global_decisions'))
    if decisions is None and windows and config.get('decisions_per_window'):
        decisions=windows*config['decisions_per_window']
    values['global_decisions_max']=decisions
    split=read_json(root/'config/data_split.json',{})
    values.update(tasks=split.get('tasks') or ([split['task']] if 'task' in split else None),
        train_episodes=len(split['train']) if 'train' in split else None,
        validation_episodes=len(split['validation']) if 'validation' in split else None)
    resume=read_json(root/'config/web_resume_latest.json',{})
    state=read_json(root/arm/'status.json',{})
    values['original_steps']=values.get('steps')
    values['steps']=state.get('total_steps',resume.get('steps',values.get('steps')))
    values['stop_after_step']=read_json(root/'metrics/stop_training.json',{}).get('stop_after_step')
    return dict(values=values,sources=list(dict.fromkeys(sources)))


def detail(request):
    root, arm = job_root(request)
    blocks, epochs, validation, loss_series = [], {}, [], []
    group = []
    recent_seconds = deque(maxlen=50)
    train_seconds, timed_steps, last_step, last_seconds = 0., 0, 0, None
    path = root/arm/'metrics/metrics.jsonl'
    with (path.open() if path.exists() else open(os.devnull)) as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row['kind']=='eval':
                validation.append({k:row.get(k) for k in ['step','validation_loss','action_mae']})
                continue
            if row['kind']!='train':
                continue
            last_step = row['step']
            seconds = row.get('seconds')
            last_seconds = seconds if type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0 else None
            if last_seconds is not None:
                train_seconds += last_seconds
                timed_steps += 1
                recent_seconds.append(last_seconds)
            count=row.get('global_decisions',1)
            loss_series.append([row['step'],row['loss'],1])
            epoch=epochs.setdefault(row['epoch'],dict(epoch=row['epoch']+1,steps=0,count=0,total=0.,first=row['step']))
            epoch.update(steps=epoch['steps']+1,count=epoch['count']+count,total=epoch['total']+row['loss']*count,last=row['step'])
            group.append((row['step'],row['loss'],count))
            if len(group)==50:
                blocks.append([group[-1][0],sum(v*n for _,v,n in group)/sum(n for _,_,n in group),len(group)])
                group=[]
    if group:
        blocks.append([group[-1][0],sum(v*n for _,v,n in group)/sum(n for _,_,n in group),len(group)])
    config=read_json(root/arm/'config/ddp.json',{})
    means=[dict(epoch=e['epoch'],first=e['first'],last=e['last'],steps=e['steps'],loss=e['total']/e['count'],
                complete=e['steps']==config.get('steps_per_epoch')) for e in epochs.values()]
    checkpoints=[]
    for entry in sorted((root/arm/'validation_queue').glob('step-*.json'), reverse=True):
        published=read_json(entry)
        if published['step'] and (Path(published['checkpoint'])/'config.json').is_file():
            checkpoints.append(dict(step=published['step'],path=str(Path(published['checkpoint']).resolve())))
    latest=read_json(root/arm/'checkpoints/latest.json',{}).get('step',0)
    published_steps={p['step'] for p in checkpoints}
    for file in (root/arm/'checkpoints').glob('step-*/config.json'):
        step=int(file.parent.name.split('-')[-1])
        if step<=latest and step not in published_steps:
            checkpoints.append(dict(step=step,path=str(file.parent.resolve())))
    checkpoints.sort(key=lambda p:p['step'],reverse=True)
    state = read_json(root/arm/'status.json', {})
    plan = recorded_plan(root)
    stop = read_json(root/'metrics/stop_training.json', {})
    target = stop.get('stop_after_step', state.get('total_steps', plan.get('steps_each')))
    step = state.get('step', last_step)
    recent_mean = sum(recent_seconds)/len(recent_seconds) if recent_seconds else None
    complete = state.get('training_complete', False) or state.get('status') in ('complete', 'completed')
    remaining_steps = max(0, target-step) if target is not None else None
    eta = 0. if complete or (remaining_steps == 0 and timed_steps > 0) else None
    if not complete and state.get('status') in ('training', 'running', 'saving_checkpoint', 'validating', 'paired_validation_barrier') and recent_mean is not None and remaining_steps is not None:
        eta = remaining_steps * recent_mean
    timing = dict(scope='optimizer_loop', elapsed_seconds=train_seconds if timed_steps else None,
        timed_steps=timed_steps, last_step_seconds=last_seconds,
        mean_step_seconds=train_seconds/timed_steps if timed_steps else None,
        recent_mean_step_seconds=recent_mean, recent_samples=len(recent_seconds), recent_window=50,
        remaining_steps=0 if complete else remaining_steps, eta_seconds=eta)
    return dict(run=root.name,arm=arm,curve=blocks,loss_series=loss_series,epochs=means,validation=validation,checkpoints=checkpoints,
        timing=timing, initial_weights=initial_weights(root,arm,config),
        notes=read_json(root/'config/web_launch.json',{}).get('notes',''),
        training_parameters=training_parameters(root,arm,config),
        log_tail=tail(root/f'logs/{arm}_torchrun.log') or tail(root/'logs/controller.log'),
        validation_log_tail=tail(root/f'logs/{arm}_validation.log',6000),
        checkpoint=str(root/arm/'checkpoints'), config=config)


def stop(request):
    root, arm = job_root(request)
    plan=recorded_plan(root)
    if len(plan['commands'])!=1:
        raise ValueError('此运行包含配对训练，不能只停止其中一组')
    state=read_json(root/arm/'status.json',{})
    target=request.get('step')
    if request.get('next_checkpoint') is True:
        target=(state.get('step',0)//1000+1)*1000
    if state.get('status') not in ('training','running'):
        raise ValueError('仅运行中的训练可以设置停止点；排队任务请使用取消排队')
    if state.get('training_complete'):
        raise ValueError('该组训练已经结束')
    if type(target) is not int or not state.get('step',0)<target<=state.get('total_steps',plan['steps_each']):
        raise ValueError('停止步数应为未来的整数步数，且不超过训练上限')
    # The asynchronous/no-validation worker saves a full model and optimizer at
    # stop_at itself. Expose that existing path for short diagnostic training.
    can_save_at_stop = plan.get('validation_enabled') is False or plan.get('async_validation_gpu') is not None
    if target%1000 and not can_save_at_stop:
        raise ValueError('此训练模式仅支持未来的1000整数倍停止点')
    control=dict(stop_after_step=target,reason='User requested checkpoint-boundary stop from training dashboard',
                 requested_at=datetime.datetime.now().isoformat(),arm=arm)
    atomic_json(root/'metrics/stop_training.json',control)
    return control


def delete(request):
    root, arm = job_root(request)
    job = next((row for row in jobs()['jobs'] if row['run']==root.name and row['arm']==arm), None)
    if not job or not job['can_delete']:
        raise ValueError('请先停止训练或取消排队，并等待独立验证结束后删除记录')
    control = dict(status='deleted',run=root.name,arm=arm,files_preserved=True,
        deleted_at=datetime.datetime.now().isoformat())
    # Keep all checkpoint paths usable by inference/evaluation and retain run numbering.
    atomic_json(root/'config'/f'dashboard_deleted_{arm}.json',control)
    return control


if __name__=='__main__':
    try:
        action=REQUEST['action']
        request_digest=REQUEST.get('source_digest')
        if action=='catalog': result=catalog()
        elif action=='preview': result=prepare(REQUEST['payload'])
        elif action=='start': result=start(REQUEST['payload'])
        elif action=='jobs': result=jobs()
        elif action=='detail': result=detail(REQUEST['payload'])
        elif action=='stop': result=stop(REQUEST['payload'])
        elif action=='cancel': result=cancel(REQUEST['payload'])
        elif action=='delete': result=delete(REQUEST['payload'])
        elif action=='resume': result=resume(REQUEST['payload'],REQUEST.get('resume_controller'))
        else: raise ValueError('不支持的训练操作')
        print(json.dumps(result,ensure_ascii=False))
    except Exception as error:
        print(json.dumps(dict(error=str(error)),ensure_ascii=False))
        raise SystemExit(1)
