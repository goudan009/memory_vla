"""Plans and local catalog for the LIBERO three-step workflow."""
import math
from pathlib import Path
import shlex
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from v1_groot_memoryvla.benchmarks.libero.protocol import (
    CONTRACT, DATA_ROOT, HORIZONS, manifest_path, read_json, task_names)

TITLES = {'libero_dataset':'LIBERO 示教准备','libero_train':'LIBERO 训练','libero_eval':'LIBERO 策略评估'}


def catalog(manager):
    datasets=[]
    for suite,names in task_names().items():
        for task,name in enumerate(names):
            data=read_json(manifest_path(suite,task),{})
            ready=bool(data and Path(data['file']).is_file() and Path(data['file']).stat().st_size==data['bytes'])
            datasets.append(dict(suite=suite,task=task,ready=ready,episodes=len(data.get('episodes',[])) if ready else 0))
    files=list(manager.runs.glob('*/training/checkpoints/step-*/libero.json'))
    # CLI integration runs remain selectable and are explicitly labelled.
    files+=list(manager.output.glob('libero_integration_*/checkpoints/step-*/libero.json'))
    checkpoints=[]
    for file in sorted(files,key=lambda p:p.stat().st_mtime,reverse=True):
        value=read_json(file)
        if value.get('contract')!=CONTRACT or not (file.parent/'config.json').is_file(): continue
        checkpoints.append(dict(path=str(file.parent.resolve()),label=f'{value["arm"]} · {value["suite"]} · Task'+
            ','.join(map(str,value['tasks']))+f' · step {value["step"]} · {file.parents[3].name}',
            resumable=(file.parent/'optimizer.pt').is_file(),**value))
    return dict(datasets=datasets,checkpoints=checkpoints,base_checkpoint=str(manager.root/'checkpoints/groot_n1d6_base'),
                data_root=str(DATA_ROOT),horizons=HORIZONS,device='本机 GPU',contract=CONTRACT)


def prepare(manager, request):
    allowed={'kind','gpu','suite','tasks','notes','checkpoint','arm','fine_tuning_mode','steps',
             'group_size','save_interval','seed','learning_rate','trials','max_steps'}
    if not isinstance(request,dict) or set(request)-allowed: raise ValueError('不支持的 LIBERO 参数')
    kind=request['kind']; suite=request.get('suite','libero_spatial'); tasks=request.get('tasks',[0])
    if kind not in TITLES or not isinstance(suite,str) or suite not in HORIZONS: raise ValueError('LIBERO 类型或套件不正确')
    names=task_names()[suite]
    if (not isinstance(tasks,list) or not tasks or any(type(t)is not int or not 0<=t<len(names) for t in tasks)
            or len(set(tasks))!=len(tasks)): raise ValueError('请选择有效且不重复的任务')
    gpu=request.get('gpu',0); seed=request.get('seed',42); notes=request.get('notes','')
    if type(gpu)is not int or gpu<0 or type(seed)is not int or not 0<=seed<2**32: raise ValueError('GPU 或 seed 不正确')
    if not isinstance(notes,str) or len(notes)>500: raise ValueError('备注最多 500 字')
    options=manager.options(); info=options['libero']
    if not manager.python.is_file(): raise ValueError('本机 V1 Python 环境尚未安装')
    device=next((g for g in options['gpus'] if g['index']==gpu),None)
    if kind!='libero_dataset' and device is None: raise ValueError('所选本机 GPU 不可用')
    config=dict(kind=kind,suite=suite,tasks=sorted(tasks),gpu=gpu,seed=seed,notes=notes.strip())
    prefix=[str(manager.python),'-m','v1_groot_memoryvla.benchmarks.libero.']
    common=['--suite',suite,'--tasks',*map(str,sorted(tasks))]
    need=0
    if kind=='libero_dataset':
        argv=[*prefix[:-1],prefix[-1]+'prepare',*common,'--output','{run}/dataset.json']
        stages=[dict(name='下载并校验官方示教',argv=argv,report='dataset.json')]
    elif kind=='libero_train':
        missing=[t for t in tasks if not any(d['suite']==suite and d['task']==t and d['ready'] for d in info['datasets'])]
        if missing: raise ValueError('请先准备示教：Task'+', '.join(map(str,missing)))
        arm=request.get('arm','memory'); mode=request.get('fine_tuning_mode','action_full')
        if arm not in ('memory','baseline') or mode not in ('action_full','lora'): raise ValueError('请选择 Baseline/Memory 和动作头全量/LoRA')
        values={k:request.get(k,v) for k,v in dict(steps=300,group_size=16,save_interval=100).items()}
        for key,maximum in [('steps',100000),('group_size',32),('save_interval',10000)]:
            if type(values[key])is not int or not 1<=values[key]<=maximum: raise ValueError(f'{key} 超出范围')
        lr=request.get('learning_rate',1e-5)
        if type(lr)not in (int,float) or not math.isfinite(lr) or not 0<lr<=.001: raise ValueError('学习率应在 0 到 0.001 之间')
        checkpoint=request.get('checkpoint',info['base_checkpoint'])
        saved=next((c for c in info['checkpoints'] if c['path']==checkpoint),None)
        if checkpoint!=info['base_checkpoint']:
            if saved is None or not saved['resumable']: raise ValueError('请选择本机可续训的 LIBERO checkpoint')
            expected=dict(suite=suite,tasks=sorted(tasks),arm=arm,fine_tuning_mode=mode,
                          group_size=values['group_size'],seed=seed,learning_rate=lr)
            if any(saved[k]!=v for k,v in expected.items()): raise ValueError('续训须沿用原任务、模型、微调方式、分组帧数、seed 和学习率')
            if values['steps']<=saved['step']: raise ValueError('续训目标总步数须大于 checkpoint 步数')
        if not (Path(checkpoint)/'config.json').is_file(): raise ValueError('初始模型尚未准备')
        config.update(arm=arm,fine_tuning_mode=mode,checkpoint=checkpoint,learning_rate=lr,**values)
        argv=[*prefix[:-1],prefix[-1]+'train',*common,'--checkpoint',checkpoint,'--arm',arm,
              '--fine-tuning-mode',mode,'--seed',str(seed),'--learning-rate',str(lr),
              '--output','{run}/training','--report','{run}/training.json']
        for key,value in values.items(): argv += ['--'+key.replace('_','-'),str(value)]
        stages=[dict(name='LIBERO '+arm+' 训练',argv=argv,report='training.json',steps=values['steps'])]
        need=28000 if mode=='action_full' else 16000
    else:
        checkpoint=request.get('checkpoint')
        saved=next((c for c in info['checkpoints'] if c['path']==checkpoint),None)
        if saved is None: raise ValueError('请先训练，再选择 LIBERO 权重；不能使用 B1K 权重')
        trials=request.get('trials',1); steps=request.get('max_steps',32)
        if type(trials)is not int or not 1<=trials<=50 or type(steps)is not int or not 0<=steps<=HORIZONS[suite]:
            raise ValueError('每任务回合数为 1–50；0 步限制表示使用套件完整时长')
        if not (manager.root/'env/libero-smoke/bin/python').is_file() or not shutil.which('ffmpeg'):
            raise ValueError('LIBERO 仿真环境或 ffmpeg 尚未安装')
        config.update(checkpoint=checkpoint,trials=trials,max_steps=steps,arm=saved['arm'])
        argv=[*prefix[:-1],prefix[-1]+'evaluate',*common,'--checkpoint',checkpoint,'--trials',str(trials),
              '--max-steps',str(steps),'--seed',str(seed),'--output','{run}/evaluation.json']
        stages=[dict(name='加载策略并运行 LIBERO',argv=argv,report='evaluation.json')]
        need=14000
    return dict(config=config,title=TITLES[kind],stages=stages,gpu=device,required_free_mb=need,
                ready=need==0 or device['free_mb']>=need,commands=[shlex.join(s['argv']) for s in stages])
