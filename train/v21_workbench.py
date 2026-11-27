"""V2_1 adapter for the existing training workbench and resource queue.

No model imports. The controller invokes the same torchrun CLI in an immutable
workbench source snapshot, records progress, and never signals other jobs.
"""
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from run_identity import canonical_version, validate_run_identity, run_fields, NAMING_SCHEMA

VERSION = 'V_2_1'
LEGACY_VERSION = 'v2.1'
TERMINAL = {'completed', 'complete', 'failed', 'stopped', 'paused', 'cancelled'}
RUN = re.compile(r'v2_1__test[0-9]{2}__exp[0-9]{2}__run(?!000)[0-9]{3}(?:__preflight[0-9]{3})?')
LEGACY_RUN = re.compile(r'v2\.1__test01__exp14__run(?!000)[0-9]{3}(?:__preflight[0-9]{3})?')


def valid_run(name):
    return isinstance(name, str) and bool(RUN.fullmatch(name) or LEGACY_RUN.fullmatch(name))


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    pending.replace(path)


def now():
    return datetime.datetime.now().isoformat()


class Backend:
    def __init__(self, remote):
        self.r = remote
        self.training = Path(remote['TRAINING'])
        self.project = Path(remote['PROJECT'])
        self.live = Path(remote['LIVE'])
        self.python = Path(remote['PYTHON'])

    def owns(self, request):
        if canonical_version(request.get('version')) == VERSION:
            return True
        name = request.get('run', '')
        if not valid_run(name):
            return False
        return canonical_version(read(self.training/name/'config/web_launch.json', {}).get('version')) == VERSION

    def root(self, request):
        name = request.get('run', '')
        if not valid_run(name):
            raise ValueError('V2_1 run_id格式错误')
        root = self.training/name
        launch = read(root/'config/web_launch.json', {})
        if root.is_symlink() or root.resolve().parent != self.training.resolve() or canonical_version(launch.get('version')) != VERSION:
            raise ValueError('V2_1运行不存在或路径不正确')
        if request.get('arm', launch['arm']) != launch['arm']:
            raise ValueError('V2_1训练组不匹配')
        return root, launch

    def prepare(self, request):
        if request.get('recipe') == 'native_random_v2':
            # Explicitly refuse rather than silently falling back to the old
            # fixed sequential recipe, which has a different LR and sampler.
            if canonical_version(request.get('version')) != VERSION:
                raise ValueError('native_random_v2须使用已登记的V_2_1')
            fields=validate_run_identity(request.get('run_id'), request['version'],
                                         preflight=request.get('preflight_only',False))
            if (fields['test_id'],fields['experiment_id'])!=('test02','exp01'):
                raise ValueError('当前重训任务属于test02/exp01；不得猜测或迁移身份')
            raise ValueError('BLOCKED_WORKBENCH_INTEGRATION: native_random_v2尚未通过原生训练器严格暂停恢复及规范checkpoint发布验收；禁止旧配方回退或CLI绕行')
        allowed = {'version','task','stage','run_id','dataset','checkpoint','labels','gpus','steps','seed',
                   'microbatch','learning_rate','global_windows_per_step','prior','fusion_layers',
                   'weak_weight','state_weight','states','split','calibration_reference','preflight_only',
                   'accept_preflight','enqueue','share_existing','notes','arm'}
        if set(request)-allowed:
            raise ValueError('不支持的V2_1参数: '+','.join(sorted(set(request)-allowed)))
        if canonical_version(request.get('version')) != VERSION or request.get('task',40) != 40:
            raise ValueError('该入口仅支持V2_1 Task40')
        stage = request.get('stage','stage1')
        if stage not in ('stage1','memory','continued'):
            raise ValueError('invalid V2_1 stage')
        preflight = request.get('preflight_only',False)
        if type(preflight) is not bool:
            raise ValueError('preflight_only必须为开关')
        run = request.get('run_id')
        if not isinstance(run,str) or not RUN.fullmatch(run) or ('__preflight' in run) != preflight:
            raise ValueError('新训练标识须为v2_1__testNN__expNN__runNNN，预检追加__preflightNNN')
        identity = validate_run_identity(run, request['version'], preflight=preflight)
        if any(p.exists() for p in (self.training/run,self.training/'.trash'/run,self.project/'deployments'/run)):
            raise ValueError('该训练标识已存在，请从原作业续跑')
        fixed = {'seed':42,'microbatch':2,'learning_rate':1e-5,'global_windows_per_step':2,
                 'weak_weight':.5,'state_weight':.01}
        for key,value in fixed.items():
            if request.get(key,value) != value or isinstance(request.get(key),bool):
                raise ValueError('V2_1研究参数固定: '+key)
        budget = 6000 if stage == 'stage1' else 3000
        if request.get('steps',budget) != budget:
            raise ValueError('V2_1训练预算固定；preflight_only由训练器限制实际更新次数')
        gpus = request.get('gpus')
        if not isinstance(gpus,list) or len(gpus)!=4 or len(set(gpus))!=4 or any(type(g) is not int for g in gpus):
            raise ValueError('V2_1 A100适配要求四个不同GPU rank')
        if not set(gpus) <= {g['index'] for g in self.r['gpu_info']()}:
            raise ValueError('所选GPU不存在')
        for key in ('enqueue','share_existing'):
            if type(request.get(key,False)) is not bool:
                raise ValueError(key+'必须为开关')
        occupied = self.r['compute_pids'](gpus)
        if occupied and not request.get('share_existing',False) and not request.get('enqueue',False):
            raise ValueError('所选GPU有现有进程，使用已有队列等待，不会停止其他作业')
        def path(key, default=None):
            value = request.get(key,default)
            if not isinstance(value,str) or not value.startswith('/') or any(c in value for c in ('\x00','\n','\r')):
                raise ValueError(key+'必须为目标设备绝对路径')
            return Path(value)
        dataset = path('dataset')
        checkpoint = path('checkpoint',str(self.r['BASE_CHECKPOINT']))
        labels = path('labels')
        if not (dataset/'meta/episodes.jsonl').is_file() or not labels.is_file() or not self.python.is_file():
            raise ValueError('数据、标签或Python未就绪')
        episodes = [json.loads(x) for x in (dataset/'meta/episodes.jsonl').read_text().splitlines() if x.strip()]
        if len(episodes)!=200 or len({e['episode_index'] for e in episodes})!=200:
            raise ValueError('Task40需要200条唯一示教')
        config = read(checkpoint/'config.json')
        if config is None or config.get('memory_enabled') or config.get('finetuning_lora'):
            raise ValueError('V2_1需要原始完整无记忆GR00T权重')
        index = read(checkpoint/'model.safetensors.index.json',{})
        weights = set(index.get('weight_map',{}).values()) or {'model.safetensors'}
        if any(not (checkpoint/p).is_file() for p in weights):
            raise ValueError('GR00T权重文件不完整')
        if stage == 'stage1' and checkpoint.resolve()!=Path(self.r['BASE_CHECKPOINT']).resolve():
            raise ValueError('V2_1 Stage1必须使用通用GR00T-N1.6-3B起点')
        if stage != 'stage1':
            meta=read(checkpoint/'v2_1_stage1.json',{})
            complete=read(checkpoint/'COMPLETE.json',{})
            if meta.get('step')!=6000 or complete.get('step')!=6000:
                raise ValueError('后续阶段需要有效Stage1 step6000')
        prior=request.get('prior','topology')
        fusion=request.get('fusion_layers',0)
        if prior not in ('topology','fk','none') or type(fusion) is not int or fusion<0:
            raise ValueError('V2_1先验或融合层配置错误')
        arm = 'memory_v1' if stage=='memory' else 'baseline'
        if request.get('arm',arm)!=arm:
            raise ValueError('arm与stage不一致')
        plan=dict(request,version=VERSION,**identity,task=40,stage=stage,arm=arm,run_name=run,training_id=run,
                  method='memory' if stage=='memory' else 'baseline',
                  dataset=str(dataset),checkpoint=str(checkpoint),labels=str(labels),gpus=gpus,steps=budget,
                  preflight_only=preflight,enqueue=request.get('enqueue',False),share_existing=request.get('share_existing',False),
                  existing_pids=occupied,global_decisions=64,train_episodes=180,validation_episodes=20,
                  validation_enabled=True,asynchronous_validation=False,validation_gpu=None,
                  checkpoint_interval=1000,category='engineering_preflight' if preflight else 'formal_training',
                  naming_schema=NAMING_SCHEMA,**fixed)
        for key in ('split','calibration_reference','accept_preflight'):
            if request.get(key) is not None:
                value=path(key)
                if not value.is_file():raise ValueError(key+'未就绪')
                plan[key]=str(value)
        if not preflight and not plan.get('accept_preflight'):
            raise ValueError('正式训练需要独立归档的preflight.json')
        return plan

    def command(self, plan, source, root, resume=None):
        argv=[str(self.python),'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
              '-m','v2_1_groot_memoryvla.train']
        values={'stage':plan['stage'],'dataset':plan['dataset'],'checkpoint':plan['checkpoint'],
                'labels':plan['labels'],'output':str(root/'run'),'steps':plan['steps'],'seed':plan['seed'],
                'microbatch':plan['microbatch'],'lr':plan['learning_rate'],'prior':plan.get('prior','topology'),
                'fusion-layers':plan.get('fusion_layers',0),'weak-weight':plan['weak_weight'],'state-weight':plan['state_weight']}
        for key in ('split','calibration_reference','accept_preflight'):
            if plan.get(key):values[key.replace('_','-')]=plan[key]
        values['checkpoint-prefix']=plan['run_name']
        if resume:values['resume']=str(resume)
        for key,value in values.items():argv += ['--'+key,str(value)]
        if plan.get('states'):argv += ['--states',*plan['states']]
        if plan['preflight_only']:argv.append('--preflight-only')
        argv.append('--execute')
        return argv

    def launch(self, root, plan, source, argv, resumed=False):
        config=root/'config/v21_controller.json'
        overrides=dict(PYTHONPATH=str(source/'src'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES=','.join(map(str,plan['gpus'])),
                 OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        env=dict(os.environ,**overrides)
        write(config,dict(root=str(root),arm=plan['arm'],argv=argv,env=overrides,source=str(source),steps=plan['steps'],
                          preflight_only=plan['preflight_only']))
        controller=[str(self.python),'-u',str(source/'train/v21_workbench.py'),'--controller',str(config)]
        if plan['enqueue']:
            queue=self.training/'queues'/root.name
            write(queue/'config.json',dict(output=str(root),arm=plan['arm'],argv=controller,gpus=plan['gpus'],
                    allowed_processes={str(p):self.r['process_start'](p) for p in plan['existing_pids']} if plan['share_existing'] else {}))
            write(queue/'status.json',dict(status='queued',jobs=[dict(run=root.name,arm=plan['arm'])],created_at=now()))
            controller=[str(self.python),'-u',str(source/'train/web_training_queue.py'),'--config',str(queue/'config.json')]
        write(root/plan['arm']/'status.json',dict(status='queued' if plan['enqueue'] else 'submitted',step=0,total_steps=plan['steps']))
        (root/'logs').mkdir(exist_ok=True)
        with (root/'logs/controller.log').open('ab') as log:
            proc=subprocess.Popen(controller,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        job=dict(run=root.name,version=VERSION,arm=plan['arm'],pid=proc.pid,pid_start=self.r['process_start'](proc.pid),
                 output=str(root),source=str(source),source_digest=plan['source_manifest']['digest'],status='queued' if plan['enqueue'] else 'submitted',
                 resumed=resumed,category=plan['category'],created_at=now())
        write(root/'config/web_process.json',job)
        if plan['enqueue']:
            write(queue/'process.json',dict(pid=proc.pid,argv=controller))
        return job

    def start(self, request, digest):
        self.training.mkdir(parents=True,exist_ok=True)
        with (self.training/'.gpu_queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            plan=self.prepare(request)
            manifest=read(self.live/'web_source_manifest.json',{})
            if not digest or manifest.get('digest')!=digest:
                raise ValueError('同步源码版本不匹配')
            source=self.project/'deployments'/plan['run_name']
            root=self.training/plan['run_name']
            root.mkdir()
            shutil.copytree(self.live,source,symlinks=True)
            argv=self.command(plan,source,root)
            plan.update(source=str(source),source_manifest=manifest,argv=argv,created_at=now())
            write(root/'config/web_launch.json',plan)
            write(root/'config/source_manifest.json',manifest)
            return self.launch(root,plan,source,argv)

    def checkpoints(self, root):
        result=[]
        for marker in (root/'run/checkpoints').glob('*/COMPLETE.json'):
            complete=read(marker,{})
            if type(complete.get('step')) is int and (marker.parent/'training_state.pt').is_file():
                result.append(dict(step=complete['step'],path=str(marker.parent)))
        return sorted(result,key=lambda x:x['step'],reverse=True)

    def live_processes(self, root):
        """Include orphaned ranks whose --output is the nested trainer directory."""
        found=set(self.r['live_run_processes'](root))
        for file in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                argv=file.read_bytes().decode().strip('\x00').split('\x00')
                for flag,target in (('--output',str(root/'run')),('--controller',str(root/'config/v21_controller.json'))):
                    if flag in argv and argv[argv.index(flag)+1]==target:
                        found.add(int(file.parent.name))
            except (OSError,UnicodeError,IndexError):
                continue
        return sorted(found)

    def resume_info(self, root, launch, state):
        checkpoints=self.checkpoints(root)
        latest=checkpoints[0] if checkpoints else None
        allowed=state.get('status') in TERMINAL and latest is not None and latest['step']<launch['steps'] and not launch['preflight_only']
        return dict(available=allowed,reason='' if allowed else '需要已停止的正式作业与完整训练断点',
                    checkpoint=latest['path'] if latest else None,step=latest['step'] if latest else 0,
                    source=launch['source'],gpus=launch['gpus'],default_steps=launch['steps'],min_steps=launch['steps'],
                    fixed_budget=True,validation_enabled=True,validation_gpu=None,checkpoint_interval=1000,
                    default_checkpoint_interval=1000,checkpoint_interval_editable=False)

    def jobs(self):
        result=[]
        for file in sorted(self.training.glob('*/config/web_launch.json'),reverse=True):
            launch=read(file,{})
            if canonical_version(launch.get('version')) != VERSION:continue
            root=file.parents[1];arm=launch['arm']
            if (root/'config'/('dashboard_deleted_'+arm+'.json')).exists():continue
            state=read(root/arm/'status.json',{})
            process=read(root/'config/web_process.json',{})
            queue=read(self.training/'queues'/root.name/'status.json',{})
            if state.get('status') in ('queued','submitted') and queue.get('status') in ('queued','waiting_for_gpus','failed'):
                state=dict(state,status=queue['status'])
            if state.get('status') not in TERMINAL and process.get('pid') and self.r['process_start'](process['pid']) is None:
                state=dict(state,status='failed',error='Controller exited without a terminal record')
            status=state.get('status','submitted');terminal=status in TERMINAL
            result.append(dict(run=root.name,name=root.name,version=VERSION,model_version=VERSION,task=40,task_id=40,
                 arm=arm,method=launch['method'],training_id=root.name,train_number=launch['train_number'],
                 status=status,step=state.get('step',0),max_steps=launch['steps'],gpus=','.join(map(str,launch['gpus'])),
                 output=str(root),source=launch['source'],source_digest=launch['source_manifest']['digest'],
                 category=launch['category'],training_complete=status=='completed',validation={'status':'in_process'},
                 can_delete=terminal,can_delete_files=terminal,can_stop=status in ('training','running','validating','submitted'),
                 can_cancel=status in ('queued','waiting_for_gpus'),resume=self.resume_info(root,launch,state),
                 checkpoint_interval=1000,notes=launch.get('notes',''),queue=queue or None,
                 delete_block_reason='' if terminal else '请等待作业停止',validation_enabled=True,last=state.get('last')))
        return result

    def detail(self, request):
        root,launch=self.root(request)
        metrics=[]
        file=root/'run/metrics.jsonl'
        if file.is_file():
            for line in file.read_text().splitlines():
                try:metrics.append(json.loads(line))
                except ValueError:continue
        return dict(run=root.name,arm=launch['arm'],config=launch,state=read(root/launch['arm']/'status.json',{}),
             checkpoints=self.checkpoints(root),checkpoint=str(root/'run/checkpoints'),
             curve=[[r['step'],r.get('fm',0)+launch['state_weight']*r.get('state',0),1] for r in metrics if 'step' in r],
             loss_series=[[r['step'],r.get('fm',0)+launch['state_weight']*r.get('state',0),1] for r in metrics if 'step' in r],
             metrics=metrics[-100:],preflight=read(root/'run/preflight.json'),
             log_tail=self.r['tail'](root/'logs/controller.log'),epochs=[],validation=[],
             initial_weights=dict(checkpoint=launch['checkpoint'],label='V2_1 原始起点',mode='fresh'),
             training_parameters=dict(values=launch,sources=[str(root/'config/web_launch.json')]),
             resume=self.resume_info(root,launch,read(root/launch['arm']/'status.json',{})))

    def stop(self, request):
        root,launch=self.root(request)
        state=read(root/launch['arm']/'status.json',{})
        if state.get('status') in TERMINAL:raise ValueError('该作业已结束')
        current=state.get('step',0)
        target=request.get('step',current+1)
        if request.get('next_checkpoint'):target=min((current//1000+1)*1000,launch['steps'])
        if type(target) is not int or not current<target<=launch['steps']:
            raise ValueError('停止点须为未来optimizer步且不超过预算')
        control=dict(stop_after_step=target,reason='Workbench V2_1 checkpoint-boundary pause',requested_at=now())
        write(root/'run/stop.json',control)
        return control

    def resume(self, request):
        root,launch=self.root(request)
        with (self.training/'.gpu_queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=read(root/launch['arm']/'status.json',{})
            info=self.resume_info(root,launch,state)
            if not info['available']:raise ValueError(info['reason'])
            if self.live_processes(root):raise ValueError('原作业仍有活跃进程')
            if request.get('steps',launch['steps'])!=launch['steps'] or request.get('gpus',launch['gpus'])!=launch['gpus']:
                raise ValueError('V2_1续训保留原预算与四rank')
            occupied=self.r['compute_pids'](launch['gpus'])
            enqueue=request.get('enqueue',launch['enqueue'])
            if type(enqueue) is not bool:raise ValueError('enqueue须为开关')
            if occupied and not enqueue:raise ValueError('GPU有现有进程，请使用队列等待')
            preview=dict(run=root.name,arm=launch['arm'],steps=launch['steps'],resume_step=info['step'],
                         checkpoint=info['checkpoint'],source=launch['source'],gpus=launch['gpus'],enqueue=enqueue)
            if request.get('preview'):return preview
            stop=root/'run/stop.json'
            if stop.exists():stop.rename(root/'config'/('stop-before-resume-'+str(info['step'])+'.json'))
            attempt=len(list((root/'config').glob('resume-event-*.json')))+1
            write(root/'config'/f'resume-event-{attempt:03d}.json',dict(preview,requested_at=now()))
            launch=dict(launch,enqueue=enqueue,existing_pids=occupied)
            return self.launch(root,launch,Path(launch['source']),self.command(launch,Path(launch['source']),root,info['checkpoint']),True)

    def delete(self, request):
        root,launch=self.root(request)
        with (self.training/'.gpu_queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=read(root/launch['arm']/'status.json',{})
            if state.get('status') not in TERMINAL or self.live_processes(root):
                raise ValueError('请先等待作业终止且GPU进程退出')
            remove=request.get('delete_files',False)
            if type(remove) is not bool:raise ValueError('delete_files须为开关')
            if not remove:
                write(root/'config'/('dashboard_deleted_'+launch['arm']+'.json'),dict(status='deleted',deleted_at=now()))
                return dict(run=root.name,status='deleted',files_preserved=True)
            if request.get('expected_output')!=str(root):raise ValueError('输出目录确认不匹配')
            # Any queued or completed consumer keeps its source checkpoint alive.
            for file in self.training.glob('*/config/web_launch.json'):
                if file.parents[1]==root:continue
                other=read(file,{})
                for key in ('checkpoint','labels','accept_preflight','calibration_reference'):
                    value=other.get(key)
                    if value and Path(value).resolve().is_relative_to(root.resolve()):
                        raise ValueError('其他作业仍引用该输出: '+file.parents[1].name)
            purge=request.get('purge_preflight',False)
            if type(purge) is not bool:raise ValueError('purge_preflight须为开关')
            control=dict(run=root.name,arm=launch['arm'],status='deleted',output=str(root),deleted_at=now())
            if purge:
                evidence=request.get('evidence_path')
                if not launch['preflight_only'] or not isinstance(evidence,str):
                    raise ValueError('永久清理仅限已归档证据的预检作业')
                path=Path(evidence)
                if not path.is_absolute() or path.resolve().is_relative_to(root.resolve()) or not path.is_file() or not read(path):
                    raise ValueError('必须先将非空JSON证据归档到预检目录外')
                source=Path(launch['source'])
                deployments=self.project/'deployments'
                if source.exists() and (source.is_symlink() or source.resolve().parent!=deployments.resolve() or source.name!=root.name):
                    raise ValueError('预检源码快照归属不匹配')
                if path.resolve().is_relative_to(source.resolve()):
                    raise ValueError('证据不能保存在待删除的源码快照内')
                for file in self.training.glob('*/config/web_launch.json'):
                    if file.parents[1]!=root and read(file,{}).get('source')==str(source):
                        raise ValueError('其他作业仍使用该源码快照')
                control.update(evidence_path=str(path),purged=True,source=str(source))
                write(self.training/'deletions'/(root.name+'.json'),control)
                shutil.rmtree(root)
                if source.is_dir():shutil.rmtree(source)
                return control
            shutil.rmtree(root)
            return dict(control,files_preserved=False)


def controller(config):
    """Monitor the actual torchrun child; the workbench queue remains scheduler."""
    root=Path(config['root']);arm=config['arm'];output=root/'run'
    state=dict(status='training',step=0,total_steps=config['steps'],started_at=now())
    write(root/arm/'status.json',state)
    child=subprocess.Popen(config['argv'],env=dict(os.environ,**config['env']),stdin=subprocess.DEVNULL,cwd=config['source'])
    while True:
        native=read(output/'status.json',{})
        if native:state.update(native)
        metric=output/'metrics.jsonl'
        if metric.is_file():
            with metric.open('rb') as handle:
                handle.seek(0,2);handle.seek(max(0,handle.tell()-32000))
                for line in reversed(handle.read().decode(errors='replace').splitlines()):
                    try:row=json.loads(line)
                    except ValueError:continue
                    if 'step' in row:state.update(step=row['step'],last=row);break
        write(root/arm/'status.json',dict(state,checked_at=now()))
        code=child.poll()
        if code is not None:break
        time.sleep(1)
    native=read(output/'status.json',{})
    if native:state.update(native)
    preflight=read(output/'preflight.json',{})
    if code:
        state.update(status='failed',returncode=code)
    elif config['preflight_only']:
        state.update(status='completed' if preflight.get('passed') else 'failed',preflight_passed=preflight.get('passed',False))
    else:
        state.update(status='completed' if state['step']>=config['steps'] else 'stopped')
    state.update(finished_at=now(),training_complete=state['status']=='completed')
    write(root/arm/'status.json',state)
    write(root/'status.json',state)
    return code if code else (1 if state['status']=='failed' else 0)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--controller',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(controller(read(args.controller)))
