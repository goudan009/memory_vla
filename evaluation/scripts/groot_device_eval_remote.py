"""SSH-side evaluation operations; executed using the 4090's current code."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


class ObservedProcess:
    def __init__(self, row):
        self.pid=row['pid']; self.row=row

    def poll(self):
        try:
            fields=Path(f'/proc/{self.pid}/stat').read_text().rsplit(')',1)[1].split()
            if fields[0]!='Z' and fields[19]==self.row['process_start']:
                return None
        except (FileNotFoundError, ProcessLookupError):
            pass
        result=Path(self.row['output'])/'results.json'
        return 0 if result.exists() and json.loads(result.read_text()).get('status') in ('complete','completed') else 1


def observed_job(row):
    return dict(row, process=ObservedProcess(row), gpus=set(row['gpus']), log_path=Path(row['log']))


def set_runtime_source(source):
    for variable, relative in [('GROOT_REPO', 'vendor/Isaac-GR00T'),
                               ('OMNIGIBSON_ROOT', 'vendor/OmniGibson'), ('BDDL_ROOT', 'vendor/bddl')]:
        os.environ[variable] = str(source/relative)


def set_task40_dependencies(project, dependencies):
    if not dependencies:
        raise ValueError('Task40 采集缺少已同步的运行依赖')
    files = dependencies['files']
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if digest != dependencies['digest']:
        raise ValueError('Task40 运行依赖清单校验失败')
    root = Path(dependencies.get('base', project/'setup/task40_dependencies'))/digest
    for name, expected in files.items():
        path = root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Task40 运行依赖缺失或校验失败：'+name)
    for variable, relative in dependencies['paths'].items():
        os.environ[variable] = str(root/relative)
    if dependencies.get('directed'):
        os.environ['GROOT_TASK40_ARCHIVE_ROOT'] = str(root)


def task40_status(status):
    return dict(status, output_exists=Path(status['output']).is_dir())


def dispatch(request):
    project=Path(request['project']); live=project/'deployments/web_live'
    os.environ.update(request['environment'])
    os.environ['GROOT_TASK40_EXPERIMENT_ROOT'] = request['environment'].get(
        'GROOT_TASK40_EXPERIMENT_ROOT', str(project/'Output/test_01/experiment_09'))
    storage = request.get('task40_baseline_storage')
    if storage:
        os.environ['GROOT_TASK40_BASELINE_STORAGE'] = storage
    set_runtime_source(live)
    Path(os.environ['GROOT_EVAL_ROOT']).mkdir(parents=True,exist_ok=True)
    from groot_web_launch import LaunchManager
    record=project/'setup/evaluation_jobs.json'
    record.parent.mkdir(parents=True,exist_ok=True)
    with (record.with_suffix('.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=json.loads(record.read_text()) if record.exists() else []
        action=request['action']; payload=request.get('payload',{})
        collect=payload.get('task40_collect') is True
        if action in ('start', 'prepare') and collect:
            readiness = project/'setup/simulation_ready.json'
            if not readiness.is_file() or 40 not in json.loads(readiness.read_text()).get('task_ids', []):
                raise ValueError('该设备的 Task40 仿真环境尚未验收完成')
            dependencies = request.get('task40_dependencies')
            set_task40_dependencies(project, dependencies)
            if dependencies.get('directed'):
                payload = dict(payload, restore_checkpoint=dependencies['directed'])
            if storage and payload.get('experiment') == 'experiment_11':
                for name, directory in (('GROOT_CACHE_ROOT', 'simulation'),
                                        ('XDG_CACHE_HOME', 'xdg'), ('CUDA_CACHE_PATH', 'cuda')):
                    cache = Path(storage)/'cache'/directory
                    cache.mkdir(parents=True, exist_ok=True)
                    os.environ[name] = str(cache)
        manager=LaunchManager(live,project/'setup/web_eval_logs')
        manager.policy_resolver=lambda host,port: request.get('policy_identity')
        if action=='options':
            result=manager.options()
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
            result['gpus']=[dict(zip(('index','name','total_mb','used_mb'),
                [int(v.strip()) if i!=1 else v.strip() for i,v in enumerate(line.split(','))])) for line in gpu.splitlines()]
            result['ready']=Path(request['environment']['B1K_PYTHON']).is_file() and (project/'setup/simulation_ready.json').is_file()
            readiness=project/'setup/simulation_ready.json'
            result['verified_tasks']=json.loads(readiness.read_text()).get('task_ids',[]) if readiness.exists() else []
            if result['verified_tasks']:
                result['tasks']=[task for task in result['tasks'] if task['id'] in result['verified_tasks']]
            return result
        if action=='prepare': return manager.prepare(payload)
        if action=='start':
            ready=project/'setup/simulation_ready.json'
            if not ready.is_file(): raise ValueError('该设备的仿真环境尚未验收完成')
            allowed=json.loads(ready.read_text()).get('task_ids',[])
            task_id = 40 if collect else payload.get('task',21)
            if not payload.get('resume_from') and task_id not in allowed:
                raise ValueError('该任务资产尚未准备好；当前可用任务：'+','.join(map(str,allowed)))
            manifest=json.loads((live/'web_source_manifest.json').read_text())
            if payload.get('resume_from'):
                original=next((row for row in reversed(rows) if row['output']==payload['resume_from']),None)
                if original is None: raise ValueError('找不到续跑评估的原始源码记录')
                source=Path(original['source'])
                manifest=json.loads((source/'web_source_manifest.json').read_text())
            else:
                source=project/'deployments'/('eval_'+manifest['digest'])
                if not source.exists():
                    shutil.copytree(live, source, symlinks=True,
                                    ignore=lambda path, names: {'Output'} if Path(path) == live else set())
            manager=LaunchManager(source,project/'setup/web_eval_logs')
            set_runtime_source(source)
            manager.policy_resolver=lambda host,port: request.get('policy_identity')
            for row in rows:
                if ObservedProcess(row).poll() is None:
                    manager.jobs.append(observed_job(row))
            result=manager.start(payload)
            result.update(source=str(source),process_start=Path(f'/proc/{result["pid"]}/stat').read_text().rsplit(')',1)[1].split()[19])
            result['plan']=manager.jobs[-1]['plan']
            result['source_digest']=manifest['digest']
            if collect:
                result['task40_dependency_digest']=request['task40_dependencies']['digest']
                result=task40_status(result)
            rows.append(result)
            pending=record.with_suffix('.pending'); pending.write_text(json.dumps(rows)); pending.replace(record)
            return result
        if action=='snapshot':
            result=[]
            for row in reversed(rows):
                manager=LaunchManager(row['source'])
                status=manager._status(observed_job(row))
                status=dict(status,source=row['source'],source_digest=row.get('source_digest'))
                if row.get('kind') == 'task40_collect':
                    status=task40_status(dict(status,plan=row['plan'],
                                              task40_dependency_digest=row.get('task40_dependency_digest')))
                result.append(status)
            return result
        if action=='stop_restore':
            row=next((r for r in reversed(rows)
                      if r.get('kind')=='task40_collect' and r['id']==payload.get('job')),None)
            if row is None: raise ValueError('未找到该设备上的 Task40 采集记录')
            manager=LaunchManager(row['source'])
            manager.jobs.append(observed_job(row))
            return manager.stop_restore(payload)
        if action in ('pause','delete'):
            row=next((r for r in reversed(rows) if r['folder']==payload['run']),None)
            if row is None: raise ValueError('未找到该设备上的评估记录')
            manager=LaunchManager(row['source'])
            result=getattr(manager,action)(payload)
            if action=='delete':
                rows=[r for r in rows if r['folder']!=payload['run']]
                record.write_text(json.dumps(rows))
            return result
        raise ValueError('未知设备评估操作')
