"""SSH-side evaluation operations; executed using the 4090's current code."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess


class ObservedProcess:
    def __init__(self, row):
        self.pid=row['pid']; self.row=row

    def poll(self):
        stat=Path(f'/proc/{self.pid}/stat')
        if stat.exists():
            fields=stat.read_text().rsplit(')',1)[1].split()
            if fields[0]!='Z' and fields[19]==self.row['process_start']:
                return None
        result=Path(self.row['output'])/'results.json'
        return 0 if result.exists() and json.loads(result.read_text()).get('status') in ('complete','completed') else 1


def dispatch(request):
    project=Path(request['project']); live=project/'deployments/web_live'
    os.environ.update(request['environment'])
    for variable, relative in [('GROOT_REPO','vendor/Isaac-GR00T'),('OMNIGIBSON_ROOT','vendor/OmniGibson'),('BDDL_ROOT','vendor/bddl')]:
        os.environ[variable]=str(live/relative)
    Path(os.environ['GROOT_EVAL_ROOT']).mkdir(parents=True,exist_ok=True)
    from groot_web_launch import LaunchManager
    record=project/'setup/evaluation_jobs.json'
    record.parent.mkdir(parents=True,exist_ok=True)
    with (record.with_suffix('.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=json.loads(record.read_text()) if record.exists() else []
        action=request['action']; payload=request.get('payload',{})
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
            if not payload.get('resume_from') and payload.get('task',21) not in allowed:
                raise ValueError('该任务资产尚未准备好；当前可用任务：'+','.join(map(str,allowed)))
            manifest=json.loads((live/'web_source_manifest.json').read_text())
            if payload.get('resume_from'):
                original=next((row for row in reversed(rows) if row['output']==payload['resume_from']),None)
                if original is None: raise ValueError('找不到续跑评估的原始源码记录')
                source=Path(original['source'])
                manifest=json.loads((source/'web_source_manifest.json').read_text())
            else:
                source=project/'deployments'/('eval_'+manifest['digest'])
                if not source.exists(): shutil.copytree(live,source,symlinks=True)
            manager=LaunchManager(source,project/'setup/web_eval_logs')
            for variable, relative in [('GROOT_REPO','vendor/Isaac-GR00T'),('OMNIGIBSON_ROOT','vendor/OmniGibson'),('BDDL_ROOT','vendor/bddl')]:
                os.environ[variable]=str(source/relative)
            manager.policy_resolver=lambda host,port: request.get('policy_identity')
            for row in rows:
                if ObservedProcess(row).poll() is None:
                    manager.jobs.append(dict(row,process=ObservedProcess(row),gpus=set(row['gpus']),log_path=Path(row['log'])))
            result=manager.start(payload)
            result.update(source=str(source),process_start=Path(f'/proc/{result["pid"]}/stat').read_text().rsplit(')',1)[1].split()[19])
            result['plan']=manager.jobs[-1]['plan']
            result['source_digest']=manifest['digest']
            rows.append(result)
            pending=record.with_suffix('.pending'); pending.write_text(json.dumps(rows)); pending.replace(record)
            return result
        if action=='snapshot':
            result=[]
            for row in reversed(rows):
                manager=LaunchManager(row['source'])
                status=manager._status(dict(row,process=ObservedProcess(row),gpus=set(row['gpus']),log_path=Path(row['log'])))
                result.append(dict(status,source=row['source']))
            return result
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
