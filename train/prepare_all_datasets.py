"""Persistently prepare missing A100 B1K tasks, one task at a time on the 4090 host."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_web_dataset import record


def inventory(root, host, register=True):
    code=(root/'train/dataset_inventory.py').read_text()
    code += '\nprint(json.dumps(remote_inventory("/nvme3/data_wyh/datasets",'
    code += '"/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training",register='+repr(register)+')))\n'
    result = subprocess.run(['ssh','-o','BatchMode=yes',host,'python3 -'],
                            input=code,capture_output=True,text=True,check=True,timeout=90)
    return json.loads(result.stdout)


def run(root, host, workers):
    root=Path(root).resolve()
    directory=root/'outputs/shared/training_dashboard/datasets/all_tasks'
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'status.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    if previous.get('pid') and previous['pid'] != os.getpid():
        try:
            os.kill(previous['pid'],0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Data preparation queue is already alive')
    state=dict(status='checking',pid=os.getpid(),host=host,workers=workers,ready_tasks=[],pending_tasks=[],failed_tasks=[])
    record(path,**state)
    rows=inventory(root,host)
    record(directory/'inventory.json',checked_at=datetime.now().isoformat(),tasks=rows)
    state['ready_tasks']=[r['task'] for r in rows if r['ready']]
    state['pending_tasks']=[r['task'] for r in rows if not r['ready']]
    for position,task in enumerate(state['pending_tasks']):
        work=root/f'outputs/shared/training_dashboard/datasets/task{task}'
        status=work/'status.json'
        prior=json.loads(status.read_text()) if status.exists() else {}
        if prior.get('status') in ['converting','syncing','starting']:
            try: os.kill(prior['pid'],0)
            except (ProcessLookupError,KeyError): pass
            else: raise RuntimeError(f'Task {task} already being prepared by PID '+str(prior['pid']))
        record(status,status='queued',task=task,queue_position=position+1,queue=str(path),
               log=str(work/'prepare.log'))
    tasks=list(state['pending_tasks'])
    python=root/'src/v1_groot_memoryvla/.venv/bin/python'
    for task in tasks:
        work=root/f'outputs/shared/training_dashboard/datasets/task{task}'
        state.update(status='preparing',current_task=task,updated_at=datetime.now().isoformat())
        record(path,**state)
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 NO_ALBUMENTATIONS_UPDATE='1',PYTHONUNBUFFERED='1')
        argv=[str(python),'-u',str(root/'train/prepare_web_dataset.py'),'--root',str(root),
              '--host',host,'--task',str(task),'--workers',str(workers)]
        with (work/'prepare.log').open('ab') as log:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
            state['worker_pid']=child.pid
            while child.poll() is None:
                videos=root/f'assets/datasets/b1k2025_groot_task{task}_v1_200/videos/task-{task:04d}'
                state.update(current_videos=sum(1 for p in videos.glob('*/*.mp4') if not p.name.endswith('.partial.mp4')),
                             total_videos=600,updated_at=datetime.now().isoformat())
                record(path,**state)
                time.sleep(5)
        state['pending_tasks'].remove(task)
        if child.returncode == 0:
            state['ready_tasks'].append(task)
        else:
            state['failed_tasks'].append(task)
        record(path,**state)
        print(json.dumps(dict(task=task,returncode=child.returncode,ready_count=len(state['ready_tasks']),
                              remaining=len(state['pending_tasks']))),flush=True)
    final=inventory(root,host)
    record(directory/'inventory.json',checked_at=datetime.now().isoformat(),tasks=final)
    state.update(status='completed' if all(r['ready'] for r in final) else 'completed_with_errors',
                 ready_tasks=[r['task'] for r in final if r['ready']],
                 failed_tasks=[r['task'] for r in final if not r['ready']],
                 completed_at=datetime.now().isoformat())
    record(path,**state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--host',required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    os.nice(10)
    run(args.root,args.host,args.workers)
