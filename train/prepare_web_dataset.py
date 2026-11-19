"""Prepare one of the 50 local B1K tasks and register its A100 dataset."""

import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from dataset_inventory import inspect_task


def record(path, **value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    temporary.replace(path)


def prepare(root, host, task, workers=8):
    root=Path(root).resolve()
    if type(task) is not int or not 0<=task<50:
        raise ValueError('Task必须在0–49之间')
    sys.path.insert(0,str(root/'src'))
    from v1_groot_memoryvla.data.prepare_task21_formal import prepare as convert
    work=root/f'outputs/shared/training_dashboard/datasets/task{task}'
    status=work/'status.json'
    raw=root/'assets/datasets/b1k2025'
    destination=root/f'assets/datasets/b1k2025_groot_task{task}_v1_200'
    config=work/'config'
    remote=f'/nvme3/data_wyh/datasets/b1k2025_groot_task{task}_v1_200'
    try:
        existing=json.loads(status.read_text()) if status.exists() else {}
        if existing.get('status')=='completed':
            return existing
        record(status,status='converting',task=task,pid=os.getpid(),log=str(work/'prepare.log'))
        checkpoint=config/'official_statistics'
        checkpoint.mkdir(parents=True,exist_ok=True)
        statistics=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',host,
            'cat /nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k/statistics.json'],text=True,timeout=30)
        record(checkpoint/'statistics.json',**json.loads(statistics))
        convert(raw,destination,checkpoint,config,workers=workers,seed=42,task=task)
        local_check = inspect_task(destination, task, config/'data_split.json')
        assert local_check['ready'], local_check
        record(status,status='syncing',task=task,pid=os.getpid(),dataset=str(destination),remote_dataset=remote,log=str(work/'prepare.log'))
        subprocess.run(['ssh','-o','BatchMode=yes',host,shlex.join(['mkdir','-p',remote+'/preparation'])],check=True,timeout=15)
        subprocess.run(['rsync','-a','--timeout=60','-e','ssh -o BatchMode=yes',str(destination)+'/',host+':'+remote+'/'],check=True)
        subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes',str(config/'data_split.json'),str(config/'conversion.json'),
            host+':'+remote+'/preparation/'],check=True)
        inventory_source=(root/'train/dataset_inventory.py').read_text()
        verify=inventory_source+'\nprint(json.dumps(inspect_task('+repr(remote)+','+str(task)+','+repr(remote+'/preparation/data_split.json')+')))\n'
        checked=subprocess.check_output(['ssh','-o','BatchMode=yes',host,'python3 -'],input=verify,text=True,timeout=60)
        assert json.loads(checked)['ready'], checked
        complete=dict(status='completed',task=task,dataset=remote,split=remote+'/preparation/data_split.json',
            train_episodes=180,validation_episodes=20,completed_at=datetime.datetime.now().isoformat())
        # This marker is published after all data and configuration transfers.
        payload=json.dumps(complete)
        code='from pathlib import Path\np=Path('+repr(remote+'/preparation/ready.json')+')\np.write_text('+repr(payload)+')'
        subprocess.run(['ssh','-o','BatchMode=yes',host,'python3 -'],input=code,text=True,check=True,timeout=15)
        record(status,**complete)
        print(json.dumps(complete),flush=True)
        return complete
    except Exception as error:
        record(status,status='failed',task=task,error=str(error),log=str(work/'prepare.log'))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--host',required=True)
    parser.add_argument('--task',type=int,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    prepare(args.root,args.host,args.task)
