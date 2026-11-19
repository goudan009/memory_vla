"""Copy an existing converted task to an optional workbench training device.

No training is started. Original metadata, camera videos and Episode split are
retained. Publish readiness only after checking all 200 Episodes / 600 videos.
"""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess


def prepare_dataset(root, device_id, task):
    devices=json.loads((root/'configs/workbench_devices.json').read_text())
    device=devices[device_id]
    work=root/f'outputs/shared/training_dashboard/{device_id}/datasets/task{task}'
    work.mkdir(parents=True,exist_ok=True)
    status=work/'status.json'

    def record(**value):
        pending=status.with_suffix('.pending')
        pending.write_text(json.dumps(dict(task=task,**value),ensure_ascii=False,indent=2))
        pending.replace(status)

    try:
        record(status='syncing')
        source=root/f'assets/datasets/b1k2025_groot_task{task}_v1_200'
        if not (source/'meta/info.json').is_file():
            raise ValueError('请先在 A100 准备该任务；本机尚无已转换的数据：'+str(source))
        destination=device['paths']['datasets']+f'/b1k2025_groot_task{task}_v1_200'
        origin='/nvme3/data_wyh/datasets'+f'/b1k2025_groot_task{task}_v1_200/preparation/data_split.json'
        ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5']
        split=subprocess.check_output([*ssh,devices['a100']['host'],shlex.join(['cat',origin])],text=True,timeout=20)
        # Check the local data against the exact registered split before copying.
        split_path=work/'data_split.json'
        split_path.write_text(split)
        from dataset_inventory import inspect_task
        local=inspect_task(source,task,split_path)
        if not local['ready']:
            raise ValueError('本机转换后的任务数据不完整：'+str(local['missing'][:5]))
        subprocess.run([*ssh,device['host'],shlex.join(['mkdir','-p',destination+'/preparation'])],check=True,timeout=15)
        common=['rsync','-a','--partial','--timeout=60','-e','ssh -o BatchMode=yes -o ConnectTimeout=5']
        subprocess.run([*common,'--exclude=preparation/',str(source)+'/',device['host']+':'+destination+'/'],check=True)
        subprocess.run([*common,str(split_path),device['host']+':'+destination+'/preparation/'],check=True)
        marker=dict(status='completed',task=task,dataset=destination,split=destination+'/preparation/data_split.json',
                    train_episodes=180,validation_episodes=20,completed_at=datetime.datetime.now().isoformat())
        code=(root/'train/dataset_inventory.py').read_text()+'\n'
        code+='check=inspect_task('+repr(destination)+','+str(task)+','+repr(marker['split'])+')\n'
        code+='assert check["ready"], check\n'
        code+='p=Path('+repr(destination+'/preparation/ready.json')+')\n'
        code+='p.write_text('+repr(json.dumps(marker,ensure_ascii=False))+')\nprint(json.dumps(check))\n'
        result=subprocess.check_output([*ssh,device['host'],'python3 -'],input=code,text=True,timeout=30)
        record(**{k:v for k,v in marker.items() if k!='task'},verification=json.loads(result))
        return marker
    except Exception as error:
        record(status='failed',error=str(error))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--device',default='a6000')
    parser.add_argument('--task',type=int,choices=range(50),required=True)
    parser.add_argument('--dataset-only',action='store_true',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare_dataset(args.root,args.device,args.task),ensure_ascii=False))
