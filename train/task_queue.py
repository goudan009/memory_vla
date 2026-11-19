"""Run requested task comparisons, two arms at a time, via the public CLI."""

import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import time


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    pending=path.with_suffix('.pending')
    pending.write_text(json.dumps(value,indent=2))
    pending.replace(path)


def run(config_path):
    config=read(config_path)
    directory=config_path.parent
    state_file=directory/'status.json'
    state=read(state_file)
    if state:
        raise ValueError('Queue already submitted; inspect its status rather than launching twice')
    state=dict(status='waiting_for_data',pid=os.getpid(),tasks=config['tasks'],runs=[],source=config['source'],
        global_decisions=32*config['global_windows_per_step'],steps_each=3000)
    write(state_file,state)
    source=Path(config['source'])
    spec=importlib.util.spec_from_file_location('queue_remote',source/'train/dashboard_remote.py')
    remote=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
    remote.LIVE=source
    remote.request_digest=read(source/'web_source_manifest.json')['digest']
    try:
        previous=Path(config['after_run'])
        while not (read(previous/'memory_v1/status.json',{}).get('training_complete') and
                   (previous/'memory_v1/checkpoints/step-003000/config.json').is_file()):
            state.update(status='waiting_for_previous_training')
            write(state_file,state);time.sleep(10)
        # Do not add new validators until the previous queued validation has ended.
        while read(previous/'status.json',{}).get('status')!='completed':
            prior=read(previous/'status.json',{})
            if prior.get('status')=='failed':
                raise RuntimeError('Previous Task0 controller failed; see its logs')
            state.update(status='waiting_for_previous_validation')
            write(state_file,state);time.sleep(10)
        for task in config['tasks']:
            ready=remote.DATASETS/f'b1k2025_groot_task{task}_v1_200/preparation/ready.json'
            while not ready.exists():
                state.update(status='waiting_for_data',task=task)
                write(state_file,state);time.sleep(10)
            data=read(ready)
            # Both arms share the same GPU mapping and global decision budget.
            # Independent validators occupy GPU0; existing services are preserved.
            expected=set(config['allowed_existing_pids'])
            active=set(remote.compute_pids(set(config['gpus']+[config['validation_gpu']])))
            if active-expected:
                raise RuntimeError('New GPU processes appeared; no process was stopped: '+str(sorted(active-expected)))
            pair=[]
            for arm in ['baseline','memory_v1']:
                request=dict(task=task,task_name=config['task_names'][str(task)],dataset=data['dataset'],split=data['split'],
                    version='V_1',checkpoint=str(remote.OFFICIAL),gpus=config['gpus'],validation_gpu=config['validation_gpu'],
                    arm=arm,steps=3000,global_windows_per_step=config['global_windows_per_step'],
                    microbatch=4,seed=42,learning_rate=1e-5,share_existing=True)
                launched=remote.start(request)
                pair.append(launched)
                state['runs'].append(launched)
                state.update(status='training',task=task)
                write(state_file,state)
                print(json.dumps(dict(event='submitted',**launched)),flush=True)
            while True:
                complete=[]
                for job in pair:
                    root=Path(job['output'])
                    current=read(root/'status.json',{})
                    training=read(root/job['arm']/'status.json',{})
                    if current.get('status')=='failed':
                        raise RuntimeError(job['run']+' failed; inspect '+str(root/'logs'))
                    if current.get('status')!='completed' and not Path('/proc/'+str(job['pid'])).exists():
                        raise RuntimeError(job['run']+' controller exited before completion')
                    job.update(status=current.get('status','submitted'),step=training.get('step',0),
                        training_complete=training.get('training_complete',False))
                    complete.append(current.get('status')=='completed' and training.get('step')==3000)
                write(state_file,state)
                if all(complete):break
                time.sleep(10)
            state.update(status='between_tasks')
            write(state_file,state)
        state.update(status='completed',completed_at=datetime.datetime.now().isoformat())
        write(state_file,state)
    except Exception as error:
        state.update(status='failed',error=str(error))
        write(state_file,state)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config)
