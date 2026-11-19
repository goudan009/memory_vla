"""Wait for selected GPU resources, then invoke the single public training CLI."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time

from dashboard_remote import atomic_json, read_json, compute_pids, process_start, WAITING_QUEUE_STATES


def blockers(training, directory, config):
    selected=set(config['gpus'])
    occupied=[pid for pid in compute_pids(selected)
        if process_start(pid)!=config.get('allowed_processes',{}).get(str(pid))
        or str(pid) not in config.get('allowed_processes',{})]
    waiting=[]
    for path in (training/'queues').glob('*/status.json'):
        if path.parent==directory:
            continue
        state=read_json(path,{})
        other=read_json(path.parent/'config.json',{})
        if not selected & set(other.get('gpus',[])):
            continue
        if state.get('status') in ('training','submitted'):
            waiting.append(path.parent.name)
        elif state.get('status') in WAITING_QUEUE_STATES:
            # Submission time is stable across restarts and independent of naming.
            current=read_json(directory/'status.json',{})
            if (state.get('created_at',''),path.parent.name)<(current.get('created_at',''),directory.name):
                waiting.append(path.parent.name)
    for path in training.glob('*/config/web_launch.json'):
        if path.parents[1]==Path(config['output']):
            continue
        plan=read_json(path,{})
        if plan.get('enqueue'):
            continue
        reserved=set(plan.get('gpus',[]))
        if plan.get('validation_enabled',True) and plan.get('validation_gpu') is not None:
            reserved.add(plan['validation_gpu'])
        root=path.parents[1]
        state=read_json(root/'status.json',{})
        process=read_json(root/'config/web_process.json',{})
        if selected & reserved and state.get('status') not in ('completed','failed','cancelled') and process.get('pid') and process_start(process['pid']) is not None:
            waiting.append(root.name)
    return occupied,waiting


def run(config_path):
    directory=config_path.parent
    config=read_json(config_path)
    training=directory.parent.parent
    output=Path(config['output'])
    while not (directory/'process.json').exists():
        time.sleep(.1)
    while True:
        with (training/'.gpu_queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=read_json(directory/'status.json')
            if state['status']=='cancelled':
                return
            occupied,waiting=blockers(training,directory,config)
            if not occupied and not waiting:
                state.update(status='training',pid=os.getpid())
                atomic_json(directory/'status.json',state)
                process=read_json(output/'config/web_process.json')
                atomic_json(output/'config/web_process.json',dict(process,status='submitted'))
                atomic_json(output/config['arm']/'status.json',dict(status='submitted',step=0))
                child=subprocess.Popen(config['argv'],stdin=subprocess.DEVNULL)
                break
            state.update(status='waiting_for_gpus' if occupied else 'queued',
                unexpected_pids=occupied,waiting_for_runs=waiting,pid=os.getpid())
            atomic_json(directory/'status.json',state)
        time.sleep(5)
    code=child.wait()
    state.update(status='completed' if code==0 else 'failed',returncode=code)
    atomic_json(directory/'status.json',state)
    if code:
        atomic_json(output/'status.json',dict(status='failed',error=f'Training launcher exited {code}'))
        raise SystemExit(code)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.config)
    except Exception as error:
        state=read_json(args.config.parent/'status.json',{})
        atomic_json(args.config.parent/'status.json',dict(state,status='failed',error=repr(error)))
        config=read_json(args.config)
        atomic_json(Path(config['output'])/'status.json',dict(status='failed',error=repr(error)))
        raise
