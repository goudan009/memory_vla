"""Resume with the saved training code, including older no-validation runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def run(config):
    spec=read(config);root=Path(spec['output']);arm=spec['arm']
    env=dict(os.environ,PYTHONPATH=str(Path(spec['source'])/'src'),PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        NO_ALBUMENTATIONS_UPDATE='1',OMP_NUM_THREADS='4')
    # Older public launchers reject no-validation resumes; their saved DDP
    # workers already restore the model, AdamW, rank cursors and memory correctly.
    argv=spec['argv'] if spec['validation_enabled'] else spec['worker_argv']
    if not spec['validation_enabled']:
        env['CUDA_VISIBLE_DEVICES']=','.join(map(str,spec['gpus']))
    child=None
    def record_restored():
        write(root/arm/f"config/resume-step-{spec['resume_step']:06d}.json",dict(
            checkpoint=dict(step=spec['resume_step'],path=spec['checkpoint']),
            optimizer_restored=True,rank_memory_and_cursor_restored=True,
            total_steps=spec['steps'],source=spec['source']))
    try:
        with (root/f'logs/{arm}_torchrun.log').open('ab') as log:
            child=subprocess.Popen(argv,cwd=spec['source'],env=env,stdin=subprocess.DEVNULL,
                stdout=None if spec['validation_enabled'] else log,stderr=subprocess.STDOUT)
            write(root/'status.json',dict(status='training',pid=os.getpid(),
                processes={'resume_child':child.pid},resume_step=spec['resume_step'],max_steps=spec['steps']))
            recorded=False
            while child.poll() is None:
                state=read(root/arm/'status.json')
                if not recorded and state.get('step',0)>spec['resume_step']:
                    # A completed DDP update proves every rank restored successfully.
                    record_restored();recorded=True
                time.sleep(1)
        if child.returncode:
            raise RuntimeError(f'Resumed training exited {child.returncode}; see logs/{arm}_torchrun.log')
        if not spec['validation_enabled']:
            state=read(root/arm/'status.json')
            if not state.get('training_complete') or state.get('step',0)<=spec['resume_step']:
                raise RuntimeError('Worker exited without completing a resumed training update')
            if not recorded: record_restored()
            write(root/'status.json',dict(status='completed',steps_each=state['step']))
    except BaseException as error:
        write(root/'status.json',dict(status='failed',error=repr(error)))
        state=read(root/arm/'status.json')
        write(root/arm/'status.json',dict(state,status='failed',training_complete=False,error=repr(error)))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate();child.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config)
