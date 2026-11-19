"""Isolated MuJoCo process for real GR00T LIBERO evaluation."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'third_party/LIBERO'))
sys.path.append(str(ROOT/'src/v1_groot_memoryvla/.venv/lib/python3.10/site-packages'))
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')

from v1_groot_memoryvla.benchmarks.libero.protocol import (
    CONTRACT, HORIZONS, LIBERO_REVISION, atomic_json, camera_image, native_action, quat_axis_angle, read_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-port',type=int,required=True)
    parser.add_argument('--policy-token',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--suite',choices=HORIZONS,required=True)
    parser.add_argument('--tasks',type=int,nargs='+',required=True)
    parser.add_argument('--trials',type=int,default=1)
    parser.add_argument('--max-steps',type=int,default=0)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not 1 <= args.trials <= 50 or not 0 <= args.max_steps <= HORIZONS[args.suite]:
        parser.error('Trials must be 1–50 and max-steps cannot exceed the suite horizon')
    actual_revision = subprocess.check_output(['git','-C',str(ROOT/'third_party/LIBERO'),'rev-parse','HEAD'],text=True).strip()
    if actual_revision != LIBERO_REVISION: raise ValueError('LIBERO revision differs from the evaluation contract')
    import yaml
    config = ROOT/'outputs/benchmarks/libero_config'
    config.mkdir(parents=True,exist_ok=True)
    os.environ['LIBERO_CONFIG_PATH']=str(config)
    assets = ROOT/'third_party/LIBERO/libero/libero'
    (config/'config.yaml').write_text(yaml.safe_dump(dict(benchmark_root=str(assets),
        bddl_files=str(assets/'bddl_files'),init_states=str(assets/'init_files'),
        assets=str(assets/'assets'),datasets=str(ROOT/'assets/datasets/libero'))))
    import numpy as np
    import torch
    from PIL import Image
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()[args.suite]()
    metadata = read_json(args.checkpoint/'libero.json')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def call(endpoint,value):
        request = Request(f'http://127.0.0.1:{args.policy_port}/{endpoint}',data=json.dumps(value).encode(),
                          headers={'Content-Type':'application/json','Authorization':args.policy_token})
        with urlopen(request,timeout=90) as response: return json.load(response)
    rows=[]; started=time.monotonic()
    debug = 0 < args.max_steps < HORIZONS[args.suite]
    limit = args.max_steps or HORIZONS[args.suite]
    def publish():
        valid = [r for r in rows if r['valid']]
        result = dict(type='libero_eval',suite=args.suite, checkpoint=str(args.checkpoint.resolve()),
            arm=metadata['arm'], checkpoint_step=metadata['step'], contract=CONTRACT,
            seed=args.seed, mode='debug' if debug else 'full_horizon', horizon=limit, trials=args.trials,
            requested_episodes=len(args.tasks)*args.trials, completed_episodes=len(rows),
            valid_episodes=len(valid), successful_episodes=sum(r['success'] for r in valid),
            invalid_episodes=len(rows)-len(valid),
            success_rate=(sum(r['success'] for r in valid)/len(valid)) if valid and not debug else None,
            episodes=rows, elapsed_seconds=time.monotonic()-started, libero_commit=actual_revision,
            passed=len(rows)==len(args.tasks)*args.trials and len(valid)==len(rows))
        atomic_json(args.output,result)
        return result
    for task_id in args.tasks:
        task = suite.get_task(task_id)
        env = OffScreenRenderEnv(bddl_file_name=str(assets/'bddl_files'/task.problem_folder/task.bddl_file),
                                 camera_heights=256,camera_widths=256)
        initial = torch.load(assets/'init_files'/task.problem_folder/task.init_states_file,map_location='cpu',weights_only=False)
        try:
            for trial in range(args.trials):
                row=dict(task=task_id,trial=trial,language=task.language,initial_state_index=trial,
                         seed=args.seed+trial,steps=0,decisions=0,success=False,valid=False,frames={})
                writer=None
                try:
                    env.seed(row['seed']); env.reset(); obs=env.set_init_state(initial[trial])
                    call('reset',dict(seed=row['seed']))
                    for _ in range(10): obs,_,_,_=env.step([0,0,0,0,0,0,-1])
                    directory=args.output.parent/f'task-{task_id}-episode-{trial}'
                    directory.mkdir(exist_ok=True)
                    def capture(label):
                        row['frames'][label]={}
                        for camera in ('agentview','robot0_eye_in_hand'):
                            file=directory/f'{label}-{camera}.png'
                            Image.fromarray(camera_image(obs[camera+'_image'])).save(file)
                            row['frames'][label][camera]=str(file.relative_to(args.output.parent))
                    capture('start')
                    video=directory/'rollout.mp4'
                    writer=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo',
                        '-pix_fmt','rgb24','-s','512x256','-r','20','-i','-','-an','-c:v','libx264',
                        '-preset','veryfast','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
                    done=False
                    while row['steps'] < limit and not done:
                        images={name:base64.b64encode(camera_image(obs[key]).tobytes()).decode() for name,key in
                                [('image','agentview_image'),('wrist_image','robot0_eye_in_hand_image')]}
                        actions=call('act',dict(images=images,eef=np.r_[obs['robot0_eef_pos'],quat_axis_angle(obs['robot0_eef_quat'])].tolist(),
                            gripper=obs['robot0_gripper_qpos'].tolist(),language=task.language,episode=f'{task_id}/{trial}'))['actions']
                        row['decisions']+=1
                        for action in actions[:limit-row['steps']]:
                            frame=np.concatenate([camera_image(obs[k]) for k in ('agentview_image','robot0_eye_in_hand_image')],axis=1)
                            writer.stdin.write(frame.tobytes())
                            obs,_,done,_=env.step(native_action(action))
                            row['steps']+=1
                            if done: break
                        print(json.dumps(dict(task=task_id,trial=trial,environment_steps=row['steps'],decisions=row['decisions'])),flush=True)
                    row['success']=bool(env.check_success()); row['valid']=True
                    capture('end')
                except Exception as error:
                    row['error']=str(error)
                finally:
                    if writer:
                        writer.stdin.close()
                        if writer.wait()!=0: row.update(valid=False,error='Video encoder failed')
                        elif video.is_file(): row['video']=str(video.relative_to(args.output.parent))
                rows.append(row); publish(); print(json.dumps(dict(episode=row)),flush=True)
                if not row['valid']: raise RuntimeError(row['error'])
        finally: env.close()
    result=publish()
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)


if __name__=='__main__': main()
