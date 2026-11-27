"""Align extreme decoded elbow/base commands to measured joints and video frames."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def collect(root, output, mapping):
    events = {}
    timeline=[]
    channels=json.loads(mapping.read_text())['action_channels']
    for path in sorted((root/'stages').glob('actions_task40_gpu*_env*.jsonl')):
        suffix = path.stem.removeprefix('actions_task40_')
        truth = rows(path.with_name('task40_'+suffix+'.jsonl'))
        video = rows(path.with_name('video_'+suffix+'.jsonl'))
        commands=rows(path)
        lookup={(r['instance_id'],r['seed'],r['episode_number'],r['action_end']):r for r in commands}
        for r in truth:
            if r['instance_id']!=0 or r['seed']!=0 or r['action_steps']>800:continue
            command=lookup.get((r['instance_id'],r['seed'],r['episode_number'],r['action_steps']))
            if command is None:continue
            q=next(iter(r['robots'].values()))['joint_positions']
            target=np.asarray(command['action']['action.left_arm']).reshape(-1,7)
            timeline.append(dict(action_step=r['action_steps'],left_elbow_measured_rad=q[16],
                                 left_elbow_last_requested_rad=float(target[command['action_end']-command['action_start']-1,3]),
                                 trace=str(path)))
        for action in commands:
            n = action['action_end']-action['action_start']
            if n <= 0:
                continue
            for side, sim_index in [('left',16),('right',17)]:
                channel=next(c for c in channels if c['group']==side+'_arm' and c['within_group_index']==3)
                sim_index=channel['simulation_qpos_indices'][0]
                upper=channel['physical_range'][1]
                a = np.asarray(action['action']['action.'+side+'_arm']).reshape(-1,7)[:n,3]
                index = int(a.argmax())
                value = float(a[index])
                if side in events and events[side]['decoded_target_rad'] >= value:
                    continue
                key = tuple(action[k] for k in ('instance_id','seed','episode_number'))
                step = action['action_start']+index
                same = lambda row: tuple(row[k] for k in ('instance_id','seed','episode_number')) == key
                measured = [r for r in truth if same(r)]
                frames = [r for r in video if same(r)]
                nearest = min(measured,key=lambda r:abs(r['action_steps']-step)) if measured else None
                frame = min(frames,key=lambda r:abs(r['action_steps']-step)) if frames else None
                event = dict(trace=str(path),instance_id=key[0],seed=key[1],episode_number=key[2],
                    physical_joint=side+'_arm_joint4',simulation_index=sim_index,action_step=step,
                    decoded_target_rad=value,asset_upper_limit_rad=upper,
                    overshoot_rad=value-upper,measured=None,video=None)
                if nearest:
                    robot = next(iter(nearest['robots'].values()))
                    event['measured'] = dict(action_step=nearest['action_steps'],
                        offset_steps=nearest['action_steps']-step,
                        qpos_rad=robot['joint_positions'][sim_index])
                if frame:
                    clips = list((root/'videos').rglob(f'*_instance{key[0]}_seed{key[1]}_{suffix}_s*.mp4'))
                    if len(clips) == 1:
                        event['video'] = dict(path=str(clips[0]),frame_index=frame['frame_index'],
                            action_step=frame['action_steps'],offset_steps=frame['action_steps']-step,
                            fps=frame['video_fps'],recording_id=frame.get('recording_id'))
                events[side] = event
    for side,event in events.items():
        frame = event['video']
        if frame:
            image = output/(side+'_elbow_extreme.png')
            subprocess.run(['ffmpeg','-v','error','-i',frame['path'],'-vf',
                f"select=eq(n\\,{frame['frame_index']})",'-frames:v','1','-y',str(image)],check=True)
            frame['image'] = str(image)
    events['instance0_seed0_early_elbow_timeline']=timeline
    return events


def input_alignment(recording, root, mapping):
    z=np.load(recording)
    channels=json.loads(mapping.read_text())['action_channels']
    matches=[]
    for path in sorted((root/'stages').glob('actions_task40_gpu*_env*.jsonl')):
        commands=rows(path)
        if not commands:continue
        action=commands[0]
        a=np.asarray(action['action']['action.left_arm']).reshape(-1,7)
        for batch in range(z['decoded.left_arm'].shape[0]):
            if not np.allclose(a,z['decoded.left_arm'][batch],atol=1e-8,rtol=0):continue
            suffix=path.name.removeprefix('actions_task40_')
            feedback=rows(path.with_name('feedback_'+suffix))
            if not feedback:continue
            robot=next(iter(feedback[0]['before']['robots'].values()))
            q=np.asarray(robot['joint_positions'])
            result=dict(trace=str(path),input_batch=batch,groups={})
            for group,key in [('left_arm','arm_left_qpos'),('right_arm','arm_right_qpos'),('torso','trunk_qpos')]:
                indices=[c['simulation_qpos_indices'][0] for c in channels if c['group']==group]
                given=z['state.'+key][batch,0]
                actual=q[indices]
                result['groups'][group]=dict(max_abs_error=float(np.max(np.abs(given-actual))),
                    policy_input=given.tolist(),simulation_qpos=actual.tolist())
            matches.append(result)
    return dict(input=str(recording),matches=matches)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mapping',type=Path,default=Path('v2_1_preflight/state_action_mapping.json'))
    p.add_argument('--policy-input',type=Path)
    p.add_argument('--policy-input-run',type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result = collect(args.root,args.output,args.mapping)
    if args.policy_input:
        result['online_input_alignment']=input_alignment(args.policy_input,args.policy_input_run or args.root,args.mapping)
    (args.output/'events.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
