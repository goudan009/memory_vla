"""Separate decoded target discontinuity from measured joint tracking error."""
import argparse
import json
from pathlib import Path

import numpy as np


def distribution(x):
    x=np.asarray(x).reshape(-1)
    return dict(n=int(x.size),mean=float(x.mean()),p50=float(np.median(x)),p95=float(np.quantile(x,.95)),
                p99=float(np.quantile(x,.99)),max=float(x.max())) if x.size else None


def analyze(root,mapping):
    channels=json.loads(mapping.read_text())['action_channels']
    groups={k:[r for r in channels if r['group']==k] for k in ('base','torso','left_arm','right_arm')}
    result=dict(path=str(root),groups={k:dict(interior=[],boundary=[],overlap=[],tracking=[],limits=0,elements=0,
                                            physical_max=0.,max_boundary_event=None,
                                            anchor_changes=[],residual_changes=[],
                                            channels=[dict(channel=i,joint=c.get('physical_joints'),
                                                limits=c['command_range'],min=float('inf'),max=float('-inf'),
                                                violations=0,max_overshoot=0.,first_violation=None)
                                                for i,c in enumerate(groups[k])]) for k in groups})
    total=0
    for path in sorted((root/'stages').glob('actions_task40_gpu*_env*.jsonl')):
        suffix=path.name.removeprefix('actions_task40_')
        previous={}
        for line in path.read_text().splitlines():
            r=json.loads(line)
            key=(r['instance_id'],r['seed'],r['episode_number'])
            h=r['action_end']-r['action_start']
            if h<=0:continue
            total+=1
            for group in groups:
                values=np.asarray(r['action']['action.'+group],dtype=float)
                if values.ndim==3:values=values[0]
                current=values[:h]
                out=result['groups'][group]
                out['interior'].extend(np.abs(np.diff(current,axis=0)).reshape(-1).tolist())
                out['elements']+=current.size
                out['physical_max']=max(out['physical_max'],float(np.abs(current).max()))
                limits=np.array([c['command_range'] for c in groups[group]])
                out['limits']+=int(((current<limits[:,0]-1e-5)|(current>limits[:,1]+1e-5)).sum())
                for idx,column in enumerate(current.T):
                    row=out['channels'][idx]
                    row['min']=min(row['min'],float(column.min()))
                    row['max']=max(row['max'],float(column.max()))
                    excess=np.maximum(limits[idx,0]-column,column-limits[idx,1])
                    bad=np.flatnonzero(excess>1e-5)
                    row['violations']+=int(bad.size)
                    row['max_overshoot']=max(row['max_overshoot'],float(excess.max()),0.)
                    if bad.size and row['first_violation'] is None:
                        offset=int(bad[0])
                        row['first_violation']=dict(trace=path.name,episode=key,
                            action_step=r['action_start']+offset,target=float(column[offset]))
                if key in previous and r['action_start']==previous[key]['action_end']:
                    prev=previous[key]
                    old=np.asarray(prev['action']['action.'+group])
                    if old.ndim==3:old=old[0]
                    last=prev['action_end']-prev['action_start']
                    jump=np.abs(values[0]-old[last-1])
                    out['boundary'].extend(jump.tolist())
                    out['overlap'].extend(np.abs(values[0]-old[last]).tolist())
                    event=out['max_boundary_event']
                    if event is None or jump.max()>event['jump']:
                        idx=int(jump.argmax())
                        out['max_boundary_event']=dict(trace=path.name,episode=key,action_start=r['action_start'],
                                                      channel=idx,jump=float(jump[idx]),previous=float(old[last-1,idx]),target=float(values[0,idx]))
            previous[key]=r
        # Older runs sample measured state only every80 physical steps.
        truth_path=path.with_name('task40_'+suffix)
        actions=[json.loads(l) for l in path.read_text().splitlines()]
        lookup={(r['instance_id'],r['seed'],r['episode_number'],r['action_end']):r for r in actions}
        feedback=path.with_name('feedback_'+suffix.removeprefix('task40_'))
        measured=[]
        if feedback.exists():
            last_anchor={}
            for l in feedback.read_text().splitlines():
                r=json.loads(l); measured.append((r,r['action_end'],r['after']))
                key=(r['instance_id'],r['seed'],r['episode_number'])
                command=lookup.get((*key,r['action_end']))
                if command is None:continue
                before=np.asarray(next(iter(r['before']['robots'].values()))['joint_positions'])
                for group in ('torso','left_arm','right_arm'):
                    indices=[c['simulation_qpos_indices'][0] for c in groups[group]]
                    q=before[indices]
                    a=np.asarray(command['action']['action.'+group]).reshape(-1,len(indices))
                    if r['action_start']==last_anchor.get(key,{}).get('end'):
                        prev=last_anchor[key][group]
                        result['groups'][group]['anchor_changes'].extend(np.abs(q-prev[0]).tolist())
                        result['groups'][group]['residual_changes'].extend(np.abs((a[0]-q)-prev[1]).tolist())
                    last_anchor.setdefault(key,{})[group]=(q,a[r['action_end']-r['action_start']-1]-q)
                last_anchor[key]['end']=r['action_end']
        elif truth_path.exists():
            for l in truth_path.read_text().splitlines():
                r=json.loads(l);measured.append((r,r['action_steps'],r))
        for identity,step,state in measured:
            r=lookup.get((identity['instance_id'],identity['seed'],identity['episode_number'],step))
            if r is None:continue
            robot=next(iter(state['robots'].values()))
            q=np.asarray(robot['joint_positions'])
            for group in ('torso','left_arm','right_arm'):
                action=np.asarray(r['action']['action.'+group])
                if action.ndim==3:action=action[0]
                target=action[r['action_end']-r['action_start']-1]
                indices=[c['simulation_qpos_indices'][0] for c in groups[group]]
                result['groups'][group]['tracking'].extend(np.abs(q[indices]-target).tolist())
    result['chunks']=total
    for v in result['groups'].values():
        for key in ('interior','boundary','overlap','tracking','anchor_changes','residual_changes'):v[key]=distribution(v[key])
        if v['boundary'] and v['interior']:
            v['boundary_to_interior_mean_ratio']=v['boundary']['mean']/max(v['interior']['mean'],1e-12)
            inside=v['interior']['mean']*v['interior']['n']
            boundary=v['boundary']['mean']*v['boundary']['n']
            v['all_executed_transition_abs_delta_mean']=(inside+boundary)/(v['interior']['n']+v['boundary']['n'])
            v['boundary_fraction_of_total_variation']=boundary/max(inside+boundary,1e-12)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mapping',type=Path,default=Path('v2_1_preflight/state_action_mapping.json'))
    args=p.parse_args();result=analyze(args.root,args.mapping)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
