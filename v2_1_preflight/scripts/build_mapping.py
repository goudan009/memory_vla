"""Build dimension-by-dimension processor/physical mapping from audited sources."""
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'v2_1_preflight'


def main():
    topology=json.loads((OUT/'joint_topology.json').read_text());joints={j['name']:j for j in topology['joints']}
    modality=json.loads((ROOT/'assets/datasets/b1k2025_groot_task40_v1_200/meta/modality.json').read_text())
    data=json.loads((OUT/'samples/dataset_audit.json').read_text())
    cfg=json.loads((ROOT/'checkpoints/groot_n1d6_base/processor_config.json').read_text())['processor_kwargs']
    stats=json.loads((ROOT/'checkpoints/groot_n1d6_base/statistics.json').read_text())['behavior_r1_pro']
    spec=importlib.util.spec_from_file_location('convert',ROOT/'src/v1_groot_memoryvla/data/convert_b1k.py')
    convert=importlib.util.module_from_spec(spec);spec.loader.exec_module(convert)
    qnames=['base_footprint_'+c+'_joint' for c in ('x','y','z','rx','ry','rz')]+['torso_joint'+str(i) for i in range(1,5)]+[f'{side}_arm_joint{i}' for i in range(1,8) for side in ('left','right')]+[f'{side}_gripper_finger_joint{i}' for side in ('left','right') for i in (1,2)]
    qindex={name:i for i,name in enumerate(qnames)}
    def meaning(k,i):
        if k.startswith('robot_pos'):return 'world position of moving base_link', 'm','world','xyz'[i]
        if k=='robot_2d_ori':return 'yaw about world Z','rad','world','yaw'
        if k.startswith('robot_ori'):return 'cos/sin of Euler XYZ orientation of moving base_link','1','world','xyz'[i]
        if k.startswith('robot_2d_ori_'):return 'cos/sin yaw','1','world','yaw'
        if k=='robot_lin_vel':return 'base_link linear velocity','m/s','world','xyz'[i]
        if k=='robot_ang_vel':return 'base_link angular velocity','rad/s','world','xyz'[i]
        if k.startswith('eef_'):return ('end effector position','m','base_link','xyz'[i]) if k.endswith('pos') else ('end effector quaternion XYZW','1','base_link','xyzw'[i])
        if 'gripper' in k:return 'finger prismatic position','m','joint',str(i+1)
        if 'sin' in k or 'cos' in k:return 'joint angle trigonometric embedding','1','joint',str(i+1)
        return 'physical revolute joint angle','rad','joint',str(i+1)
    fields=[];dims=[]
    for k,sl in modality['state'].items():
        b,e=sl['start'],sl['end'];rb,re=convert.STATE_SLICES[k]
        for i in range(e-b):
            text,unit,frame,component=meaning(k,i)
            physical=None
            if k.startswith('arm_'):physical=f'{k.split("_")[1]}_arm_joint{i+1}'
            if k.startswith('gripper_'):physical=f'{k.split("_")[1]}_gripper_finger_joint{i+1}'
            if k=='trunk_qpos':physical=f'torso_joint{i+1}'
            dims.append(dict(index=b+i,field=k,component=component,meaning=text,unit=unit,reference_frame=frame,
                raw256_index=rb+i,joint=physical,simulation_qpos_index=qindex.get(physical),
                dtype='float32',example=data['episodes'][0]['episode'],
                example_value=json.loads((OUT/'samples/raw_state_action_samples.json').read_text())[0]['state82'][b+i],
                observed_min=data['ranges']['state82']['min'][b+i],observed_max=data['ranges']['state82']['max'][b+i]))
        fields.append(dict(name=k,packed82_slice=[b,e],raw256_slice=[rb,re],shape=[e-b],normalization=stats['state'][k]))
    definitions=[('base',c,[f'base_footprint_{axis}_joint'],None,[None]) for c,axis in enumerate(('x','y','rz'))]
    definitions += [('torso',3+i,[f'torso_joint{i+1}'],'trunk_qpos',[78+i]) for i in range(4)]
    definitions += [('left_arm',7+i,[f'left_arm_joint{i+1}'],'arm_left_qpos',[18+i]) for i in range(7)]
    definitions += [('left_gripper',14,[f'left_gripper_finger_joint{i}' for i in (1,2)],'gripper_left_qpos',[46,47])]
    definitions += [('right_arm',15+i,[f'right_arm_joint{i+1}'],'arm_right_qpos',[48+i]) for i in range(7)]
    definitions += [('right_gripper',22,[f'right_gripper_finger_joint{i}' for i in (1,2)],'gripper_right_qpos',[76,77])]
    actions=[]
    for group,idx,physical,state_key,state_indices in definitions:
        start=modality['action'][group]['start'];offset=idx-start
        relative=group in ('torso','left_arm','right_arm')
        source='relative_action' if relative else 'action'
        def stat_col(values):
            v=np.asarray(values);return v[...,offset].tolist()
        isbase=group=='base';isgrip='gripper' in group
        j=joints[physical[0]]
        actions.append(dict(action_index=idx,processor_padded_column=idx,group=group,within_group_index=offset,
            physical_joints=physical,simulation_qpos_indices=[qindex[n] for n in physical],
            packed82_qpos_indices=[x for x in state_indices if x is not None],
            packed82_other_related_indices=(list(range(0,18)) if isbase else []),
            raw256_qpos_indices=[qindex[n] for n in physical],raw256_velocity_indices=[84+qindex[n] for n in physical],
            packed82_correspondence=('No base virtual joint positions in82; world pose0:12 and world velocity12:18 require transforms, not direct q match' if isbase else 'direct physical qpos'),
            physical_unit='dimensionless input -> m/s,m/s,rad/s' if isbase else 'dimensionless input -> m' if isgrip else 'rad',
            sim_control='local base_link velocity, scaled and rotated into canonical joint frame' if isbase else 'absolute finger position; software equal command to both fingers' if isgrip else 'absolute joint position; use_delta_commands=false',
            command_range=[-1,1] if isbase or isgrip else j['limits_SI'],
            physical_range=[[-.75,.75],[-.75,.75],[-1,1]][idx] if isbase else [0,.05] if isgrip else j['limits_SI'],
            sim_input_to_physical=f'v={ [.75,.75,1][idx]}*clip(a,-1,1)' if isbase else 'q_finger=(clip(a,-1,1)+1)*0.025m, both fingers' if isgrip else 'q_target=a rad (joint limit clipping by controller)',
            axis=j['axis'],axis_reference='joint local0 frame; rotate by quat0_xyzw to get parent frame axis',
            parent_links=[joints[n]['parent'] for n in physical],child_links=[joints[n]['child'] for n in physical],
            fk_computable=True,processor_representation='RELATIVE to state at observation t, all future offsets share t reference' if relative else 'ABSOLUTE',
            normalization_source=f'checkpoints/groot_n1d6_base/statistics.json:behavior_r1_pro/{source}/{group}',
            normalization={name:stat_col(v) for name,v in stats[source][group].items()},
            normalization_bounds_are_per_horizon=relative,
            observed_min=data['ranges']['action23']['min'][idx],observed_max=data['ranges']['action23']['max'][idx]))
    # Channel graph: follow physical parent chain until nearest controlled ancestor.
    # Three independent planar freedoms form one contracted mobile-body group.
    owner={n:a['action_index'] for a in actions for n in a['physical_joints']};by_child={j['child']:j for j in joints.values()}
    edges={tuple(sorted((0,1))),tuple(sorted((1,2))),tuple(sorted((0,2)))}
    evidence=[dict(edge=list(e),basis='three controlled generalized planar velocities of same mobile base; same-body grouping, not three serial physical links') for e in sorted(edges)]
    for a in actions[3:]:
        candidates=set()
        for name in a['physical_joints']:
            cur=by_child.get(joints[name]['parent'])
            while cur:
                if cur['name'] in owner:
                    candidates.add(owner[cur['name']]);break
                cur=by_child.get(cur['parent'])
        for ancestor in candidates:
            parents=(0,1,2) if ancestor in (0,1,2) else (ancestor,)
            for parent in parents:
                edge=tuple(sorted((parent,a['action_index'])))
                if edge not in edges:
                    edges.add(edge);evidence.append(dict(edge=list(edge),basis='nearest controlled ancestor along actual USD tree; fixed and uncontrolled virtual joints contracted'))
    result=dict(index_convention='zero-based, half-open slices',state_dim=82,raw_state_dim=256,action_dim=23,
        simulation_dof=28,state_fields=fields,state_dimensions=dims,action_channels=actions,
        simulation_qpos_order=[dict(index=i,name=n,packed82_indices=[d['index'] for d in dims if d['joint']==n],
            role='passive virtual coordinate' if i in (2,3,4) else 'controlled virtual coordinate' if i<6 else 'controlled physical DOF') for i,n in enumerate(qnames)],
        processor=dict(config_source='checkpoints/groot_n1d6_base/processor_config.json',max_state_dim=128,max_action_dim=128,max_action_horizon=50,
            actual_state_shape=['B',1,128],actual_action_shape=['B',50,128],valid_action_region='[:,0:32,0:23], additionally masked at episode tail',
            state_padding='82:128 zero',action_padding='cols23:128 and time32:50 zero; action_mask zero',
            use_percentiles=cfg['use_percentiles'],clip_outliers=cfg['clip_outliers'],
            normalize='z=clip(2*(x-min)/(max-min)-1,-1,1); isclose(min,max) -> z=0',
            inverse='x=(clip(z,-1,1)+1)/2*(max-min)+min; relative groups then add current observation q_t',
            caution='relative_action statistics are32×group_dim, horizon-specific; do not collapse to one bound'),
        topology_prior=dict(status='candidate only; network feature choice not made',nodes=list(range(23)),edges=[list(e) for e in sorted(edges)],evidence=evidence,
            contraction='gripper two finger joints contracted to one command node; fixed/passive joints traversed, wheel joints fixed in USD'))
    (OUT/'state_action_mapping.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    target=OUT/'samples/processor';target.mkdir(parents=True,exist_ok=True)
    for name in ('processor_config.json','statistics.json','config.json','embodiment_id.json','download_provenance.json'):
        if (ROOT/'checkpoints/groot_n1d6_base'/name).exists():shutil.copy2(ROOT/'checkpoints/groot_n1d6_base'/name,target/name)
    print(json.dumps(dict(state_dimensions=len(dims),action_channels=len(actions),candidate_edges=len(edges))))


if __name__=='__main__':main()
