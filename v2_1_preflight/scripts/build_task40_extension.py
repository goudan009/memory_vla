"""Reproducible 200-demo extension, CPU only; frozen original labels untouched."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from label_api import Label, negate, conjunction, disjunction, ever, inside_candidate, grasp_candidate

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'v2_1_preflight';OUT=BASE/'label_extension'
RAW=Path('/nvme3/datasets/b1k2025_raw_task40_v2_audit/task-0040')


def write(name,obj):
    (OUT/name).parent.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def eef_world(state,begin):
    base=Rotation.from_euler('xyz',np.arctan2(state[:,6:9],state[:,3:6]))
    return np.c_[base.apply(state[:,begin:begin+3])+state[:,:3],
        (base*Rotation.from_quat(state[:,begin+3:begin+7])).as_quat()]


def statistics(eid,name,label):
    v,q=label.value,label.quality;k=q>0
    index=np.flatnonzero(k)
    return dict(episode_index=eid,state=name,frames=len(v),positive=int(np.sum(v&k)),negative=int(np.sum(~v&k)),unknown=int(np.sum(~k)),
        certified_positive=int(np.sum(v&(q==1))),certified_negative=int(np.sum(~v&(q==1))),
        weak_positive=int(np.sum(v&(q==2))),weak_negative=int(np.sum(~v&(q==2))),
        adjacent_valid_switches=int(np.sum(k[1:]&k[:-1]&(v[1:]!=v[:-1]))),
        minimum_changes_between_known_observations=int(np.sum(v[index[1:]]!=v[index[:-1]])))


def changes(label):
    idx=np.flatnonzero(label.quality>0);result=[]
    for a,b in zip(idx[:-1],idx[1:]):
        if label.value[a]!=label.value[b]:
            result.append(dict(last_opposite_frame=int(a),first_new_frame=int(b),value=bool(label.value[b]),
                quality=int(label.quality[b]),exact_adjacent_boundary=bool(b==a+1)))
    return result


def main():
    config=json.loads((OUT/'task40_config.json').read_text())
    original=pd.read_parquet(BASE/'task40_labels.parquet')
    geo=np.load(OUT/'predicate_geometry.npz')
    files=sorted((ROOT/'assets/datasets/b1k2025_groot_task40_v1_200').glob('data/**/*.parquet'))
    writer=None;stats=[];episodes=[];state_names=None
    for count,path in enumerate(files):
        eid=int(path.stem.split('_')[-1]);old=original[original.episode_index==eid].reset_index(drop=True)
        state=np.stack(pd.read_parquet(path,columns=['observation.state'])['observation.state'])
        T=len(state);assert T==len(old)
        np.testing.assert_array_equal(old.frame_index,np.arange(T))
        with h5py.File(RAW/(path.stem+'.hdf5')) as f:scene=json.loads(f['data'].attrs['scene_file'])
        roles={role:scene['metadata']['task']['inst_to_name'][instance] for role,instance in config['roles'].items()}
        asset_args={role:scene['objects_info']['init_info'][roles[role]]['args'] for role in ('bag','microwave')}
        for role in asset_args:assert asset_args[role]['model']==config['expected_models'][role]
        with np.load(BASE/'samples/decoded'/(path.stem+'.npz')) as z:
            bag=z['bag_pose'];micro=z['microwave_pose']
            inside,center,local,margin=inside_candidate(bag,micro,geo['bag_collision_hull'],geo['microwave_container_hull'],
                asset_args['bag']['scale'],asset_args['microwave']['scale'],config['inside_boundary_unknown_m'])
            labels={name:Label(old[name].fillna(False).to_numpy(bool),np.where(old[name+'_valid'],1,0))
                for name in ('door_open','started','cooked_popcorn_exists')}
            labels['raw_popcorn_exists']=Label(z['raw_particle_count']>0,1)
            numeric=dict(bag_world_position_m=bag[:,:3],bag_world_quaternion_xyzw=bag[:,3:7],
                bag_aabb_center_world_m=center,bag_aabb_center_microwave_unscaled_local_m=local)
        labels['bag_inside']=inside
        features={}
        for side in ('left','right'):
            begin,end=config['eef_indices_state82'][side];fb,fe=config['finger_indices_state82'][side]
            eef=eef_world(state,begin)
            grasp,feature=grasp_candidate(bag,eef,state[:,fb:fe],config['grasp'])
            labels['bag_grasped_'+side]=grasp;features[side]=feature
            numeric[side+'_eef_world_position_m']=eef[:,:3];numeric[side+'_eef_world_quaternion_xyzw']=eef[:,3:7]
        labels['bag_grasped']=disjunction(labels['bag_grasped_left'],labels['bag_grasped_right'])
        labels['door_closed']=negate(labels['door_open'])
        for source,target in [('door_open','door_ever_opened'),('bag_inside','bag_ever_inside'),
                ('bag_grasped','bag_ever_grasped'),('started','microwave_ever_started'),('cooked_popcorn_exists','cooked_popcorn_ever_observed')]:
            labels[target]=ever(labels[source])
        labels['door_reclosed_after_open']=conjunction(labels['door_ever_opened'],labels['door_closed'])
        labels['door_ever_reclosed_after_open']=ever(labels['door_reclosed_after_open'])
        labels['bag_inside_with_door_closed']=conjunction(inside,labels['door_closed'])
        labels['bag_released_inside_candidate']=conjunction(inside,negate(labels['bag_grasped']))
        labels['microwave_on_with_closed_door']=conjunction(labels['started'],labels['door_closed'])
        labels['microwave_on_with_open_door']=conjunction(labels['started'],labels['door_open'])
        labels['bag_inside_while_on']=conjunction(inside,labels['started'])
        columns={k:old[k] for k in old.columns if k not in labels and not any(k==n+'_valid' for n in labels)}
        for name,label in labels.items():columns.update(label.columns(name));stats.append(statistics(eid,name,label))
        columns['inside_geometry_margin_m']=margin.astype(np.float32)
        for side,feature in features.items():
            for name,values in feature.items():
                columns[f'grasp_{side}_{name}']=values.astype(np.float32) if values.dtype.kind=='f' else values
        # Full poses already exist in original observations and decoded inputs.
        # Keep trajectory arrays in selected NPZ samples instead of duplicating
        # forty continuous coordinates across the primary label table.
        frame=pd.DataFrame(columns)
        for name,label in labels.items():
            assert np.array_equal(frame[name].notna(),label.quality>0)
            assert np.array_equal(frame[name+'_certified_valid'],label.quality==1)
        assert not frame.bag_inside_certified_valid.any() and not frame.bag_grasped_certified_valid.any()
        if state_names is None:state_names=list(labels)
        assert state_names==list(labels)
        table=pa.Table.from_pandas(frame,preserve_index=False)
        if writer is None:writer=pq.ParquetWriter(OUT/'task40_labels.parquet',table.schema,compression='zstd')
        writer.write_table(table)
        event_names=['door_open','bag_inside','bag_grasped','started','cooked_popcorn_exists']
        events={n:changes(labels[n]) for n in event_names}
        episodes.append(dict(episode=eid,frames=T,roles=roles,asset_args=asset_args,events=events,
            final={n:dict(value=bool(v.value[-1]) if v.quality[-1] else None,quality=int(v.quality[-1])) for n,v in labels.items()},
            numerical_ranges=dict(inside_margin_m=[float(margin.min()),float(margin.max())],
                bag_eef_distance_m={s:[float(f['distance_m'].min()),float(f['distance_m'].max())] for s,f in features.items()})))
        if eid in (400010,400020,403000):
            np.savez_compressed(OUT/f'samples/episode_{eid:08d}_trajectories.npz',**numeric,
                **{f'{s}_{k}':v for s,f in features.items() for k,v in f.items()},inside_margin_m=margin,
                bag_scale=np.array(asset_args['bag']['scale']),microwave_scale=np.array(asset_args['microwave']['scale']))
        if count%20==0:print(json.dumps(dict(complete=count+1,episode=eid,inside=statistics(eid,'bag_inside',inside),grasp=statistics(eid,'bag_grasped',labels['bag_grasped']))),flush=True)
    writer.close()
    stat=pd.DataFrame(stats);total=stat.groupby('state',sort=False).sum(numeric_only=True).reset_index();total.episode_index='ALL_200'
    pd.concat([stat,total],ignore_index=True).to_csv(OUT/'label_statistics.csv',index=False)
    write('episode_evidence.json',episodes)
    write('summary.json',dict(episodes=len(episodes),frames=sum(e['frames'] for e in episodes),states=state_names,
        quality=config['quality'],statistics=total.to_dict(orient='records'),
        source_frozen_labels_sha256=hashlib.sha256((BASE/'task40_labels.parquet').read_bytes()).hexdigest(),
        timing='Observation t / recorded state t, before action t; 30Hz; extra terminal state T excluded',
        retrospective_mask_policy='Existing started +/-8 boundary unknown mask retained; new history values never use future values',
        asset_identity_certified=False,grasp_runtime_state_serialized=False,simulation_replay_used=False))
    print(total.to_string(index=False))


if __name__=='__main__':
    (OUT/'samples').mkdir(exist_ok=True,parents=True)
    main()
