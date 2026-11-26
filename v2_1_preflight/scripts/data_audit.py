"""Audit every demo, reuse content-verified recorded labels, preserve unknowns."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'v2_1_preflight'
DATA=ROOT/'assets/datasets/b1k2025_groot_task40_v1_200'
RAW=Path('/nvme3/datasets/b1k2025_raw_task40_v2_audit/task-0040')
HISTORY=Path('/nvme3/projects/memory_vla-v2-v3-backup-oEWaKh/Output/test_02/experiment_01/evidence')
sys.path.insert(0,str(ROOT/'evaluation/scripts'))
from v2_recording_alignment import matching_demo
spec=importlib.util.spec_from_file_location('convert',ROOT/'src/v1_groot_memoryvla/data/convert_b1k.py')
convert=importlib.util.module_from_spec(spec);spec.loader.exec_module(convert)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    (OUT/path).parent.mkdir(parents=True,exist_ok=True)
    (OUT/path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--decoded-report',type=Path)
    parser.add_argument('--bundled-started',action='store_true',help='Use bundled recorded values and archived certified boundary policy, not external historical NPZ files')
    args=parser.parse_args()
    cert=json.loads(((OUT/'samples/historical_started_certification.json') if args.bundled_started else (HISTORY/'certified_started_labels.json')).read_text())
    certified={x['episode']:x for x in cert['episodes']};assert len(certified)==200
    decoded=json.loads(args.decoded_report.read_text()) if args.decoded_report else None
    assert not decoded or decoded['passed']
    if decoded:write('samples/official_decode_report.json',decoded)
    out_labels=OUT/'task40_labels.parquet';writer=None;manifest=[];stats=[]
    ranges={k:[np.full(d,np.inf),np.full(d,-np.inf)] for k,d in [('state82',82),('raw_state256',256),('action23',23)]}
    all_samples=[]
    for ep_i,p in enumerate(sorted(DATA.glob('data/**/*.parquet'))):
        eid=int(p.stem.split('_')[-1]);entry=certified[eid]
        df=pd.read_parquet(p);s=np.stack(df['observation.state']);a=np.stack(df.action).astype(np.float32);T=len(df)
        assert s.shape==(T,82) and a.shape==(T,23) and np.isfinite(s).all() and np.isfinite(a).all()
        assert np.array_equal(df.frame_index,np.arange(T)) and np.max(np.abs(df.timestamp-np.arange(T)/30))<1e-4
        original=next((ROOT/'assets/datasets/b1k2025/data').glob(f'**/{p.name}'))
        rdf=pd.read_parquet(original,columns=['observation.state','action']);rs=np.stack(rdf['observation.state'])
        convert.validate_raw_state(rs)
        np.testing.assert_array_equal(s,np.concatenate([rs[:,b:e] for b,e in convert.STATE_SLICES.values()],axis=1))
        np.testing.assert_array_equal(a,np.stack(rdf.action))
        raw=RAW/(p.stem+'.hdf5');assert sha(raw)==entry['raw_sha256']
        with h5py.File(raw) as f:
            name,attempts=matching_demo(f['data'],a);g=f['data'][name]
            assert name==entry['selected_raw_group'] and g['state'].shape[0]==T+1
            if ep_i==0:
                scene=json.loads(f['data'].attrs['scene_file']);config=json.loads(f['data'].attrs['config'])
                write('samples/scene_initial_state.json',scene)
                write('samples/actual_recording_config.json',config)
            reward=g['reward'][:];terminated=g['terminated'][:];truncated=g['truncated'][:]
        archive=Path(entry['artifact'])
        if args.bundled_started:
            archive=OUT/'samples/decoded'/(p.stem+'.npz')
            with np.load(archive) as z:
                y=z['started'].astype(bool);explicit=z['explicitly_recorded'][:,0]
            transitions=(np.flatnonzero(y[1:]!=y[:-1])+1).tolist()
            assert transitions==entry['transition_frames']
            known=np.ones(T,dtype=bool);radius=cert['boundary_unknown_radius_frames']
            for t in transitions:known[max(0,t-radius):min(T,t+radius+1)]=False
            assert int((y&known).sum())==entry['positive_valid']
            assert int((~y&known).sum())==entry['negative_valid']
            assert int((~known).sum())==entry['unknown']
        else:
            assert sha(archive)==entry['artifact_sha256']
            with np.load(archive) as z:y=z['labels'][:,0].astype(bool);known=z['known'][:,0].astype(bool);explicit=z['object_explicitly_recorded']
        assert y.shape==(T,) and known.shape==(T,)
        labels=pd.DataFrame(dict(episode_index=np.full(T,eid,dtype=np.int64),frame_index=np.arange(T,dtype=np.int64),
            timestamp_s=df.timestamp.to_numpy(),raw_state_index=np.arange(T,dtype=np.int64),
            started=pd.arrays.BooleanArray(y,~known),started_valid=known,started_recorded_value=y,
            started_object_explicitly_recorded=explicit,raw_reward=reward,
            raw_terminated=terminated,raw_truncated=truncated))
        for label in ('door_open','bag_grasped','bag_inside'):
            labels[label]=pd.array([None]*T,dtype='boolean');labels[label+'_valid']=False
        if decoded:
            with np.load(OUT/'samples/decoded'/(p.stem+'.npz')) as z:
                assert np.array_equal(z['started'],y)
                joint=z['microwave_joint'];threshold=decoded['open_joints'][0]['threshold']
                assert not decoded['open_both_sides'] and len(decoded['open_joints'])==1
                direction=decoded['open_joints'][0]['direction'];door=joint>threshold if direction==1 else joint<threshold
                door_known=np.isfinite(joint)&(np.abs(joint-threshold)>1e-5)
                # This is a direct current-frame joint predicate, no delayed physics propagation.
                labels['door_open']=pd.arrays.BooleanArray(door,~door_known);labels['door_open_valid']=door_known
                labels['microwave_door_joint_rad']=joint
                labels['microwave_temperature_c']=z['temperature'][:,0];labels['bag_temperature_c']=z['temperature'][:,1]
                labels['microwave_state_explicitly_recorded']=z['explicitly_recorded'][:,0]
                labels['bag_state_explicitly_recorded']=z['explicitly_recorded'][:,1]
                if 'cooked_particle_count' in z:
                    labels['cooked_popcorn_particle_count']=z['cooked_particle_count']
                    labels['raw_popcorn_particle_count']=z['raw_particle_count']
                    labels['cooked_popcorn_exists']=pd.array(z['cooked_particle_count']>0,dtype='boolean')
                    labels['cooked_popcorn_exists_valid']=True
        table=pa.Table.from_pandas(labels,preserve_index=False)
        if writer is None:writer=pq.ParquetWriter(out_labels,table.schema,compression='zstd')
        writer.write_table(table)
        for label in ('door_open','bag_grasped','bag_inside','started')+ (('cooked_popcorn_exists',) if decoded else ()):
            values=labels[label];valid=labels[label+'_valid'].to_numpy();yv=values.fillna(False).to_numpy(dtype=bool)
            pair=valid[1:]&valid[:-1]
            nchanges=int(np.sum(pair&(yv[1:]!=yv[:-1])))
            recorded=int(np.sum(y[1:]!=y[:-1])) if label=='started' else (int(np.sum(door[1:]!=door[:-1])) if decoded and label=='door_open' else int(np.sum(yv[1:]!=yv[:-1])) if label=='cooked_popcorn_exists' else None)
            stats.append(dict(episode_index=eid,state=label,frames=T,positive=int(np.sum(yv&valid)),negative=int(np.sum(~yv&valid)),unknown=int(np.sum(~valid)),
                              adjacent_valid_switches=nchanges,recorded_switches_before_mask=recorded,
                              switch_count_status='known' if recorded is not None else 'unknown'))
        for k,arr in [('state82',s),('raw_state256',rs),('action23',a)]:
            ranges[k][0]=np.minimum(ranges[k][0],arr.min(0));ranges[k][1]=np.maximum(ranges[k][1],arr.max(0))
        manifest.append(dict(episode=eid,frames=T,parquet=str(p),parquet_sha256=sha(p),raw_hdf5=str(raw),raw_sha256=entry['raw_sha256'],
            selected_group=name,attempts=attempts,label_archive_sha256=sha(archive),label_archive=str(archive),
            label_source='bundled recorded values plus archived boundary policy' if args.bundled_started else 'original historical certified archive',
            timestamps_checked=True,raw256_to82_exact=True,actions_exact=True,
            reward_range=[float(reward.min()),float(reward.max())],terminated_frames=np.flatnonzero(terminated).tolist(),truncated_frames=np.flatnonzero(truncated).tolist()))
        if eid in (400010,400020,403000):
            for t in sorted(set([0,T-1]+[max(0,min(T-1,c+d)) for c in entry['transition_frames'] for d in (-9,0,9)])):
                all_samples.append(dict(episode=eid,frame=t,state82=s[t].tolist(),raw_state256=rs[t].tolist(),action23=a[t].tolist()))
        print(json.dumps(dict(step=ep_i+1,episode=eid,frames=T)),flush=True)
    writer.close();assert len(manifest)==200
    stat=pd.DataFrame(stats);aggregate=[]
    for label,g in stat.groupby('state',sort=False):
        r=dict(episode_index='ALL_200',state=label)
        for k in ['frames','positive','negative','unknown','adjacent_valid_switches']:r[k]=int(g[k].sum())
        r['recorded_switches_before_mask']=int(g.recorded_switches_before_mask.sum()) if g.switch_count_status.eq('known').all() else None
        r['switch_count_status']='known' if r['recorded_switches_before_mask'] is not None else 'unknown';aggregate.append(r)
    pd.concat([stat,pd.DataFrame(aggregate)],ignore_index=True).to_csv(OUT/'label_statistics.csv',index=False)
    write('samples/dataset_audit.json',dict(episodes=manifest,total_episodes=len(manifest),total_frames=sum(x['frames'] for x in manifest),
        summary=aggregate,ranges={k:dict(min=v[0].tolist(),max=v[1].tolist()) for k,v in ranges.items()}))
    write('samples/raw_state_action_samples.json',all_samples)
    write('samples/historical_started_certification.json',cert)
    for name in ('label_alignment_400010.json','label_alignment_400010.png','started_alignment_400020.png','started_alignment_403000.png','episode_00400010_audit.npz'):
        if (HISTORY/name).exists():shutil.copy2(HISTORY/name,OUT/'samples'/name)
    print(json.dumps(dict(passed=True,episodes=200,frames=sum(x['frames'] for x in manifest),summary=aggregate)))


if __name__=='__main__':main()
