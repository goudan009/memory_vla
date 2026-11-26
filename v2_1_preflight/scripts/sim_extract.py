"""Read recorded states through initialized official deserializers; never step replay.

Executed only by the existing BenchmarkManager. No training or policy inference.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BEHAVIOR = Path('/nvme3/projects/GR00T/BEHAVIOR-1K-task-progress')
DATA = ROOT/'assets/datasets/b1k2025_groot_task40_v1_200'
RAW = Path('/nvme3/datasets/b1k2025_raw_task40_v2_audit/task-0040')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = ROOT/'v2_1_preflight/samples/decoded'
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(ROOT/'evaluation/scripts'), str(BEHAVIOR/'OmniGibson'), str(BEHAVIOR/'bddl')]
    from task40_restore_worker import lease_sim_cache
    lease = lease_sim_cache(args.output.parent/'isaac_appdata', 0, 2)
    import isaacsim  # noqa: F401
    import h5py
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    import omnigibson as og
    from omnigibson.envs import DataPlaybackWrapper
    from omnigibson.macros import gm
    from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
    from omnigibson import object_states
    from omnigibson.object_states.open_state import _compute_joint_threshold
    from v2_recording_alignment import matching_demo
    gm.ENABLE_TRANSITION_RULES = False
    gm.RENDER_VIEWER_CAMERA = False
    raw = RAW/'episode_00400010.hdf5'
    full = next(p for p in sorted((BEHAVIOR/'joylo/sampled_task/make_microwave_popcorn').glob('*.json'))
                if 'partial_rooms' not in p.name)
    with (Path(gm.DATA_PATH)/'2025-challenge-task-instances/metadata/B50_task_misc.csv').open() as f:
        rooms = next(row[2].strip().split('\n') for row in csv.reader(f) if 'make_microwave_popcorn' in row[1])
    env = DataPlaybackWrapper.create_from_hdf5(input_path=str(raw),
        output_path=str(args.output.parent/'unused_playback.hdf5'), robot_obs_modalities=['proprio'],
        robot_proprio_keys=list(PROPRIOCEPTION_INDICES['R1Pro']),
        robot_sensor_config={'VisionSensor': {'modalities': []}}, external_sensors_config={},
        n_render_iterations=1, full_scene_file=str(full), load_room_instances=rooms,
        include_robot_control=False, include_contacts=False, flush_every_n_traj=1)
    microwave = env.task.object_scope['microwave.n.02_1'].wrapped_obj
    both, joints, directions = microwave.states[object_states.Open].relevant_joints_info
    thresholds = [dict(name=j.joint_name, lower=float(j.lower_limit), upper=float(j.upper_limit),
                       direction=d, threshold=float(_compute_joint_threshold(j,d)[0]))
                  for j,d in zip(joints,directions)]
    # Capture actual initialized serialization schemas / object-state availability.
    schemas = []
    optimized = []
    for obj in env.scene.objects:
        schemas.append(dict(name=obj.name, class_name=type(obj).__name__,
            joints=list(obj.joints), states=[c.__name__ for c in getattr(obj,'states',{})],
            stateful=[c.__name__ for c,s in getattr(obj,'states',{}).items() if s.stateful]))
        # These serializers have fixed lengths: rigid root pose/vel, fixed nDOF,
        # fixed controller shapes, and scalar Temperature/ToggledOn/AttachedTo.
        # Keep the official registry's UUID iteration. Derive the skipped length
        # by calling each object's official deserializer once, never by scanning.
        states={c.__name__ for c,s in getattr(obj,'states',{}).items() if s.stateful}
        if obj.name not in ('microwave_hjjxmi_0','popcorn_bag_73') and states <= {'Temperature','ToggledOn','AttachedTo'}:
            original=obj.deserialize
            cache={}
            def skip_fixed(state,original=original,cache=cache):
                if 'size' not in cache:
                    _,cache['size']=original(state)
                return {},cache['size']
            optimized.append((obj,original,skip_fixed))
            obj.deserialize=skip_fixed
    report = dict(passed=False, method='official og.sim.deserialize, no load_state / no physics stepping',
                  open_both_sides=both, open_joints=thresholds, schemas=schemas, episodes=[],
                  optimization='Skip only unrelated fixed-length object payloads after official size derivation; official UUID registry and variable states unchanged; full parse comparison each500 frames')
    previous_report=ROOT/'outputs/benchmarks/web/20261126-200257-15a1de03/preflight_extract.json'
    reusable={x['episode']:x for x in json.loads(previous_report.read_text())['episodes']} if previous_report.exists() else {}
    def save():
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for p in sorted(DATA.glob('data/**/*.parquet')):
        eid=int(p.stem.split('_')[-1]); start=time.monotonic()
        if eid in reusable:
            item=reusable[eid];target=Path(item['artifact'])
            assert hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']
            report['episodes'].append(dict(item,reused_from=str(previous_report)))
            save();print(json.dumps(dict(step=len(report['episodes']),episode=eid,reused=True)),flush=True)
            continue
        frame=pd.read_parquet(p,columns=['action','frame_index','timestamp'])
        with h5py.File(RAW/(p.stem+'.hdf5')) as f:
            scene=json.loads(f['data'].attrs['scene_file'])
            name, attempts=matching_demo(f['data'],np.stack(frame.action).astype(np.float32))
            group=f['data'][name]
            for transition in json.loads(group.attrs['transitions']).values():
                for system in transition['systems']['add']:
                    env.scene.get_system(system,force_init=True)
            names=scene['metadata']['task']['inst_to_name']
            mn,bn=names['microwave.n.02_1'],names['popcorn__bag.n.01_1']
            previous={k:scene['state']['registry']['object_registry'][k] for k in (mn,bn)}
            vectors,sizes=group['state'][:],group['state_size'][:]
            values=[]; present=[]; bagpose=[]; mpose=[]; temp=[]; sizes_seen=[]
            for t in range(len(frame)):
                size=int(sizes[t]); state,used=og.sim.deserialize(torch.from_numpy(vectors[t,:size]))
                assert used==size,(eid,t,used,size)
                objects=state[0]['registry']['object_registry']
                if t%500==0 or t==len(frame)-1:
                    for obj,original,fast in optimized:obj.deserialize=original
                    full_state,full_used=og.sim.deserialize(torch.from_numpy(vectors[t,:size]))
                    for obj,original,fast in optimized:obj.deserialize=fast
                    assert full_used==used
                    for k in (mn,bn):
                        assert (k in objects)==(k in full_state[0]['registry']['object_registry'])
                        if k in objects:
                            other=full_state[0]['registry']['object_registry'][k]
                            for field in ('pos','ori'):
                                assert torch.equal(objects[k]['root_link'][field],other['root_link'][field])
                            if k==mn:
                                assert torch.equal(objects[k]['joint_pos'],other['joint_pos'])
                                assert objects[k]['non_kin']['ToggledOn']==other['non_kin']['ToggledOn']
                present.append([mn in objects,bn in objects])
                for k in (mn,bn):
                    if k in objects: previous[k]=objects[k]
                m,b=previous[mn],previous[bn]
                def array(x): return np.asarray(x,dtype=np.float32)
                mq=array(m['joint_pos']).reshape(-1)
                values.append([float(mq[0]),float(m['non_kin']['ToggledOn']['value'])])
                bagpose.append(np.r_[array(b['root_link']['pos']),array(b['root_link']['ori'])])
                mpose.append(np.r_[array(m['root_link']['pos']),array(m['root_link']['ori'])])
                temp.append([float(m['non_kin']['Temperature']['temperature']),float(b['non_kin']['Temperature']['temperature'])])
                if t%1000==0: print(json.dumps(dict(episode=eid,frame=t,step=len(report['episodes']))),flush=True)
            values=np.asarray(values,dtype=np.float32)
            target=out/(p.stem+'.npz')
            np.savez_compressed(target,microwave_joint=values[:,0],started=values[:,1].astype(bool),
                explicitly_recorded=np.asarray(present,dtype=bool),bag_pose=np.asarray(bagpose),
                microwave_pose=np.asarray(mpose),temperature=np.asarray(temp))
            report['episodes'].append(dict(episode=eid,frames=len(frame),selected_raw_group=name,
                attempts=attempts,seconds=time.monotonic()-start,artifact=str(target),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
            save()
        print(json.dumps(dict(step=len(report['episodes']),episode=eid,complete=True)),flush=True)
    report['passed']=len(report['episodes'])==200
    save()
    og.shutdown()


if __name__=='__main__': main()
