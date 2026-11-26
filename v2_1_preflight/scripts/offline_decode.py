"""CPU decoder of the traced official registry grammar; verify against official output.

No float-pattern search: consume the system/object UUID registries sequentially,
using initialized schemas and variable-count headers. Refuse unknown schemas.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'v2_1_preflight'
RAW=Path('/nvme3/datasets/b1k2025_raw_task40_v2_audit/task-0040')
sys.path.insert(0,str(ROOT/'evaluation/scripts'))
from v2_recording_alignment import matching_demo


def uuid(name):
    return int(np.float32(int(hashlib.md5(name.encode()).hexdigest(),16)%100000000))


def main():
    p=argparse.ArgumentParser();p.add_argument('--official-report',type=Path,required=True)
    p.add_argument('--all',action='store_true')
    p.add_argument('--schema-only',action='store_true',help='Reuse archived schema without claiming a new comparison to original official NPZ files')
    a=p.parse_args()
    official=json.loads(a.official_report.read_text());schemas={uuid(s['name']):s for s in official['schemas']}
    assert len(schemas)==len(official['schemas'])
    system_names=['popcorn','cooked__popcorn'];system_ids={uuid(n):n for n in system_names}
    available={x['episode']:x for x in official['episodes']}
    files=sorted((ROOT/'assets/datasets/b1k2025_groot_task40_v1_200').glob('data/**/*.parquet'))
    results=[];checked=0;scene_poses=[]
    for file in files:
        eid=int(file.stem.split('_')[-1])
        if not a.all and eid not in available:continue
        actions=np.stack(pd.read_parquet(file,columns=['action']).action)
        with h5py.File(RAW/(file.stem+'.hdf5')) as f:
            scene=json.loads(f['data'].attrs['scene_file']);gname,_=matching_demo(f['data'],actions);g=f['data'][gname]
            names=scene['metadata']['task']['inst_to_name'];mn,bn=names['microwave.n.02_1'],names['popcorn__bag.n.01_1']
            prev={n:scene['state']['registry']['object_registry'][n] for n in (mn,bn)}
            # ControllerBase always consumes goal_dim+1 even if its goal flag is false.
            controllers=scene['state']['registry']['object_registry']['robot_r1']['controllers']
            controller_size=sum(1+len(c['goal']['target'])+(len(c['vel_filter']['past_samples'])*len(c['vel_filter']['past_samples'][0])+2 if 'vel_filter' in c else 0) for c in controllers.values())
            rows=g['state'][:];sizes=g['state_size'][:];T=len(actions)
            arrays={k:[] for k in ['microwave_joint','started','explicitly_recorded','bag_pose','microwave_pose','temperature','cooked_particle_count','raw_particle_count']}
            for t in range(T):
                row=rows[t,:int(sizes[t])];idx=7;scene_pose=row[:7]
                assert int(row[idx])==2 and int(row[idx+1])==uuid('system_registry')
                idx+=2
                nsystems=int(row[idx]);idx+=1;counts={n:0 for n in system_names}
                for _ in range(nsystems):
                    name=system_ids[int(row[idx])];idx+=1;n=int(row[idx]);counts[name]=n
                    # MacroPhysicalParticleSystem: Base7+7N, Macro3N+counter1, physical6N.
                    idx+=8+16*n
                assert int(row[idx])==uuid('object_registry');idx+=1
                nobjects=int(row[idx]);idx+=1;present=set()
                for _ in range(nobjects):
                    assert int(row[idx]) in schemas,(eid,t,_,idx,int(row[idx]),locals().get('name'),locals().get('sch'))
                    sch=schemas[int(row[idx])];idx+=1;name=sch['name'];start=idx
                    nj=len(sch['joints'])
                    original=scene['state']['registry']['object_registry'].get(name)
                    assert original is not None,(eid,t,name,'missing original rigid schema')
                    base_size=14 if 'lin_vel' in original['root_link'] else 8
                    idx+=base_size+2*nj
                    if name=='robot_r1':idx+=controller_size
                    nonkin={}
                    for state in sch['stateful']:
                        if state=='Temperature':nonkin[state]={'temperature':float(row[idx])};idx+=1
                        elif state=='ToggledOn':nonkin[state]={'value':bool(row[idx])};idx+=2
                        elif state=='AttachedTo':idx+=1
                        elif state=='ParticleRemover':idx+=1
                        elif state=='ModifiedParticles':idx+=1+2*int(row[idx])
                        elif state=='Saturated':idx+=3+2*int(row[idx])
                        else:raise ValueError((name,state))
                    assert idx<=len(row),(eid,t,name,idx,len(row))
                    if name in prev:
                        present.add(name)
                        prev[name]=dict(root_link=dict(pos=row[start+1:start+4],ori=row[start+4:start+8]),joint_pos=row[start+base_size:start+base_size+nj],non_kin=nonkin)
                assert idx==len(row),(eid,t,idx,len(row))
                m,b=prev[mn],prev[bn]
                arrays['microwave_joint'].append(m['joint_pos'][0]);arrays['started'].append(m['non_kin']['ToggledOn']['value'])
                arrays['explicitly_recorded'].append([mn in present,bn in present])
                for key,obj in [('bag_pose',b),('microwave_pose',m)]:arrays[key].append(np.r_[obj['root_link']['pos'],obj['root_link']['ori']])
                arrays['temperature'].append([m['non_kin']['Temperature']['temperature'],b['non_kin']['Temperature']['temperature']])
                arrays['cooked_particle_count'].append(counts['cooked__popcorn']);arrays['raw_particle_count'].append(counts['popcorn'])
            arrays={k:np.asarray(v,dtype=bool if k in ('started','explicitly_recorded') else np.int32 if k.endswith('count') else np.float32) for k,v in arrays.items()}
            target=OUT/'samples/decoded'/(file.stem+'.npz')
            compared=eid in available and not a.schema_only
            if compared:
                assert hashlib.sha256(target.read_bytes()).hexdigest()==available[eid]['sha256'],('Original official reference NPZ required',eid,'Use --schema-only for a fresh extraction without independent reference comparison')
                with np.load(target) as z:
                    for k in z.files:np.testing.assert_array_equal(arrays[k],z[k],err_msg=f'{eid}:{k}')
                checked+=T
            if a.all:np.savez_compressed(target,**arrays)
            results.append(dict(episode=eid,frames=T,artifact=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None,official_framewise_comparison=compared,selected_raw_group=gname))
            scene_poses.append(dict(episode=eid,scene_pose=scene_pose.tolist()))
        print(json.dumps(dict(episode=eid,complete=len(results),official_compared_frames=checked)),flush=True)
    report=dict(official,method='Source-traced CPU sequential registry decoder; schema-only rerun, no new independent comparison' if a.schema_only else 'Source-traced CPU sequential registry decoder, exact framewise comparison to initialized official deserializers',episodes=results,
        passed=len(results)==200 if a.all else bool(results),official_reference=str(a.official_report),official_compared_episodes=sum(x['official_framewise_comparison'] for x in results),official_compared_frames=checked,
        no_physics_replay=True,scene_poses=scene_poses,grammar_sources=['scenes/scene_base.py:1168','utils/registry_utils.py:476','prims/entity_prim.py:1699','objects/controllable_object.py:778','objects/stateful_object.py:606','systems/macro_particle_system.py:1519','systems/system_base.py:376','object_states/saturated.py:83,276'])
    dest=OUT/'samples'/('cpu_redecode_report.json' if a.schema_only else 'cpu_decode_report.json' if a.all else 'cpu_decoder_comparison.json')
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],episodes=len(results),compared_frames=checked)))


if __name__=='__main__':main()
