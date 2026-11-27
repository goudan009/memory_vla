"""Same observed RGB/state and same noise across Euler4/8/16; RPC fault check."""
import argparse
import json
from pathlib import Path

import numpy as np
from v1_groot_memoryvla.policy.server_client import PolicyClient


def main(recording,output,port,noise_sensitivity=False):
    entries=[json.loads(l) for l in (recording/'requests.jsonl').read_text().splitlines()]
    selected=[entries[i] for i in sorted(set([0,min(10,len(entries)-1),len(entries)-1]))]
    client=PolicyClient('172.28.62.17',port,strict=False)
    result=[]
    for entry in selected:
        z=np.load(recording/entry['file'])
        observation=dict(video={},state={},language=entry['language'])
        for key in z.files:
            for mode in ('state','video'):
                if key.startswith(mode+'.'):observation[mode][key[len(mode)+1:]]=z[key]
        if noise_sensitivity:
            variants={}
            for seed in (42,43,44):
                variants[seed]=client.call_endpoint('diagnose_actions',dict(observation=observation,seed=seed,steps=[4]))['4']
            arrays={str(seed)+'.'+k:np.asarray(v) for seed,value in variants.items() for k,v in value['action'].items()}
            result=dict(input_file=entry['file'],inference_steps=4,seeds=[42,43,44],pairwise={})
            for left,right in ((42,43),(42,44),(43,44)):
                result['pairwise'][f'{left}_vs_{right}']={k:dict(
                    mean_abs_difference=float(np.abs(np.asarray(v)-np.asarray(variants[right]['action'][k])).mean()),
                    max_abs_difference=float(np.abs(np.asarray(v)-np.asarray(variants[right]['action'][k])).max()))
                    for k,v in variants[left]['action'].items()}
            np.savez_compressed(output/'noise_sensitivity.npz',**arrays)
            (output/'noise_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2))
            return
        response=client.call_endpoint('diagnose_actions',dict(observation=observation,seed=42,steps=[4,8,16]))
        arrays={}
        summary=dict(input_file=entry['file'],seed=42,variants={})
        for step,value in response.items():
            raw=np.asarray(value['normalized_action'])
            arrays['raw_'+step]=raw
            row={}
            for group,a in value['action'].items():
                a=np.asarray(a)
                arrays[step+'.'+group]=a
                diff=np.abs(np.diff(a,axis=1))
                row[group]=dict(interior_abs_delta_mean=float(diff.mean()),interior_abs_delta_max=float(diff.max()),
                                min=float(a.min()),max=float(a.max()))
            summary['variants'][step]=dict(groups=row,raw_active_clipped_fraction=float((np.abs(raw[:,:32,:23])>1).mean()))
        for left,right in [('4','8'),('8','16'),('4','16')]:
            delta=np.abs(arrays['raw_'+left][:,:32,:23]-arrays['raw_'+right][:,:32,:23])
            summary[left+'_vs_'+right]=dict(normalized_mean=float(delta.mean()),normalized_max=float(delta.max()))
        np.savez_compressed(output/f'paired_{entry["index"]:06d}.npz',**arrays)
        result.append(summary)
    (output/'paired_integrator_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=5565)
    p.add_argument('--noise-sensitivity',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);main(args.recording,args.output,args.port,args.noise_sensitivity)
