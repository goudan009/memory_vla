"""CPU audit of the *saved* processor against real Task40 actions; no training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v1_groot_memoryvla.data.state_action.state_action_processor import StateActionProcessor
from v1_groot_memoryvla.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader


def audit(checkpoint, dataset):
    config = json.loads((checkpoint/'processor_config.json').read_text())['processor_kwargs']
    statistics = json.loads((checkpoint/'statistics.json').read_text())
    tag = 'behavior_r1_pro'
    processor = StateActionProcessor(config['modality_configs'], statistics=statistics,
        **{k: config[k] for k in ('use_percentiles', 'clip_outliers', 'apply_sincos_state_encoding', 'use_relative_action')})
    processor.eval()
    loader=LeRobotEpisodeLoader(dataset,processor.modality_configs[tag],video_backend='pyav')
    split = json.loads((checkpoint/'v2_1_stage1.json').read_text())['split']
    info = json.loads((dataset/'meta/info.json').read_text())
    groups = config['modality_configs'][tag]['action']['modality_keys']
    states = config['modality_configs'][tag]['state']['modality_keys']
    result = dict(checkpoint=str(checkpoint), dataset=str(dataset), config={k: config[k] for k in ('use_relative_action','use_percentiles','clip_outliers')},
                  file_sha256={n:hashlib.sha256((checkpoint/n).read_bytes()).hexdigest()
                               for n in ('config.json','processor_config.json','statistics.json','v2_1_stage1.json','COMPLETE.json')},
                  split={k:len(split[k]) for k in ('train','validation')}, groups={})
    for k in groups:
        params=processor.norm_params[tag]['action'][k]
        span=np.asarray(params['max'])-np.asarray(params['min'])
        result['groups'][k]=dict(normalization_shape=list(span.shape), min_span=float(span.min()),
                                max_span=float(span.max()), denormalization_gain_max=float(span.max()/2),
                                near_constant=int(np.isclose(params['min'],params['max']).sum()),
                                train_max_roundtrip_error=0.,validation_max_roundtrip_error=0.,
                                train_clipped_elements=0,validation_clipped_elements=0, train_elements=0,validation_elements=0)
    sampled=[]
    h0_deltas={k:[] for k in ('torso','left_arm','right_arm')}
    h0_extremes={}
    for partition in ('train','validation'):
        for eid in split[partition]:
            chunk_size=info.get('chunks_size',1000)
            path=dataset/info['data_path'].format(episode_chunk=eid//chunk_size,episode_index=eid)
            # Dataset names its task-directory by episode//10000, see meta chunks_size.
            if not path.exists():
                matches=list((dataset/'data').glob(f'*/episode_{eid:08d}.parquet'))
                if len(matches)!=1:raise FileNotFoundError(path)
                path=matches[0]
            df=loader._load_parquet_data(eid)
            if partition=='train':
                for k,sk in [('torso','trunk_qpos'),('left_arm','arm_left_qpos'),('right_arm','arm_right_qpos')]:
                    delta=np.stack(df['action.'+k])-np.stack(df['state.'+sk])
                    h0_deltas[k].append(delta)
                    frame,channel=np.unravel_index(np.abs(delta).argmax(),delta.shape)
                    extreme=dict(episode=eid,frame=int(frame),channel=int(channel),delta=float(delta[frame,channel]),
                        state=np.asarray(df['state.'+sk].iloc[frame]).tolist(),
                        action=np.asarray(df['action.'+k].iloc[frame]).tolist())
                    if k not in h0_extremes or abs(extreme['delta'])>abs(h0_extremes[k]['delta']):h0_extremes[k]=extreme
            for frame in np.unique(np.linspace(0,len(df)-32,5,dtype=int)):
                q={k:np.stack(df['state.'+k].iloc[frame:frame+1]) for k in states}
                a={k:np.stack(df['action.'+k].iloc[frame:frame+32]) for k in groups}
                processor.clip_outliers=False
                raw=processor.apply_action(a,tag,q)
                processor.clip_outliers=True
                z=processor.apply_action(a,tag,q)
                recovered=processor.unapply_action(z,tag,q)
                # Batch decode must use the same observation anchor for all32 steps.
                batched=processor.unapply_action({k:v[None] for k,v in z.items()},tag,{k:v[None] for k,v in q.items()})
                for k in groups:
                    row=result['groups'][k]
                    error=float(np.max(np.abs(recovered[k]-a[k])))
                    field=partition+'_max_roundtrip_error'
                    row[field]=max(row[field],error)
                    row[partition+'_clipped_elements']+=int((np.abs(raw[k])>1.00001).sum())
                    row[partition+'_elements']+=int(raw[k].size)
                    np.testing.assert_allclose(batched[k][0],recovered[k],atol=2e-6,rtol=0)
                if len(sampled)<3:
                    sampled.append(dict(episode=eid,frame=int(frame),state={k:v.tolist() for k,v in q.items()},
                                        action_t0={k:v[0].tolist() for k,v in a.items()},
                                        normalized_t0={k:v[0].tolist() for k,v in z.items()},
                                        recovered_t0={k:v[0].tolist() for k,v in recovered.items()}))
    result['samples']=sampled
    result['train_h0_delta_distribution']={}
    for k,arrays in h0_deltas.items():
        values=np.abs(np.concatenate(arrays))
        result['train_h0_delta_distribution'][k]=dict(elements=int(values.size),
            abs_quantiles={str(p):np.quantile(values,p,axis=0).tolist() for p in (.5,.95,.99,.999,1.)},
            fraction_abs_gt_point1=(values>.1).mean(0).tolist(),extreme=h0_extremes[k])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=audit(args.checkpoint,args.dataset)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
