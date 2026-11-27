"""Read-only collection of the actual Stage1 run and pinned upstream sources."""
import argparse
import json
from pathlib import Path
import subprocess


REMOTE = r'''
import hashlib,json,math,pathlib,statistics
P=pathlib.Path
root=P('/nvme3/projects/memory_vla')
deployment=root/'deployments/v2.1__test01__exp14__run001'
run=P('/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/v2.1__test01__exp14__run001/run')
checkpoint=run/'checkpoints/v2.1__test01__exp14__run001__step006000'
official=P('/nvme3/projects/GR00T/Isaac-GR00T-n1d6-9b37aa1/gr00t')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
model_source=deployment/'src/v2_1_groot_memoryvla'
h=hashlib.sha256()
for p in sorted(model_source.rglob('*')):
    if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts:
        h.update(str(p.relative_to(model_source)).encode());h.update(p.read_bytes())
rows=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines() if l.strip()]
def dist(values):
    values=sorted(values)
    return dict(n=len(values),min=values[0],median=statistics.median(values),max=values[-1],mean=statistics.mean(values))
result=dict(run=str(run),checkpoint=str(checkpoint),training_config=json.loads((run/'config.json').read_text()),
    complete=json.loads((checkpoint/'COMPLETE.json').read_text()),actual_model_source_digest=h.hexdigest(),
    source_hashes={},validation={},inference={},metrics={})
roots=[deployment,root/'deployments/inference_cfeabb58a50b5aaec34679524f8e441dbea508d775b7104b65916cd3aa67a21a',
       root/'deployments/inference_13b51a8721ce3cdca2c4d57c616f0923586baf155e9e4e39cb91cf2b031d1395']
result['training_serving_source_comparison']={}
for name in ['v1_groot_memoryvla/model/gr00t_n1d6/gr00t_n1d6.py','v1_groot_memoryvla/model/modules/dit.py',
             'v1_groot_memoryvla/model/modules/eagle_backbone.py','v1_groot_memoryvla/model/gr00t_n1d6/processing_gr00t_n1d6.py',
             'v1_groot_memoryvla/data/state_action/state_action_processor.py','v1_groot_memoryvla/configs/model/gr00t_n1d6.py',
             'v2_1_groot_memoryvla/loading.py','v2_1_groot_memoryvla/baseline_policy.py']:
    result['training_serving_source_comparison'][name]={str(r):sha(r/'src'/name) if (r/'src'/name).exists() else None for r in roots}
for base,names in [(deployment,['src/v2_1_groot_memoryvla/train.py','src/v2_1_groot_memoryvla/loading.py','src/v2_1_groot_memoryvla/data/windows.py','src/v2_1_groot_memoryvla/distributed.py']),
                   (official,['model/gr00t_n1d6/gr00t_n1d6.py','configs/finetune_config.py','configs/training/training_config.py','experiment/trainer.py','policy/gr00t_policy.py'])]:
    for name in names:
        p=base/name
        if p.exists():result['source_hashes'][str(p)]=sha(p)
for p in sorted(run.glob('validation-*.json')):result['validation'][p.name]=json.loads(p.read_text())
for key in ('step','epoch','decisions','action_elements','grad_norm','fm','seconds','lr'):
    values=[r[key] for r in rows if isinstance(r.get(key),(int,float)) and math.isfinite(r[key])]
    if values:result['metrics'][key]=dist(values)
result['metrics']['last']=rows[-1]
result['metrics']['nonfinite_gradient_steps']=[r['step'] for r in rows if not math.isfinite(r.get('grad_norm',0))]
from collections import Counter
result['metrics']['updates_per_epoch']=dict(Counter(r['epoch'] for r in rows))
dataset=P(result['training_config']['dataset'])
split=json.loads((run/'data_split.json').read_text())
meta=[json.loads(l) for l in (dataset/'meta/episodes.jsonl').read_text().splitlines() if l.strip()]
lengths={r['episode_index']:r['length'] for r in meta}
result['sampling_coverage']=dict(raw_train_observation_frames=sum(lengths[e] for e in split['train']),
    unique_train_anchor_frames_per_epoch=sum(len(range(0,lengths[e],8)) for e in split['train']),
    anchor_phase=0,anchor_stride=8,future_action_offsets=list(range(32)),
    seen_decisions=sum(r['decisions'] for r in rows),
    note='Only one eighth of observation anchors are used; future action supervision still spans offsets0..31 and is not limited to phase0.')
for name,path in [('stage1',checkpoint),('official_behavior',P('/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k')),('generic_init',P('/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B'))]:
    cfg=json.loads((path/'config.json').read_text())
    proc=json.loads((path/'processor_config.json').read_text())
    result['inference'][name]=dict(path=str(path),config=cfg,processor=proc,
        hashes={n:sha(path/n) for n in ('config.json','processor_config.json','statistics.json') if (path/n).exists()})
    stats=json.loads((path/'statistics.json').read_text()).get('behavior_r1_pro') if (path/'statistics.json').exists() else None
    if stats:result['inference'][name]['r1pro_statistics']=stats
print(json.dumps(result))
'''


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--host', default='172.28.62.17')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    answer = subprocess.run(['ssh','-o','BatchMode=yes',args.host,'python3','-'],input=REMOTE,
                            text=True,capture_output=True,check=True)
    result = json.loads(answer.stdout)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(output=str(args.output),digest=result['actual_model_source_digest'],metrics=result['metrics']),indent=2))
