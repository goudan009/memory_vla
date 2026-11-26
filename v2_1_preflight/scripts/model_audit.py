"""Inspect existing CPU feature cache and archive actual source; no model execution."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'v2_1_preflight'
BACKUP=Path('/nvme3/projects/memory_vla-v2-v3-backup-oEWaKh')


def main():
    current='0d28bc506c3def15b2f76bdbdc0a3ccb1b4567a8'
    historical='b710fc4df586216b4b6a1a3776f09c725300cd98'
    paths=['src/v1_groot_memoryvla/model/modules/eagle_backbone.py',
        'src/v1_groot_memoryvla/model/modules/nvidia/Eagle-Block2A-2B-v2/modeling_eagle3_vl.py',
        'src/v1_groot_memoryvla/model/memory/adapter.py','src/v1_groot_memoryvla/model/memory/bank.py',
        'src/v1_groot_memoryvla/model/gr00t_n1d6/gr00t_n1d6.py',
        'src/v1_groot_memoryvla/model/gr00t_n1d6/processing_gr00t_n1d6.py',
        'src/v1_groot_memoryvla/data/convert_b1k.py','src/v1_groot_memoryvla/data/dataset/episode_windows.py',
        'src/v1_groot_memoryvla/data/dataset/history_dataset.py','src/v1_groot_memoryvla/experiment/task21_formal.py',
        'src/v1_groot_memoryvla/data/utils.py',
        'src/v1_groot_memoryvla/data/state_action/state_action_processor.py',
        'src/v1_groot_memoryvla/data/state_action/action_chunking.py',
        'src/v1_groot_memoryvla/data/state_action/pose.py',
        'src/v1_groot_memoryvla/configs/data/embodiment_configs.py',
        'evaluation/scripts/v2_recording_alignment.py',
        'evaluation/scripts/v2_label_audit.py',
        'evaluation/scripts/v2_certify_started.py']
    paths += [str(p.relative_to(BACKUP)) for p in (BACKUP/'src/v2_groot_memoryvla').rglob('*.py')]
    sources=[]
    for rel in paths:
        version='historical_v2' if rel.startswith('src/v2_') else 'current_v1'
        src=(BACKUP if version=='historical_v2' else ROOT)/rel
        if not src.exists():continue
        blob=src.read_bytes();commit=historical if version=='historical_v2' else current
        git=subprocess.run(['git','show',commit+':'+rel],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        dest=OUT/'samples/source_evidence'/version/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        sources.append(dict(source=str(src),bundle=str(dest.relative_to(OUT)),sha256=hashlib.sha256(blob).hexdigest(),
            git_commit=commit if git.returncode==0 and git.stdout==blob else None,
            compared_git_commit=commit,byte_equal_git_blob=git.returncode==0 and git.stdout==blob,git_error=git.stderr.decode() if git.returncode else None))
    cache=ROOT/'outputs/diagnostics/v1_real_replay_features.pt'
    x=torch.load(cache,map_location='cpu',mmap=True,weights_only=False)
    sample=x['samples'][0];features=sample['features'];ids=sample['input_ids'];attention=features['backbone_attention_mask']
    positions=torch.arange(ids.shape[1]).expand_as(ids);indices=positions.masked_fill(~((ids==151645)&attention),-1).max(1).values
    eos=features['backbone_features'][torch.arange(len(ids)),indices]
    assert torch.equal(eos[0],x['eos'][0]['eos'])
    def shapes(d):return {k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in d.items()}
    samples=[]
    for s in x['samples']:
        samples.append(dict(episode=s['episode'],task=s['task'],timestep=s['timestep'],features=shapes(s['features']),actions=shapes(s['actions']),predictions=shapes(s['predictions'])))
    arrays={k:v.float().numpy() for k,v in features.items()}
    arrays.update(input_ids=ids.numpy(),eos=eos.float().numpy(),state=sample['actions']['state'].float().numpy(),action_mask=sample['actions']['action_mask'].float().numpy())
    np.savez_compressed(OUT/'samples/real_feature_sample.npz',**arrays)
    result=dict(method='Read-only CPU load of an existing real replay feature artifact; no new forward or ablation performed',
        cache=str(cache),cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),sample_count=len(samples),samples=samples,
        first_features=shapes(features),first_action_inputs=shapes(sample['actions']),eos_shape=list(eos.shape),eos_index=indices.tolist(),
        eos_equals_existing_capture=True,image_token_count=features['image_mask'].sum(1).tolist(),
        effective_mask_count=sample['actions']['action_mask'].sum().item(),
        caution='Cache metadata task=0/episode=10 retained verbatim; this proves real three-camera Eagle and action tensor interfaces, not freshness or full source revision identity of historical cache',sources=sources)
    (OUT/'samples/model_interface_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(samples=len(samples),source_files=len(sources),git_exact=sum(s['byte_equal_git_blob'] for s in sources),eos_verified=True)))


if __name__=='__main__':main()
