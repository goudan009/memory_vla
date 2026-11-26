"""Single-run V2 joint training with audited physical labels and frozen Eagle.

Launch only through BenchmarkManager. Windows remain chronological within an
episode; history is detached between microbatches, never replaced with truth.
"""
import argparse
import csv
import gc
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from v1_groot_memoryvla.data.dataset.episode_windows import EpisodeWindows
from v1_groot_memoryvla.experiment.first_training import cache_sequence, seeded, slice_features
from v1_groot_memoryvla.experiment.task21_diagnostic import load_model, head_hash
from v1_groot_memoryvla.experiment.task21_formal import window_context
from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from .model.gr00t_n1d6 import Gr00tN1d6
from .training import parameter_manifest
from .training_smoke import file_hash, selected_parameters
from .training_smoke import FEATURE_KEYS
from .model.state import masked_state_loss


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def validate_label_manifest(manifest, split):
    if manifest.get('labels_certified') is not True or manifest.get('label_names') != ['started']:
        raise ValueError('Training requires the certified recorded started label')
    train, validation = set(split['train']), set(split['validation'])
    if len(train) != 180 or len(validation) != 20 or train & validation:
        raise ValueError('Expected the locked disjoint 180/20 episode split')
    episodes = {e['episode']: e for e in manifest['episodes']}
    if len(episodes) != 200 or set(episodes) != train | validation:
        raise ValueError('Label coverage does not match the locked split')
    return episodes


def window_labels(archive, frames):
    labels, known = archive['labels'], archive['known']
    if labels.shape != known.shape or labels.ndim != 2 or labels.shape[1] != 1 or known.dtype != np.bool_:
        raise ValueError('Invalid physical label array')
    if not frames or min(frames) < 0 or max(frames) >= len(labels):
        raise ValueError('Observation frame outside physical label recording')
    return torch.from_numpy(labels[frames].copy()), torch.from_numpy(known[frames].copy())


def joint_weights(sequence, start, end):
    """Normalize action elements and known state entries independently."""
    mask = sequence['actions']['action_mask']
    action = float(mask[start:end].sum()) / max(float(mask.sum()), 1.)
    known = sequence['state_known']
    state = float(known[start:end].sum()) / max(float(known.sum()), 1.)
    return action, state


def part_loss_sums(unreduced, parts):
    """The action head has already applied the padding mask to this loss."""
    return torch.stack([unreduced.detach()[..., p['start']:p['end']].float().sum()
                        for p in parts.values()]).cpu().tolist()


def inputs(sequence, start, end):
    return dict(slice_features(sequence['features'], start, end),
        **slice_features(sequence['actions'], start, end),
        input_ids=sequence['input_ids'][start:end].cuda(),
        state_targets=sequence['state_targets'][start:end].cuda(),
        state_known=sequence['state_known'][start:end].cuda(),
        memory_context=window_context(sequence, start, end))


def cached_forward(model, seq, start, end):
    """Same joint graph as V2.forward, starting at this window's frozen Eagle output.

    Keep the real backbone attached for complete checkpoint serialization. Never
    feed Eagle's own output back through its image-input preparation method.
    """
    batch = inputs(seq,start,end)
    features = model.memory(BatchFeature({k:batch[k] for k in FEATURE_KEYS}),
                            batch['input_ids'],batch['memory_context'])
    action_inputs = BatchFeature({k:batch[k] for k in ('state','action','action_mask','embodiment_id')})
    output = model.action_head(features,action_inputs)
    output['action_loss_unreduced'] = output['action_loss']
    output['action_loss'] = output['loss']
    output['state_loss'] = masked_state_loss(features['state_logits'],batch['state_targets'],batch['state_known'])
    output['loss'] = output['action_loss']+model.config.memory_state_loss_weight*output['state_loss']
    for key in ('state_logits','state_feature','retrieved_memory','memory_gate'):
        output[key] = features[key]
    return output


def sequence(model, processor, dataset, index, episodes, archives, augmentation_seed):
    # Current frozen checkpoint and processor, fresh training augmentation.
    seq = cache_sequence(model, processor, dataset, index, microbatch=2,
                         augmentation_seed=augmentation_seed)
    eid = seq['episode']
    if eid not in archives:
        item = episodes[eid]
        path = Path(item['artifact'])
        if file_hash(path) != item['artifact_sha256']:
            raise ValueError(f'Label artifact changed: {eid}')
        with np.load(path, allow_pickle=False) as data:
            archives[eid] = {k:data[k].copy() for k in ('labels','known')}
        if len(archives[eid]['labels']) != dataset.episode_lengths[0,eid]:
            raise ValueError(f'Observation/physical-label length differs: {eid}')
    seq['state_targets'], seq['state_known'] = window_labels(archives[eid],seq['frame_indices'])
    return seq


def frozen_hash(model):
    import hashlib
    digest = hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def preflight(model, seq):
    model.train()
    model.memory.reset()
    # This is a no-update probe, not another training run.
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        cached_forward(model,seq,0,2)
    if len(model.memory.cognitive.bank[seq['episode_key']]) != 2:
        raise RuntimeError('Preflight requires two genuine past slots')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output = cached_forward(model,seq,2,3)
    tracked = selected_parameters(model)
    action_names = [n for n in tracked if n != 'state_head']
    grads = torch.autograd.grad(output['action_loss'],[tracked[n] for n in action_names],retain_graph=True)
    norms = {n:float(g.float().norm()) for n,g in zip(action_names,grads)}
    state_grad, = torch.autograd.grad(output['state_loss'],tracked['state_head'])
    norms['state_head'] = float(state_grad.float().norm())
    if not all(np.isfinite(v) and v>0 for v in norms.values()):
        raise RuntimeError(f'Disconnected/nonfinite real-label gradients: {norms}')
    action, state = float(output['action_loss'].detach()),float(output['state_loss'].detach())
    weight = .1*action/state
    if not np.isfinite(weight) or weight <= 0:
        raise RuntimeError('Cannot calibrate state loss weight')
    model.memory.reset()
    return dict(action_loss=action,state_loss=state,gradients=norms,
                state_loss_weight=weight,rule='0.1 * real action_loss / real state_loss, one calibration, no search')


def save_and_reload(model, processor, directory, seq):
    model.eval()
    model.memory.reset()
    # Fixed real prefix, followed by a fixed-noise action-loss forward.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        cached_forward(model,seq,0,2)
        runtime = model.memory.runtime_state_dict()
        seeded(9876)
        before = cached_forward(model,seq,2,3)
    keys = ('action_loss_unreduced','state_logits','state_feature','retrieved_memory','memory_gate')
    before = {k:before[k].detach().cpu() for k in keys}
    model.memory.reset()
    directory.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(directory,max_shard_size='5GB')
    processor.save_pretrained(directory)
    # Reuse the existing model's storage while auditing the HF loading contract.
    # CPU-only reload keeps the second copy off the training GPU.
    restored = Gr00tN1d6.from_pretrained(directory,torch_dtype=torch.float32,
        transformers_loading_kwargs={'trust_remote_code':True,'local_files_only':True})
    restored.backbone.bfloat16()
    original_state = model.state_dict()
    differences = {}
    for name,value in restored.state_dict().items():
        target = original_state[name].detach().cpu()
        if not torch.equal(value.cpu().to(target.dtype),target):
            raise RuntimeError(f'Full checkpoint tensor mismatch: {name}')
    del original_state,target,value
    # Transfer ownership of GPU modules after checking all serialized tensors.
    # Free the original GPU copy before moving the restored model onto CUDA.
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    restored.backbone.cuda()
    restored.action_head.cuda()
    restored.memory.cuda()
    restored.eval()
    restored.memory.load_runtime_state_dict(runtime)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        seeded(9876)
        after = cached_forward(restored,seq,2,3)
    for key in keys:
        differences[key] = float((before[key]-after[key].detach().cpu()).float().abs().max())
    if any(v != 0 for v in differences.values()):
        raise RuntimeError(f'Full HF forward reload differs: {differences}')
    restored.memory.reset()
    loaded_processor = Gr00tN1d6Processor.from_pretrained(directory,
        transformers_loading_kwargs={'trust_remote_code':True,'local_files_only':True})
    write_json(directory/'COMPLETE.json',dict(complete=True,reload_max_abs=differences,
        sha256={p.name:file_hash(p) for p in directory.glob('*.safetensors')}))
    return restored,loaded_processor,differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--split',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1000)
    args = parser.parse_args()
    if args.steps != 1000:
        parser.error('This single exploratory run is locked to 1000 optimizer steps')
    start = time.monotonic()
    torch.set_num_threads(4)
    seeded(42)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    split = json.loads(args.split.read_text())
    manifest = json.loads(args.labels.read_text())
    episodes = validate_label_manifest(manifest,split)
    processor = Gr00tN1d6Processor.from_pretrained(args.checkpoint,
        transformers_loading_kwargs={'trust_remote_code':True,'local_files_only':True})
    processor.train()
    dataset = EpisodeWindows(args.dataset,processor,split['train'],group_size=32)
    old = load_model(args.checkpoint,True,42)
    model = Gr00tN1d6.from_v1_model(old,state_labels=('started',),state_loss_weight=1.)
    del old
    model.backbone.cuda()
    model.train()
    original_frozen = frozen_hash(model)
    order = dataset.order(42)
    archives = {}
    probe = sequence(model,processor,dataset,order[0],episodes,archives,42)
    calibration = preflight(model,probe)
    parts = json.loads((args.dataset/'meta/modality.json').read_text())['action']
    model.config.memory_state_loss_weight = calibration['state_loss_weight']
    config = dict(category='exploratory_validation',model_version='v2',test_id='test02',experiment_id='exp01',
        run_id='run001',steps=1000,seed=42,world_size=1,microbatch=2,grad_accum_steps=16,
        effective_batch=32,batch_unit='chronological observation decisions; shorter episode tails use actual count',
        learning_rate=1e-5,weight_decay=.01,optimizer='AdamW',scheduler='constant',gradient_clip=1.,
        precision='BF16 autocast; FP32 cognitive parameters',state_labels=['started'],calibration=calibration,
        action_parts=parts,
        label_manifest=str(args.labels),label_manifest_sha256=file_hash(args.labels),
        data_split=str(args.split),data_split_sha256=file_hash(args.split),
        base_checkpoint=str(args.checkpoint),frozen_sha256=original_frozen,
        augmentation='checkpoint processor training transform, fresh frozen Eagle each window',
        parameter_manifest=parameter_manifest(model),
        holdout_scope='V2 updates only; V1 initialization has seen all 200 demonstrations')
    config.update(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  executed_runner_sha256=file_hash(Path(__file__)),
                  base_checkpoint_sha256={p.name:file_hash(p) for p in args.checkpoint.glob('*.safetensors')})
    write_json(args.output.parent/'train_config.json',config)
    model,processor,reload_preflight = save_and_reload(model,processor,args.output.parent/'preflight_full_checkpoint',probe)
    if frozen_hash(model) != original_frozen:
        raise RuntimeError('Frozen parameters changed during reload')
    processor.train()
    dataset.processor = processor
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=1e-5,weight_decay=.01,foreach=False)
    tracked = selected_parameters(model)
    initial = {n:p.detach().clone() for n,p in tracked.items()}
    samples = 0
    monitor = args.output.parent/'train_monitor.csv'
    with monitor.open('x',newline='') as stream:
        writer = None
        for step in range(1,args.steps+1):
            tick = time.monotonic()
            epoch,position = divmod(step-1,len(order))
            if position == 0:
                order = dataset.order(42+epoch)
            seq = sequence(model,processor,dataset,order[position],episodes,archives,42+step)
            optimizer.zero_grad(set_to_none=True)
            count = len(seq['timesteps'])
            sums = dict(action_loss=0.,state_loss=0.)
            per_part = np.zeros(len(parts))
            part_counts = [float(seq['actions']['action_mask'][...,p['start']:p['end']].sum())
                           for p in parts.values()]
            for begin in range(0,count,2):
                end = min(begin+2,count)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output = cached_forward(model,seq,begin,end)
                    aw,sw = joint_weights(seq,begin,end)
                    loss = output['action_loss']*aw+model.config.memory_state_loss_weight*output['state_loss']*sw
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss at optimizer step {step}')
                loss.backward()
                sums['action_loss'] += float(output['action_loss'].detach())*aw
                sums['state_loss'] += float(output['state_loss'].detach())*sw
                per_part += part_loss_sums(output['action_loss_unreduced'], parts)
            norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError('Frozen branch received parameter gradients')
            optimizer.step()
            if step == 1:
                updates = {n:float((p.detach()-initial[n]).abs().max()) for n,p in tracked.items()}
                if not all(v>0 and np.isfinite(v) for v in updates.values()):
                    raise RuntimeError(f'Parameters did not update: {updates}')
                write_json(args.output.parent/'preflight.json',dict(passed=True,calibration=calibration,
                    updates=updates,full_reload_max_abs=reload_preflight))
            samples += count
            torch.cuda.synchronize()
            row = dict(step=step,loss=sums['action_loss']+model.config.memory_state_loss_weight*sums['state_loss'],
                **sums,learning_rate=1e-5,gradient_norm=float(norm),
                cognitive_q_gradient=float(tracked['query'].grad.float().norm()),
                episode=seq['episode'],window=seq['window'],decisions=count,seen_samples=samples,
                state_known=int(seq['state_known'].sum()),state_positive=int(seq['state_targets'][seq['state_known']].sum()),
                seconds=time.monotonic()-tick,allocated_cuda_gib=torch.cuda.memory_allocated()/2**30,
                reserved_cuda_gib=torch.cuda.memory_reserved()/2**30,
                peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
            row.update({f'action_loss_{name}':float(value/max(count,1.))
                        for name,value,count in zip(parts,per_part,part_counts)})
            if writer is None:
                writer = csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);stream.flush()
            print(json.dumps(row),flush=True)
            write_json(args.output.parent/'training_status.json',dict(status='running',**row))
    if frozen_hash(model) != original_frozen:
        raise RuntimeError('Frozen backbone/expert/perception/scalar changed')
    del optimizer,params,tracked,initial
    model.memory.reset()
    # The last training window need not start an episode; use the initial
    # real-label probe for a clean two-slot reload comparison.
    destination = args.output.parent/'v2__test02__exp01__run001__step001000'
    model,processor,reloaded = save_and_reload(model,processor,destination,probe)
    report = dict(passed=True,category='exploratory_validation',steps=args.steps,
        checkpoint=str(destination),seen_samples=samples,frozen_parameters_unchanged=True,
        full_reload_max_abs=reloaded,preflight_reload_max_abs=reload_preflight,
        state_loss_weight=model.config.memory_state_loss_weight,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
        elapsed_seconds=time.monotonic()-start,closed_loop_pairs_completed=0,diagnostic_windows_completed=0)
    index = args.output.parent/'checkpoint_index.csv'
    identity = dict(model_version='v2',test_id='test02',experiment_id='exp01',run_id='run001',
        optimizer_step=1000,checkpoint_id=destination.name,base_checkpoint=str(args.checkpoint),
        code_commit=config['code_commit'],config_hash=file_hash(args.output.parent/'train_config.json'),seed=42,
        data_manifest=str(args.labels),job_id=args.output.parent.name,attempt_id=args.output.parent.name,
        resume_event='',sha256=file_hash(destination/'COMPLETE.json'),status='COMPLETE')
    with index.open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(identity));writer.writeheader();writer.writerow(identity)
    write_json(args.output,report)
    write_json(args.output.parent/'training_status.json',dict(status='complete',**report))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
