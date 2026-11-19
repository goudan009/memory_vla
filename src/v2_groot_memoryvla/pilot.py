"""Bounded real-checkpoint validation and few-batch training, without simulation.

Defaults: four chronological decisions, twelve optimizer updates per V2 arm;
frozen reused V1. --train-v1 also enables the original memory/action expert.
Outputs are training diagnostics, not task success or MAE evaluation.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from v1_groot_memoryvla.experiment.first_training import cache_sequence,slice_features,seeded
from v1_groot_memoryvla.experiment.task21_formal import supervision_count
from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from .config import V2Config
from .data import BodyMapper,slice_body
from .dataset import V2HistoryDataset
from .model import V2Model


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")


def hash_parameters(module):
    h=hashlib.sha256()
    for name,p in module.named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def context(seq,i,training=False):
    return {"mode":"train_sequence" if training else "stream","episode_keys":[seq["episode_key"]],
        "timesteps":[seq["timesteps"][i]],"sequence_start":i==0,"sequence_end":i==len(seq["timesteps"])-1}


def loss_sequence(model,seq,device,backward=False,noise=10):
    model.reset(); total=0.
    units=supervision_count(seq)
    if units<=0: raise ValueError("Sequence contains no valid action supervision")
    for i in range(len(seq["timesteps"])):
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            features=model.fuse(slice_features(seq["features"],i,i+1,device),seq["input_ids"][i:i+1].to(device),
                context(seq,i,True),slice_body(seq["body"],i,i+1,device))
            seeded(noise+i)
            loss=model.base.action_head(features,slice_features(seq["actions"],i,i+1,device))["loss"]
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite action loss")
        weight=supervision_count(seq,i,i+1)/units
        total+=float(loss.detach())*weight
        if backward: (loss*weight).backward()
    return total


def augmented_training_sequence(model,processor,dataset,index,device,seed,reference):
    processor.train()
    model.base.backbone.to(device)
    try:
        seq=cache_sequence(model.base,processor,dataset,index,1,augmentation_seed=seed)
        if seq["episode_key"]!=reference["episode_key"] or seq["frame_indices"]!=reference["frame_indices"]:
            raise ValueError("Fixed cache does not match the selected training window")
        return seq
    finally:
        model.base.backbone.cpu()
        processor.eval()


@torch.no_grad()
def influence(model,seq,device,condition="normal"):
    model.eval(); model.reset()
    body={k:v.clone() for k,v in seq["body"].items()}
    if condition=="current_body":
        # Change one measured arm configuration, keep all vision/state-action
        # expert inputs fixed to isolate the new Body path.
        node=model.mapper.graph.index["left_arm_joint2"]
        body["q"][-1,node]+=.3
    elif condition=="past_body":
        node=model.mapper.graph.index["left_arm_joint2"]
        body["q"][:-1,node]+=.3
    elif condition=="past_action":
        node=model.mapper.graph.index["left_arm_joint2"]
        body["previous_action"][:-1,node]+=.3
    elif condition in {"empty_embodied","vl_query"}: pass
    elif condition!="normal": raise ValueError(condition)
    last=len(seq["timesteps"])-1
    for i in range(last+1):
        if i==last and condition=="empty_embodied": model.extension.reset()
        hook=None
        if i==last and condition=="vl_query":
            hook=model.extension.query.body.register_forward_pre_hook(lambda module,args:(torch.zeros_like(args[0]),))
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            fused=model.fuse(slice_features(seq["features"],i,i+1,device),seq["input_ids"][i:i+1].to(device),
                context(seq,i),slice_body(body,i,i+1,device))
        if hook is not None: hook.remove()
    diag=model.extension.diagnostics[-1] if model.config.body else {}
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
        seeded(701)
        pred=model.base.action_head.get_action(fused,slice_features(seq["actions"],last,last+1,device))["action_pred"]
    return pred.float().cpu(),{k:v.detach().float().cpu() if torch.is_tensor(v) else v for k,v in diag.items()}


def difference(a,b):
    d=(a-b).float()
    return {"rms":float(d.square().mean().sqrt()),"max_abs":float(d.abs().max()),"changed":bool(torch.count_nonzero(d))}


def run(args):
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    device=torch.device(args.device)
    mapper=BodyMapper(args.urdf)
    records={}
    for variant in args.variants:
        seeded(42)
        cp=args.baseline_checkpoint if variant=="baseline" else args.checkpoint
        config=V2Config.load(args.config) if args.config else V2Config(variant=variant)
        model=V2Model.from_v1(cp,config,args.urdf).to(device).eval()
        # Full-size Eagle is used to create the real cache, then moved to CPU
        # for diagnostics to leave GPU memory for action-head backpropagation.
        processor=Gr00tN1d6Processor.from_pretrained(cp); processor.eval()
        if not records and args.sequence_cache:
            seq=torch.load(args.sequence_cache,weights_only=False,map_location="cpu")
            torch.save(seq,output/"fixed_sequence.pt")
        elif not records:
            dataset=V2HistoryDataset([args.dataset],processor,mapper=mapper,raw_dataset=args.raw_dataset,
                group_size=args.frames,history_stride=8)
            seq=cache_sequence(model.base,processor,dataset,args.group,1)
            # Baseline Eagle does not normally return pre-LLM vision features.
            # The diagnostic cache is generated with this explicit opt-in flag.
            if "visual_features" not in seq["features"]:
                model.base.backbone.return_visual_features=True
                seq=cache_sequence(model.base,processor,dataset,args.group,1)
            torch.save(seq,output/"fixed_sequence.pt")
            save_json(output/"sample.json",{"episode":seq["episode_key"],"frames":seq["frame_indices"],
                "feature_shapes":{k:list(v.shape) for k,v in seq["features"].items()},
                "body_shapes":{k:list(v.shape) for k,v in seq["body"].items()},
                "action_shapes":{k:list(v.shape) for k,v in seq["actions"].items()}})
        else:
            # Backbones must be identical for fair feature reuse. Hash frozen
            # Eagle once per arm rather than silently mixing checkpoints.
            if hash_parameters(model.base.backbone)!=records[next(iter(records))]["backbone_sha256"]:
                raise ValueError("Frozen backbones differ; regenerate the feature cache")
        record={"checkpoint":str(cp),"loading_info":model.loading_info,
            "memory_conditioning":model.base.config.memory_conditioning,
            "memory_length":model.base.config.memory_length,
            "embodied_memory_capacity":config.memory_capacity,
            "image_augmentation":args.image_augmentation,
            "backbone_sha256":hash_parameters(model.base.backbone),
            "reused_parameters":sum(p.numel() for p in model.base.parameters()),
            "new_parameters":sum(p.numel() for p in model.extension.parameters())}
        before=hash_parameters(model.base)
        model.base.backbone.cpu()
        if not args.train_v1: model.base.requires_grad_(False)
        parameters=[p for p in model.parameters() if p.requires_grad]
        initial=loss_sequence(model,seq,device)
        record["initial_loss"]=initial
        if variant.startswith("v2"):
            optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=.01)
            losses=[]
            if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
            tick=time.monotonic()
            for step in range(args.steps):
                # Eval-mode overfit uses fixed noise/dropout-free head to make
                # few-batch learnability observable. Gradients remain enabled.
                model.eval(); optimizer.zero_grad(set_to_none=True)
                loss=loss_sequence(model,seq,device,True)
                gradients={n:float(p.grad.float().norm()) if p.grad is not None else None
                           for n,p in model.extension.named_parameters()}
                if any(v is None or not np.isfinite(v) for v in gradients.values()):
                    raise RuntimeError(f"Missing/nonfinite extension gradients: {gradients}")
                if step==0:
                    save_json(output/f"{variant}_gradients.json",gradients)
                    assert gradients["body_encoder.dynamic.0.weight"]>0
                    if config.history:
                        for n in ("history_encoder.0.weight","history.q.weight","query.fuse.0.weight","memory_projection.weight"):
                            assert gradients[n]>0,n
                torch.nn.utils.clip_grad_norm_(parameters,1.)
                optimizer.step(); losses.append(loss)
                print(json.dumps({"variant":variant,"step":step+1,"loss":loss}),flush=True)
            record.update(losses=losses,final_loss=loss_sequence(model,seq,device),
                training_seconds=time.monotonic()-tick,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=="cuda" else None)
            if args.steps and record["final_loss"]>=initial: raise AssertionError("Few-batch loss did not decrease")
            # Additional short-cycle training uses varying noise and training
            # mode, rather than mistaking fixed-noise fitting for normal SGD.
            model.train()
            short=[]
            for step in range(args.short_steps):
                # Fixed eval features belong only to the overfit/influence
                # checks. Normal training re-augments and re-encodes Eagle.
                train_seq=seq
                if args.image_augmentation:
                    train_dataset=V2HistoryDataset([args.dataset],processor,mapper=mapper,
                        raw_dataset=args.raw_dataset,group_size=args.frames,history_stride=8)
                    train_seq=augmented_training_sequence(model,processor,train_dataset,args.group,
                        device,10000+step*100,seq)
                optimizer.zero_grad(set_to_none=True)
                value=loss_sequence(model,train_seq,device,True,noise=10000+step*100)
                torch.nn.utils.clip_grad_norm_(parameters,1.); optimizer.step(); short.append(value)
            record["short_train_losses"]=short
            model.eval()
            normal,normal_diag=influence(model,seq,device)
            arrays={"normal":normal.numpy()}
            shifts={}
            for condition in (["current_body"] if variant=="v2a" else
                    ["current_body","past_body","past_action","empty_embodied"]+(["vl_query"] if variant=="v2c" else [])):
                pred,diag=influence(model,seq,device,condition)
                arrays[condition]=pred.numpy()
                # Compare only the meaningful first 8 executed / 23 B1K dims.
                row={"action":difference(normal[:,:8,:23],pred[:,:8,:23])}
                for name in ("query","attention"):
                    if name in diag and diag[name].shape==normal_diag[name].shape:
                        row[name]=difference(normal_diag[name],diag[name])
                shifts[condition]=row
            record["influence"]=shifts
            assert shifts["current_body"]["action"]["changed"],"Current Body does not influence actions"
            if config.history:
                for condition in ("past_body","past_action","empty_embodied"):
                    assert shifts[condition]["action"]["changed"],condition
                if variant=="v2c":
                    assert shifts["current_body"]["query"]["changed"]
                    assert shifts["current_body"]["attention"]["changed"]
                    assert shifts["vl_query"]["action"]["changed"],"Query path does not influence actions"
                else:
                    assert not shifts["current_body"]["query"]["changed"]
                    assert not shifts["current_body"]["attention"]["changed"]
            np.savez_compressed(output/f"{variant}_actions.npz",**arrays)
            if not args.train_v1:
                assert before==hash_parameters(model.base),"Reused frozen V1 weights changed"
            if not args.train_v1 and not model.loading_info["missing_keys"]:
                model.save_extension(output/f"{variant}_checkpoint")
                # Test strict extension restoration in the SAME real action
                # expert before releasing it. Separate loader test below.
                with torch.no_grad():
                    for p in model.extension.parameters(): p.zero_()
                model.extension.load_state_dict(torch.load(output/f"{variant}_checkpoint/extension.pt",weights_only=True),strict=True)
                reload_pred,_=influence(model,seq,device)
                record["checkpoint_actions_exact"]=torch.equal(normal,reload_pred)
                assert record["checkpoint_actions_exact"]
            else: model.save(output/f"{variant}_checkpoint",processor)
            del optimizer
        else:
            with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                pred=model.forward_cached(slice_features(seq["features"],0,1,device),
                    seq["input_ids"][:1].to(device),context(seq,0),slice_features(seq["actions"],0,1,device),
                    slice_body(seq["body"],0,1,device),predict=True)["action_pred"]
                record["action_shape"]=list(pred.shape)
        records[variant]=record
        save_json(output/"results.json",records)
        del model,processor
        gc.collect()
        if device.type=="cuda": torch.cuda.empty_cache()
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True,type=Path)
    parser.add_argument("--config",type=Path,help="One ablation JSON; replaces --variants")
    parser.add_argument("--sequence-cache",type=Path,help="Trusted cache generated by this pilot, same frozen Eagle")
    parser.add_argument("--baseline-checkpoint",type=Path,default=Path("checkpoints/groot_n1d6_behavior1k"))
    parser.add_argument("--dataset",type=Path,default=Path("assets/datasets/b1k2025_groot_task21_diagnostic"))
    parser.add_argument("--raw-dataset",type=Path)
    parser.add_argument("--urdf",type=Path,default=Path("assets/simulation/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf"))
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--variants",nargs="+",default=["baseline","v1","v2a","v2b","v2c"],choices=["baseline","v1","v2a","v2b","v2c"])
    parser.add_argument("--steps",type=int,default=12)
    parser.add_argument("--short-steps",type=int,default=2)
    parser.add_argument("--frames",type=int,default=4)
    parser.add_argument("--group",type=int,default=0)
    parser.add_argument("--lr",type=float,default=.003)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--train-v1",action="store_true")
    parser.add_argument("--image-augmentation",action=argparse.BooleanOptionalAction,default=True,
                        help="Re-encode augmented observations per short training step; fitting stays deterministic")
    args=parser.parse_args()
    if args.config: args.variants=[V2Config.load(args.config).variant]
    if not 3<=args.frames<=32 or args.steps>100 or args.steps<0 or not 0<=args.short_steps<=10:
        parser.error("Pilot requires 3..32 frames, 0..100 fitting steps and 0..10 short steps")
    run(args)


if __name__=="__main__": main()
