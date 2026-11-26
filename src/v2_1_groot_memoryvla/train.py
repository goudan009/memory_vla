"""Single-GPU chronological training. Explicit --execute plus preflight required.

Stage1 / continued: original action expert only; memory: frozen entire base.
No automatic launch, no implicit V1 memory fallback. Resume restores the exact
optimizer, schedule cursor, RNG and detached runtime; real-server validation is required.
"""
import argparse,copy,json,hashlib,random
from pathlib import Path
import numpy as np
import torch
from .config import Config,CORE_STATES
from .runtime import source_digest,export_runtime,import_runtime,rng_state,restore_rng
from .data.labels import Labels,state_terms
from .model.integration import Integrated
from .data.windows import Windows,make_split,schedule,fit_statistics,TAG
from v1_groot_memoryvla.experiment.first_training import seeded
from v1_groot_memoryvla.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor

def jsonwrite(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n")
def hash_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for x in iter(lambda:f.read(1024*1024),b""):h.update(x)
    return h.hexdigest()
def base_hash(base):
    h=hashlib.sha256()
    for name,p in base.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def load_base(path,stage):
    saved=json.loads((Path(path)/"config.json").read_text())
    if saved.get("memory_enabled") or saved.get("finetuning_lora"):raise ValueError("expected original no-memory full checkpoint")
    base,info=Gr00tN1d6.from_pretrained(path,memory_enabled=False,tune_llm=False,tune_visual=False,
        tune_top_llm_layers=0,backbone_trainable_params_fp32=False,torch_dtype=torch.float32,output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):
        raise ValueError("base checkpoint load mismatch: "+str(info))
    base.backbone.requires_grad_(False).bfloat16().eval()
    tune=stage!="memory"
    base.action_head.set_trainable_parameters(tune,tune,tune)
    base.action_head.float();base.to("cuda")
    if stage=="memory":base.requires_grad_(False).eval()
    else:base.action_head.train()
    return base
def initialize(args,resume=None):
    seeded(args.seed)
    split=json.loads((args.output/'data_split.json').read_text()) if resume else (make_split(args.dataset,args.seed) if args.split is None else json.loads(args.split.read_text()))
    if split.get("task")!=40 or len(split["train"])!=180 or len(split["validation"])!=20 or len(set(split["train"]+split["validation"]))!=200:
        raise ValueError("invalid fixed Task40 split")
    processor=Gr00tN1d6Processor.from_pretrained(resume or args.checkpoint)
    if not processor.use_relative_action or processor.use_percentiles:raise ValueError("processor convention mismatch")
    mods=processor.modality_configs[TAG]
    if list(mods["action"].delta_indices)!=list(range(32)) or processor.max_action_horizon!=50 or processor.max_action_dim!=128:
        raise ValueError("processor action contract mismatch")
    train=Windows(args.dataset,processor,split["train"],seed=args.seed)
    if args.stage=="stage1" and resume is None:
        stats,lo,hi=fit_statistics(train,processor);processor.set_statistics(stats,override=True)
    else:
        ref=resume if args.stage=="stage1" else args.checkpoint
        meta=json.loads((ref/"v2_1_stage1.json").read_text())
        if meta["split"]!=split or (args.stage!="stage1" and meta["step"]!=6000):
            raise ValueError("expected fixed split and common Stage1 step6000")
        if meta["statistics_sha256"]!=hash_file(ref/"statistics.json"):raise ValueError("statistics mismatch")
        if resume is not None and hash_file(resume/"statistics.json")!=meta["statistics_sha256"]:
            raise ValueError("resumed processor statistics mismatch")
        lo,hi=meta["raw_lo"],meta["raw_hi"]
    processor.train();validation_processor=copy.deepcopy(processor);validation_processor.eval()
    val=Windows(args.dataset,validation_processor,split["validation"],seed=args.seed)
    labels=Labels(args.labels,args.states)
    lengths={e:n for (_,e),n in train.episode_lengths.items()}
    if set(labels.data)!=set(split["train"]+split["validation"]) or any(len(labels.data[e][0])!=lengths[e] for e in labels.data):
        raise ValueError("label/episode length mismatch")
    pos=labels.class_weights(split["train"],weak_weight=args.weak_weight).cuda()
    # This also verifies that the selected fields have both classes in held-out observations.
    labels.class_weights(split["validation"],weak_weight=args.weak_weight)
    base=load_base(resume if resume and args.stage!="memory" else args.checkpoint,args.stage)
    model=None
    if args.stage=="memory":
        if resume:
            model=Integrated.load_sidecar(base,resume,"cuda").train()
            args.state_weight=model.cfg.state_weight
        else:
            model=Integrated(base,Config(prior=args.prior,fusion_layers=args.fusion_layers,states=tuple(args.states),weak_weight=args.weak_weight,
                                         state_weight=args.state_weight),lo,hi).cuda()
    return split,processor,train,val,labels,pos,base,model,lo,hi

def groups(ds,pair,seed,labels):
    out=[]
    for k in pair:
        seeded(seed+k);g=ds.get_group(k,sampling_seed=seed+k)
        g["target"],g["quality"]=labels.get(g["episode"],g["frame_indices"])
        out.append(g)
    return out

def grad_add(total,grads):
    for i,g in enumerate(grads):
        if g is not None:total[i]=g.detach().float() if total[i] is None else total[i]+g.detach().float()
def norm(grads):
    return float(torch.stack([g.square().sum() for g in grads if g is not None]).sum().sqrt()) if any(g is not None for g in grads) else 0.

def run_update(ds,pair,seed,base,model,labels,pos,args,optimizer=None,calibrate=False):
    gs=groups(ds,pair,seed,labels)
    units=sum(float(np.asarray(s["action_mask"]).sum()) for g in gs for s in g["steps"])
    counts=sum((g["quality"]>0).sum(0) for g in gs).cuda()
    if units<=0:raise ValueError("no supervised actions")
    if optimizer is not None:optimizer.zero_grad(set_to_none=True)
    fm_total=0.;state_total=0.
    fgr=sgr=None
    if calibrate:
        shared=list(model.memory.enhance.parameters())
        fgr=[None]*len(shared);sgr=[None]*len(shared)
    stats={n:[0,0,0] for n in args.states} # TP, FP, FN
    for g in gs:
        for a in range(0,len(g["steps"]),args.microbatch):
            b=min(a+args.microbatch,len(g["steps"]))
            inputs=ds.processor.collator(g["steps"][a:b])["inputs"]
            bk,act=base.prepare_input(inputs)
            seeded(seed+g["group_id"]*100+a)
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                feat=base.backbone(bk)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                if model is None:
                    output=base.action_head(feat,act)
                    fm=output["action_loss"].float().sum()/units
                    state=fm.new_zeros(())
                else:
                    context=dict(mode="train_sequence",episode_keys=[g["episode_key"]]*(b-a),
                        timesteps=g["timesteps"][a:b],sequence_start=g["episode_start"] and a==0,
                        sequence_end=g["episode_end"] and b==len(g["steps"]))
                    cache=model.prepare_cache(feat,bk["input_ids"],torch.from_numpy(g["raw_q"][a:b]).cuda(),context)
                    # Reset action RNG after all memory computation.
                    seeded(seed+g["group_id"]*100+a)
                    output=model.loss_from_features(feat,act,cache)
                    fm=output["fm_numerator"]/units
                    target=g["target"][a:b].cuda();quality=g["quality"][a:b].cuda()
                    numerator,_=state_terms(cache.logits,target,quality,pos,args.weak_weight)
                    state=(numerator/counts.clamp_min(1)).sum()/(counts>0).sum().clamp_min(1)
                    with torch.no_grad():
                        pred=cache.logits>=0;known=quality>0;truth=target==1
                        for i,n in enumerate(args.states):
                            stats[n][0]+=int((pred[:,i]&truth[:,i]&known[:,i]).sum())
                            stats[n][1]+=int((pred[:,i]&~truth[:,i]&known[:,i]).sum())
                            stats[n][2]+=int((~pred[:,i]&truth[:,i]&known[:,i]).sum())
            loss=fm+args.state_weight*state
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite training loss")
            if calibrate:
                if fm.requires_grad:grad_add(fgr,torch.autograd.grad(fm,shared,retain_graph=True,allow_unused=True))
                if state.requires_grad:grad_add(sgr,torch.autograd.grad(state,shared,retain_graph=True,allow_unused=True))
            if optimizer is not None and loss.requires_grad:loss.backward()
            fm_total+=float(fm.detach());state_total+=float(state.detach())
    if optimizer is not None:
        params=[p for g in optimizer.param_groups for p in g["params"]]
        gn=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        optimizer.step()
    else:gn=0.
    module_grads={}
    if optimizer is not None and args.preflight_only and model is not None:
        for name,module in dict(visual=model.memory.visual,enhance=model.memory.enhance,state_head=model.memory.state_head,
             body=model.memory.robot.body,whole=model.memory.robot.whole,identity=model.memory.robot.id_embed,
             cognition=model.memory.cog_read,cognitive_gate=model.memory.cog_gate,cognitive_time=model.memory.cog_time,
             task_c=model.memory.task_c,task_v=model.memory.task_v,query=model.memory.query,key=model.memory.key,value=model.memory.value,
             action=model.refinement.action,fusion=model.refinement.fusion,global_blocks=model.refinement.global_blocks,output=model.refinement.output,
             **({"local":model.refinement.local} if model.refinement.local is not None else {})).items():
            module_grads[name]=norm([p.grad for p in module.parameters()])
    return dict(module_grads=module_grads,fm=fm_total,state=state_total,grad_norm=gn,state_counts=stats,
                fm_shared_grad=norm(fgr) if calibrate else None,state_shared_grad=norm(sgr) if calibrate else None,
                action_elements=units,state_elements=counts.cpu().tolist())

def optimizer_for(base,model,args):
    params=[p for p in (model or base).parameters() if p.requires_grad]
    if not params or any(p.dtype!=torch.float32 for p in params):raise ValueError("trainable FP32 parameters required")
    return torch.optim.AdamW(params,lr=args.lr,betas=(.9,.999),eps=1e-8,weight_decay=.01)
def stream(ds,seed):
    epoch=0
    while True:
        for pair in schedule(ds,seed,epoch):yield epoch,pair
        epoch+=1
def calibration(ds,base,model,labels,pos,args):
    # Save CPU master weights, isolate preflight from formal training entirely.
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith("base.")}
    def once():
        model.reset();opt=optimizer_for(base,model,args);seq=stream(ds,args.seed);rows=[]
        for step in range(20):
            _,pair=next(seq);rows.append(run_update(ds,pair,args.seed+step*100000,base,model,labels,pos,args,opt,True))
        model.load_state_dict(initial,strict=False);model.reset();del opt;seeded(args.seed)
        f=float(np.median([x["fm_shared_grad"] for x in rows]));s=float(np.median([x["state_shared_grad"] for x in rows]))
        return f,s,rows
    f,s,rows=once()
    if min(f,s)<=0 or not np.isfinite([f,s]).all():raise RuntimeError("shared gradients not established")
    ratio=args.state_weight*s/f
    if not .1<=ratio<=10:
        args.state_weight=float(np.clip(.5*f/s,.001,.1))
        f,s,rows2=once()
        ratio=args.state_weight*s/f
        if not .1<=ratio<=10:raise RuntimeError("state/FM calibration failed")
    else:rows2=None
    return dict(fm=f,state=s,ratio=ratio,lambda_s=args.state_weight,initial=rows,recheck=rows2)

def validate(ds,base,model,labels,pos,args,step):
    prior=base.action_head.training;old_banks=model.memory._banks if model is not None else None
    if model is not None:model.memory._banks=None;model.eval()
    base.eval()
    try:
        total=0.;units=0.;stats={n:[0,0,0] for n in args.states}
        with torch.no_grad():
            for i,pair in enumerate(schedule(ds,args.seed,0)):
                r=run_update(ds,pair,100000000+100000*i,base,model,labels,pos,args)
                total+=r["fm"]*r["action_elements"];units+=r["action_elements"]
                for n in stats:stats[n]=[x+y for x,y in zip(stats[n],r["state_counts"][n])]
        return {"step":step,"fm":total/units,"states":{n:dict(tp=t[0],fp=t[1],fn=t[2],f1=2*t[0]/max(2*t[0]+t[1]+t[2],1)) for n,t in stats.items()}}
    finally:
        if model is not None:model.memory._banks=old_banks;model.train()
        else:base.action_head.train(prior)
def identity(args):
    return {k:getattr(args,k) for k in ("stage","prior","fusion_layers","states","seed","microbatch","lr","weak_weight")} | {
        "source_digest":source_digest(),"label_sha256":hash_file(args.labels),
        "initial_checkpoint":str(args.checkpoint.resolve()),"dataset":str(args.dataset.resolve())}

def save(args,base,model,processor,split,lo,hi,step,initial_hash,opt):
    final=args.output/"checkpoints"/f"step-{step:06d}"
    dest=final.with_name(final.name+".partial")
    if final.exists():raise FileExistsError(final)
    dest.mkdir(parents=True,exist_ok=False)
    if model is None:
        base.save_pretrained(dest,max_shard_size="5GB");processor.save_pretrained(dest)
    else:
        # Keep the exact immutable Stage1 checkpoint files, no dtype conversion on save.
        import os,shutil
        for p in args.checkpoint.iterdir():
            if not p.is_file() or p.name in ('v2_1_run.json','v2_1.json','v2_1.pt','training_state.pt','COMPLETE.json'):continue
            if p.suffix in ('.safetensors','.bin'):
                try:os.link(p,dest/p.name)
                except OSError:shutil.copy2(p,dest/p.name)
            else:shutil.copy2(p,dest/p.name)
        # Avoid overwriting linked files; Stage2 processor is identical to Stage1.
        from dataclasses import replace
        model.cfg=replace(model.cfg,state_weight=args.state_weight)
        model.save_sidecar(dest)
    if args.stage=="stage1":
        jsonwrite(dest/"v2_1_stage1.json",dict(step=step,split=split,raw_lo=np.asarray(lo).tolist(),raw_hi=np.asarray(hi).tolist(),
                                             statistics_sha256=hash_file(dest/"statistics.json")))
    jsonwrite(dest/"v2_1_run.json",dict(stage=args.stage,step=step,base_initial_hash=initial_hash,lambda_s=args.state_weight,
                                       label_sha256=hash_file(args.labels),prior=args.prior,states=args.states,seed=args.seed))
    torch.save({"identity":identity(args),"step":step,"optimizer":opt.state_dict(),"runtime":export_runtime(model),
                "rng":rng_state(),"lambda_s":args.state_weight,"base_initial_hash":initial_hash},dest/"training_state.pt")
    jsonwrite(dest/"COMPLETE.json",{"step":step,"source_digest":source_digest()})
    dest.rename(final)

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--stage",choices=["stage1","memory","continued"],required=True)
    p.add_argument("--dataset",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--labels",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",type=Path);p.add_argument("--calibration-reference",type=Path,help="A calibration.json for B/C/D");p.add_argument("--prior",choices=["topology","fk","none"],default="topology")
    p.add_argument("--fusion-layers",type=int,default=0);p.add_argument("--states",nargs="+",default=list(CORE_STATES))
    p.add_argument("--steps",type=int);p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--microbatch",type=int,choices=[1,2],default=2)
    p.add_argument("--weak-weight",type=float,default=.5);p.add_argument("--state-weight",type=float,default=.01)
    p.add_argument("--resume",type=Path,help="Completed checkpoint from this same output run (trusted local file)")
    p.add_argument("--execute",action="store_true");p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--accept-preflight",type=Path,help="Required signed-off local preflight evidence for formal run")
    args=p.parse_args()
    args.steps=(6000 if args.stage=="stage1" else 3000) if args.steps is None else args.steps
    if args.steps<1 or not np.isfinite(args.lr) or args.lr<=0:raise ValueError('invalid training budget')
    Config(prior=args.prior,fusion_layers=args.fusion_layers,states=tuple(args.states),weak_weight=args.weak_weight,state_weight=args.state_weight)
    spec={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    print(json.dumps(spec,indent=2))
    if not args.execute:return
    if args.stage=="stage1" and (args.checkpoint/"v2_1_stage1.json").exists():raise ValueError("Stage1 must start from generic GR00T, not an existing Stage1 result")
    if args.resume and args.preflight_only:raise ValueError('resume is not a disposable preflight')
    if not args.preflight_only and not args.resume:
        if args.accept_preflight is None or not args.accept_preflight.exists():raise SystemExit("formal run requires local preflight evidence")
        gate=json.loads(args.accept_preflight.read_text())
        if not gate.get("passed") or gate.get("stage")!=args.stage or gate.get("prior")!=args.prior or gate.get("label_sha256")!=hash_file(args.labels) or gate.get("checkpoint")!=str(args.checkpoint.resolve()) or gate.get("states")!=args.states or gate.get("microbatch")!=args.microbatch or gate.get("fusion_layers")!=args.fusion_layers or gate.get('source_digest')!=source_digest():
            raise SystemExit("preflight evidence mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise SystemExit("select one GPU explicitly")
    saved=None
    if args.resume:
        if args.resume.resolve().parent!=(args.output/"checkpoints").resolve() or not (args.resume/"COMPLETE.json").is_file():
            raise ValueError("resume must be a completed checkpoint of this run")
        # Local, user-owned training state includes Python/NumPy RNG objects.
        saved=torch.load(args.resume/"training_state.pt",map_location="cpu",weights_only=False)
        if saved["identity"]!=identity(args) or not 0<=saved["step"]<args.steps:
            raise ValueError("resume configuration/source or budget mismatch")
        args.state_weight=float(saved["lambda_s"])
    else:args.output.mkdir(parents=True,exist_ok=False)
    split,proc,tr,val,labels,pos,base,model,lo,hi=initialize(args,args.resume)
    if saved is None:
        jsonwrite(args.output/"config.json",spec);jsonwrite(args.output/"data_split.json",split)
    initial_hash=base_hash(base) if model is not None else None
    if saved is not None and model is not None and initial_hash!=saved["base_initial_hash"]:
        raise ValueError("base checkpoint changed during resume")
    if model is not None and saved is None:
        if args.prior!='topology':
            if args.calibration_reference is None:raise ValueError('B/C/D require A calibration reference')
            ref=json.loads(args.calibration_reference.read_text())
            if ref.get('prior')!='topology' or ref.get('label_sha256')!=hash_file(args.labels) or ref.get('states')!=args.states or ref.get('split')!=split:
                raise ValueError('A calibration reference is incompatible')
            args.state_weight=float(ref['lambda_s'])
            if not .001<=args.state_weight<=.1:raise ValueError('invalid reference lambda_s')
            result=dict(lambda_s=args.state_weight,reference=str(args.calibration_reference),ratio=ref['ratio'])
        else:result=calibration(tr,base,model,labels,pos,args)
        result.update(prior=args.prior,label_sha256=hash_file(args.labels),states=args.states,split=split)
        jsonwrite(args.output/"calibration.json",result)
    if args.preflight_only:
        from .preflight import equivalence
        initial_equiv=equivalence(base,model,tr,args.output/'initial',args.seed)
        # Disposable actual optimization; separate run, never promoted to checkpoint.
        before={k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad} if model is not None else {}
        opt=optimizer_for(base,model,args)
        seq=stream(tr,args.seed);rows=[]
        for i in range(3):
            _,pair=next(seq);rows.append(run_update(tr,pair,args.seed+i*100000,base,model,labels,pos,args,opt))
        unchanged=(base_hash(base)==initial_hash) if model is not None else None
        updated={k:bool(not torch.equal(before[k],p.detach().cpu())) for k,p in model.named_parameters() if k in before} if model is not None else {}
        module_names=set().union(*(r['module_grads'] for r in rows))
        missing_gradient=[n for n in sorted(module_names) if not any(np.isfinite(r['module_grads'].get(n,0.)) and r['module_grads'].get(n,0.)>0 for r in rows[1:])]
        trained_equiv=equivalence(base,model,tr,args.output/'trained',args.seed)
        jsonwrite(args.output/"preflight.json",dict(passed=all(np.isfinite(r["grad_norm"]) and r["grad_norm"]>0 for r in rows) and unchanged is not False and not missing_gradient and (bool(any(updated.values())) if model is not None else True),
                   initial_equivalence=initial_equiv,trained_equivalence=trained_equiv,updated_parameters=updated,missing_gradient=missing_gradient,
                   stage=args.stage,prior=args.prior,rows=rows,base_unchanged=unchanged,
                   label_sha256=hash_file(args.labels),checkpoint=str(args.checkpoint.resolve()),states=args.states,
                   microbatch=args.microbatch,fusion_layers=args.fusion_layers,source_digest=source_digest()))
        return
    opt=optimizer_for(base,model,args);seq=stream(tr,args.seed)
    if saved is None:
        start=0
        save(args,base,model,proc,split,lo,hi,0,initial_hash,opt)
        jsonwrite(args.output/"validation-000000.json",validate(val,base,model,labels,pos,args,0))
    else:
        start=int(saved["step"])
        opt.load_state_dict(saved["optimizer"]);import_runtime(model,saved["runtime"]);restore_rng(saved["rng"])
        for _ in range(start):next(seq)
    for step in range(start+1,args.steps+1):
        epoch,pair=next(seq);r=run_update(tr,pair,args.seed+(step-1)*100000,base,model,labels,pos,args,opt)
        r.update(step=step,epoch=epoch,windows=pair)
        with (args.output/"metrics.jsonl").open("a") as f:f.write(json.dumps(r)+"\n")
        print(json.dumps(r),flush=True)
        if step%1000==0 or step==args.steps:
            if model is not None and base_hash(base)!=initial_hash:raise RuntimeError("frozen base changed")
            save(args,base,model,proc,split,lo,hi,step,initial_hash,opt)
        if step%(3000 if args.stage=="stage1" else 1000)==0 or step==args.steps:
            jsonwrite(args.output/f"validation-{step:06d}.json",validate(val,base,model,labels,pos,args,step))
if __name__=="__main__":main()
