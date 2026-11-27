"""Chronological single/multi-GPU training. Explicit --execute plus preflight required.

Stage1 / continued: original action expert only; memory: frozen entire base.
No automatic launch, no implicit V1 memory fallback. Resume restores the exact
optimizer, schedule cursor, RNG and detached runtime; real-server validation is required.
"""
import argparse,copy,json,hashlib,random,time,signal
from pathlib import Path
import numpy as np
import torch
from .config import Config,CORE_STATES
from .runtime import source_digest,export_runtime,import_runtime,rng_state,restore_rng
from .data.labels import Labels,state_terms
from .model.integration import Integrated
from .data.windows import Windows,make_split,schedule,fit_statistics,TAG
from v1_groot_memoryvla.experiment.first_training import seeded
from .loading import load_base
from . import distributed as parallel
from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor

_stop_requested=False
def request_stop(*_):
    global _stop_requested
    _stop_requested=True

def stopping(args,step):
    control=args.output/"stop.json"
    target=json.loads(control.read_text()).get("stop_after_step",step) if control.exists() else None
    return _stop_requested or (target is not None and step>=int(target))

def status(args,state,step,**extra):
    jsonwrite(args.output/"status.json",dict(status=state,step=step,total_steps=args.steps,
              training_complete=state=="completed",world_size=parallel.world_size(),**extra))

def jsonwrite(p,x):
    if parallel.rank()!=0:return
    p.parent.mkdir(parents=True,exist_ok=True)
    pending=p.with_name(p.name+".pending")
    pending.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n")
    pending.replace(p)
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
def tensor_tree_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str((x.dtype,tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()
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
        complete=json.loads((ref/"COMPLETE.json").read_text())
        if complete["step"]!=meta["step"]:raise ValueError("incomplete Stage1 checkpoint")
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
        seeded(seed+k)
        transform=ds.processor.train_image_transform
        if hasattr(transform,"set_random_seed"):transform.set_random_seed(seed+k)
        g=ds.get_group(k,sampling_seed=seed+k)
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
    ordinal=0
    input_hashes=[];feature_hashes=[]
    if calibrate:
        shared=list(model.memory.enhance.parameters())
        fgr=[None]*len(shared);sgr=[None]*len(shared)
    stats={n:[0,0,0] for n in args.states} # TP, FP, FN
    for g in gs:
        for a in range(0,len(g["steps"]),args.microbatch):
            b=min(a+args.microbatch,len(g["steps"]))
            owned=ordinal%parallel.world_size()==parallel.rank();ordinal+=1
            if model is None and not owned:continue
            inputs=ds.processor.collator(g["steps"][a:b])["inputs"]
            bk,act=base.prepare_input(inputs)
            if args.preflight_only:input_hashes.append(tensor_tree_hash((bk,act)))
            seeded(seed+g["group_id"]*100+a)
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                feat=base.backbone(bk)
            if args.preflight_only:feature_hashes.append(tensor_tree_hash(feat))
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
                    # All replicas advance identical detached memory; only the
                    # owner contributes this decision's gradients and metrics.
                    if not owned:continue
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
        parallel.sum_gradients(params)
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
    totals=parallel.sum_tensor(torch.tensor([fm_total,state_total],device=pos.device,dtype=torch.float64)).tolist()
    flat=parallel.sum_tensor(torch.tensor([stats[n] for n in args.states],device=pos.device,dtype=torch.int64)).cpu().tolist()
    stats=dict(zip(args.states,flat))
    if calibrate:
        for grads in (fgr,sgr):
            for i,p in enumerate(shared):
                if grads[i] is None:grads[i]=torch.zeros_like(p,dtype=torch.float32)
                parallel.sum_tensor(grads[i])
    return dict(input_hashes=input_hashes,feature_hashes=feature_hashes,module_grads=module_grads,fm=totals[0],state=totals[1],grad_norm=gn,state_counts=stats,
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
    diagnostic=dict(passed=False,passes=[])
    def record():jsonwrite(args.output/"calibration_diagnostics.json",diagnostic)
    def once():
        model.reset();opt=optimizer_for(base,model,args);seq=stream(ds,args.seed);rows=[]
        entry=dict(lambda_s=args.state_weight,steps=rows)
        diagnostic["passes"].append(entry);record()
        for step in range(20):
            _,pair=next(seq);rows.append(run_update(ds,pair,args.seed+step*100000,base,model,labels,pos,args,opt,True))
            record()
        model.load_state_dict(initial,strict=False);model.reset();del opt;seeded(args.seed)
        f=float(np.median([x["fm_shared_grad"] for x in rows]));s=float(np.median([x["state_shared_grad"] for x in rows]))
        entry.update(fm=f,state=s,ratio=args.state_weight*s/f if f>0 else None);record()
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
    diagnostic["passed"]=True;record()
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
        "requested_state_weight":getattr(args,"requested_state_weight",args.state_weight),
        "world_size":parallel.world_size(),
        "calibration_reference_sha256":hash_file(args.calibration_reference) if args.calibration_reference is not None else None,
        "source_digest":source_digest(),"label_sha256":hash_file(args.labels),
        "initial_checkpoint":str(args.checkpoint.resolve()),"dataset":str(args.dataset.resolve())}

def verify_calibrated_weight(gate,args):
    if gate.get("lambda_s")!=args.state_weight:
        raise ValueError("preflight calibrated state weight mismatch")

def save(args,base,model,processor,split,lo,hi,step,initial_hash,opt):
    ranks=parallel.gather({"runtime":export_runtime(model),"rng":rng_state()})
    if parallel.rank()!=0:
        parallel.barrier();return
    final=args.output/"checkpoints"/f"{args.checkpoint_prefix}__step{step:06d}"
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
                "rng":rng_state(),"ranks":ranks,"world_size":parallel.world_size(),"lambda_s":args.state_weight,"base_initial_hash":initial_hash},dest/"training_state.pt")
    jsonwrite(dest/"COMPLETE.json",{"step":step,"world_size":parallel.world_size(),"source_digest":source_digest()})
    dest.rename(final)
    jsonwrite(args.output/"checkpoints/latest.json",dict(step=step,path=str(final),world_size=parallel.world_size()))
    parallel.barrier()

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
    p.add_argument("--checkpoint-prefix",help="Explicit V2_1 run ID prefix supplied by the workbench")
    args=p.parse_args()
    if args.execute and not args.checkpoint_prefix:
        raise SystemExit("--execute requires an explicit --checkpoint-prefix; select the experiment identity through the workbench")
    # Preserve the requested configuration while calibration/resume changes the effective loss.
    args.requested_state_weight=args.state_weight
    args.steps=(6000 if args.stage=="stage1" else 3000) if args.steps is None else args.steps
    if args.steps<1 or not np.isfinite(args.lr) or args.lr<=0:raise ValueError('invalid training budget')
    Config(prior=args.prior,fusion_layers=args.fusion_layers,states=tuple(args.states),weak_weight=args.weak_weight,state_weight=args.state_weight)
    spec={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    print(json.dumps(spec,indent=2))
    if not args.execute:return
    parallel.setup()
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    if args.stage=="stage1" and (args.checkpoint/"v2_1_stage1.json").exists():raise ValueError("Stage1 must start from generic GR00T, not an existing Stage1 result")
    if args.resume and args.preflight_only:raise ValueError('resume is not a disposable preflight')
    if not args.preflight_only and not args.resume:
        if args.accept_preflight is None or not args.accept_preflight.exists():raise SystemExit("formal run requires local preflight evidence")
        gate=json.loads(args.accept_preflight.read_text())
        if not gate.get("passed") or gate.get("identity")!=identity(args):
            raise SystemExit("preflight evidence mismatch")
    if not torch.cuda.is_available():raise SystemExit("CUDA required")
    if parallel.world_size()==1 and torch.cuda.device_count()!=1:raise SystemExit("select one GPU or launch torchrun")
    saved=None
    if args.resume:
        if args.resume.resolve().parent!=(args.output/"checkpoints").resolve() or not (args.resume/"COMPLETE.json").is_file():
            raise ValueError("resume must be a completed checkpoint of this run")
        # Local, user-owned training state includes Python/NumPy RNG objects.
        saved=torch.load(args.resume/"training_state.pt",map_location="cpu",weights_only=False)
        if saved["identity"]!=identity(args) or not 0<=saved["step"]<args.steps:
            raise ValueError("resume configuration/source or budget mismatch")
        args.state_weight=float(saved["lambda_s"])
    elif parallel.rank()==0:args.output.mkdir(parents=True,exist_ok=False)
    parallel.barrier()
    status(args,"initializing",0)
    split,proc,tr,val,labels,pos,base,model,lo,hi=initialize(args,args.resume)
    if saved is not None:
        verify_calibrated_weight(saved,args)
    if saved is None:
        jsonwrite(args.output/"config.json",spec);jsonwrite(args.output/"data_split.json",split)
    initial_hash=saved["base_initial_hash"] if saved is not None and model is None else base_hash(base)
    if not args.preflight_only and saved is None and (gate["split"]!=split or gate["base_initial_hash"]!=initial_hash):
        raise ValueError("preflight split or initial weights changed")
    if saved is not None and model is not None and initial_hash!=saved["base_initial_hash"]:
        raise ValueError("base checkpoint changed during resume")
    if model is not None and saved is None:
        if args.prior!='topology':
            if args.calibration_reference is None:raise ValueError('B/C/D require A calibration reference')
            ref=json.loads(args.calibration_reference.read_text())
            if ref.get('prior')!='topology' or ref.get('label_sha256')!=hash_file(args.labels) or ref.get('states')!=args.states or ref.get('split')!=split or ref.get('base_initial_hash')!=initial_hash or ref.get('weak_weight')!=args.weak_weight:
                raise ValueError('A calibration reference is incompatible')
            args.state_weight=float(ref['lambda_s'])
            if not .001<=args.state_weight<=.1:raise ValueError('invalid reference lambda_s')
            result=dict(lambda_s=args.state_weight,reference=str(args.calibration_reference),ratio=ref['ratio'])
        else:result=calibration(tr,base,model,labels,pos,args)
        result.update(prior=args.prior,label_sha256=hash_file(args.labels),states=args.states,split=split,base_initial_hash=initial_hash,weak_weight=args.weak_weight)
        jsonwrite(args.output/"calibration.json",result)
    if saved is None and not args.preflight_only:
        verify_calibrated_weight(gate,args)
    if args.preflight_only:
        from .preflight import equivalence
        initial_equiv=equivalence(base,model,tr,args.output/f'rank-{parallel.rank()}'/'initial',args.seed)
        trainable_initial=tensor_tree_hash({n:p for n,p in (model or base).named_parameters() if p.requires_grad})
        backbone_initial=base_hash(base.backbone)
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
        trained_equiv=equivalence(base,model,tr,args.output/f'rank-{parallel.rank()}'/'trained',args.seed)
        trainable_updated=trainable_initial!=tensor_tree_hash({n:p for n,p in (model or base).named_parameters() if p.requires_grad})
        backbone_unchanged=backbone_initial==base_hash(base.backbone)
        # Save a real full-HF checkpoint + optimizer + every rank's runtime/RNG,
        # then compare the next actual update with a complete independent reload.
        save(args,base,model,proc,split,lo,hi,3,initial_hash,opt)
        checkpoint=args.output/'checkpoints'/f'{args.checkpoint_prefix}__step000003'
        reloaded_state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
        full_hash=base_hash(base)
        saved_parameter_hashes={n:tensor_tree_hash(p) for n,p in (model or base).state_dict().items()}
        _,next_pair=next(seq)
        expected_row=run_update(tr,next_pair,args.seed+300000,base,model,labels,pos,args,opt)
        expected_weights=tensor_tree_hash((model or base).state_dict())
        expected_optimizer=tensor_tree_hash(opt.state_dict())
        expected_runtime=tensor_tree_hash(export_runtime(model))
        expected_parameter_hashes={n:tensor_tree_hash(p) for n,p in (model or base).state_dict().items()}
        # Release the first model before rebuilding to keep the diagnostic's
        # peak memory bounded by the normal training footprint.
        del before,opt,model,base
        import gc
        gc.collect();torch.cuda.empty_cache()
        _,proc,tr,val,labels,pos,base,model,lo,hi=initialize(args,checkpoint)
        full_reload=base_hash(base)==full_hash
        reload_differences=[n for n,p in (model or base).state_dict().items() if tensor_tree_hash(p)!=saved_parameter_hashes[n]]
        opt=optimizer_for(base,model,args);opt.load_state_dict(reloaded_state['optimizer'])
        local=reloaded_state['ranks'][parallel.rank()]
        import_runtime(model,local['runtime']);restore_rng(local['rng'])
        replay_row=run_update(tr,next_pair,args.seed+300000,base,model,labels,pos,args,opt)
        resume_equal=(expected_weights==tensor_tree_hash((model or base).state_dict()) and
                      expected_optimizer==tensor_tree_hash(opt.state_dict()) and
                      expected_runtime==tensor_tree_hash(export_runtime(model)))
        diagnostics=dict(full_reload=full_reload,resume_equal=resume_equal,reload_differences=reload_differences,
                         update_differences=[n for n,p in (model or base).state_dict().items() if tensor_tree_hash(p)!=expected_parameter_hashes[n]],
                         weights_equal=expected_weights==tensor_tree_hash((model or base).state_dict()),
                         optimizer_equal=expected_optimizer==tensor_tree_hash(opt.state_dict()),
                         runtime_equal=expected_runtime==tensor_tree_hash(export_runtime(model)),
                         reference=expected_row,replay=replay_row)
        jsonwrite(args.output/'resume_diagnostics.json',diagnostics)
        if not full_reload or not resume_equal:raise RuntimeError('full HF reload / next optimizer update mismatch; see resume_diagnostics.json')
        ranks_match=len(set(parallel.gather(tensor_tree_hash((model or base).state_dict()))))==1
        jsonwrite(args.output/"preflight.json",dict(passed=all(np.isfinite(r["grad_norm"]) and r["grad_norm"]>0 for r in rows) and unchanged is not False and not missing_gradient and trainable_updated and backbone_unchanged and ranks_match and full_reload and resume_equal,
                   full_hf_reload_matches=full_reload,strict_resume_next_update_matches=resume_equal,
                   trainable_parameters_updated=trainable_updated,backbone_unchanged=backbone_unchanged,
                   world_size=parallel.world_size(),replicated_weights_match=ranks_match,resume_reference=expected_row,resume_reloaded=replay_row,
                   initial_equivalence=initial_equiv,trained_equivalence=trained_equiv,updated_parameters=updated,missing_gradient=missing_gradient,
                   stage=args.stage,prior=args.prior,rows=rows,base_unchanged=unchanged,
                   label_sha256=hash_file(args.labels),checkpoint=str(args.checkpoint.resolve()),states=args.states,
                   microbatch=args.microbatch,fusion_layers=args.fusion_layers,source_digest=source_digest(),
                   identity=identity(args),lambda_s=args.state_weight,split=split,base_initial_hash=initial_hash))
        status(args,"completed",3)
        return
    opt=optimizer_for(base,model,args);seq=stream(tr,args.seed)
    if saved is None:
        start=0
        save(args,base,model,proc,split,lo,hi,0,initial_hash,opt)
        jsonwrite(args.output/"validation-000000.json",validate(val,base,model,labels,pos,args,0))
    else:
        start=int(saved["step"])
        opt.load_state_dict(saved["optimizer"])
        local=saved.get("ranks",[saved])[parallel.rank()]
        import_runtime(model,local["runtime"]);restore_rng(local["rng"])
        for _ in range(start):next(seq)
    for step in range(start+1,args.steps+1):
        began=time.monotonic()
        status(args,"training",step-1)
        epoch,pair=next(seq);r=run_update(tr,pair,args.seed+(step-1)*100000,base,model,labels,pos,args,opt)
        r.update(step=step,epoch=epoch,windows=pair,seconds=time.monotonic()-began,
                 world_size=parallel.world_size(),decisions=sum(len(tr.frame_indices(i)) for i in pair),
                 peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,lr=args.lr)
        if parallel.rank()==0:
            with (args.output/"metrics.jsonl").open("a") as f:f.write(json.dumps(r)+"\n")
            print(json.dumps(r),flush=True)
        stop=bool(parallel.sum_tensor(torch.tensor(int(stopping(args,step)),device=pos.device)).item())
        if step%1000==0 or step==args.steps or stop:
            if model is not None and base_hash(base)!=initial_hash:raise RuntimeError("frozen base changed")
            save(args,base,model,proc,split,lo,hi,step,initial_hash,opt)
        if stop and step<args.steps:
            status(args,"stopped",step);parallel.barrier();return
        if step%(3000 if args.stage=="stage1" else 1000)==0 or step==args.steps:
            jsonwrite(args.output/f"validation-{step:06d}.json",validate(val,base,model,labels,pos,args,step))
        status(args,"training",step)
    status(args,"completed",args.steps)
if __name__=="__main__":main()
