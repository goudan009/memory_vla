"""Fixed held-out history/scale interventions and per-state quality-separated metrics.

Uses only original observations in the policy; labels select/evaluate interventions offline.
No simulator start, no official test-instance tuning.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from .data.windows import Windows,TAG
from .data.labels import Labels
from .model.integration import Integrated
from .model.memory import Slot
from .model.velocity import inference_velocity
from .loading import load_base
from v1_groot_memoryvla.experiment.first_training import seeded
from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor

GROUPS={"base":(0,3),"torso":(3,7),"left_arm":(7,14),"left_gripper":(14,15),
        "right_arm":(15,22),"right_gripper":(22,23)}
def snapshot(banks,key):
    def cpu(slots):
        return [Slot(s.start,s.end,s.count,s.value.detach().cpu().clone(),None if s.q is None else s.q.detach().cpu().clone()) for s in slots]
    prev=banks.prev.get(key)
    return (cpu(banks.p.get(key,[])),cpu(banks.c.get(key,[])),
            None if prev is None else (prev[0],prev[1].detach().cpu().clone()),banks.last.get(key,-1))
def restore(banks,key,snap):
    p,c,prev,last=snap
    def gpu(slots):
        return [Slot(s.start,s.end,s.count,s.value.cuda(),None if s.q is None else s.q.cuda()) for s in slots]
    banks.p[key]=gpu(p);banks.c[key]=gpu(c)
    if prev is not None:banks.prev[key]=(prev[0],prev[1].cuda())
    if last>=0:banks.last[key]=last
def rms(x,mask):
    return float(((x.float().square()*mask).sum()/mask.sum().clamp_min(1)).sqrt())
def metrics(out,act,gain):
    pred=out["action_pred"].float();target=act.action.float();mask=act.action_mask.float()
    error=(pred-target).abs()
    results={}
    for group,(a,b) in GROUPS.items():
        m=mask[:,:8,a:b];den=float(m.sum())
        row={"mae_first8":float((error[:,:8,a:b]*m).sum()/den) if den else None}
        stats=[]
        for u,d in out.get("residual_stats",[]):
            m=mask[:,:,a:b];u0=rms(u[:,:,a:b],m);raw=rms(d[:,:,a:b],m);eff=abs(gain)*raw
            stats.append({"u0_rms":u0,"raw_residual_rms":raw if gain!=0 else None,"effective_residual_rms":eff,
                          "ratio":eff/(u0+1e-8),"near_zero_base":u0<1e-5})
        row["flow_steps"]=stats
        results[group]=row
    return results
def state_summary(scores,y,q):
    result={}
    for quality,valid in (("all",q>0),("certified",q==1),("weak",q==2)):
        s=scores[valid];truth=y[valid]==1
        if not len(s):result[quality]={"count":0};continue
        pred=s>=.5;tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
        order=np.argsort(-s,kind="stable");t=truth[order];sorted_scores=s[order]
        ends=np.r_[np.flatnonzero(sorted_scores[:-1]!=sorted_scores[1:]),len(s)-1]
        cum=np.cumsum(t)[ends];prec=cum/(ends+1)
        ap=float((np.diff(np.r_[0,cum])*prec).sum()/cum[-1]) if cum[-1]>0 else None
        result[quality]={"count":len(s),"positive":int(truth.sum()),"tp":tp,"fp":fp,"fn":fn,
                         "precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),"f1":2*tp/max(2*tp+fp+fn,1),
                         "average_precision":ap}
    return result
def event_summary(scores,y,q,tolerance=2):
    # Known-to-known transitions can span unknown frames; preserve their intervals.
    # Count predicted transitions only where they could be matched to known observations.
    known=np.flatnonzero(q>0);truth=[]
    for a,b in zip(known,known[1:]):
        if y[a]!=y[b]:truth.append((int(a),int(b),bool(y[b])))
    pred=scores>=.5
    changes=[(int(t),bool(pred[t])) for t in np.flatnonzero(pred[1:]!=pred[:-1])+1
             if len(known) and known[0]<=t<=known[-1]]
    used=set();matched=0
    for a,b,kind in truth:
        candidates=[(abs(t-b),i) for i,(t,k) in enumerate(changes) if i not in used and k==kind and a-tolerance<=t<=b+tolerance]
        if candidates:
            _,i=min(candidates);used.add(i);matched+=1
    return {"true_intervals":len(truth),"predicted_events":len(changes),"matched":matched}

def choose_points(ds,labels):
    ids=sorted(ds.by_episode);lengths={e:len(labels.data[e][0][::8]) for e in ids}
    points=[]
    for e in ids:
        length=lengths[e]
        for frac in (.1,.3,.5,.7,.9):
            t=min(length-1,max(1,round(frac*(length-1))))
            y,q=labels.data[e];candidates=[]
            for d in ids:
                if d==e or t>=lengths[d] or not .8<=lengths[d]/length<=1.25:continue
                dy,dq=labels.data[d];i=(t*8)
                # Compare prior observed state, not future or target actions.
                j=(t-1)*8
                conflict=(q[j]>0)&(dq[j]>0)&(y[j]!=dy[j])
                if conflict.any():
                    cert=conflict&(q[j]==1)&(dq[j]==1)
                    candidates.append((-int(cert.sum()),-int(conflict.sum()),abs(lengths[d]-length),d))
            donor=min(candidates)[-1] if candidates else None
            points.append((e,t,donor))
    return points

@torch.no_grad()
def run(args):
    seeded(args.seed)
    checkpoint=args.checkpoint
    base=load_base(checkpoint,"memory")
    model=Integrated.load_sidecar(base,checkpoint,"cuda").eval()
    proc=Gr00tN1d6Processor.from_pretrained(checkpoint);proc.eval()
    split=json.loads((checkpoint/"v2_1_stage1.json").read_text())["split"]
    ds=Windows(args.dataset,proc,split["validation"],seed=42)
    labels=Labels(args.labels,model.cfg.states)
    points=choose_points(ds,labels);wanted={}
    for e,t,d in points:
        wanted.setdefault(e,set()).add(t)
        if d is not None:wanted.setdefault(d,set()).add(t)
    snapshots={};features={};state_rows={}
    # Each episode encoded exactly once; bank snapshots are CPU, detached, strictly pre-read.
    for e in sorted(ds.by_episode):
        model.reset();key=str(e);scores=[]
        for index in ds.by_episode[e]:
            g=ds.get_group(index)
            for a in range(len(g["steps"])):
                t=g["timesteps"][a]
                inp=proc.collator(g["steps"][a:a+1])["inputs"]
                bk,act=base.prepare_input(inp)
                raw=torch.from_numpy(g["raw_q"][a:a+1]).cuda()
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    feat=base.backbone(bk)
                    if t in wanted.get(e,set()):
                        snapshots[e,t]=snapshot(model.memory.banks,key)
                        if any(x[0]==e and x[1]==t for x in points):
                            # Store only tensors, preserving exact dtype.
                            features[e,t]=({k:v.detach().cpu() for k,v in feat.items()},bk["input_ids"].cpu(),
                                          {k:act[k].detach().cpu() for k in ("state","action","action_mask","embodiment_id")},raw.cpu())
                    cache=model.prepare_cache(feat,bk["input_ids"],raw,{"episode_keys":[key],"timesteps":[t],"sequence_start":t==0})
                scores.append(cache.logits.float().sigmoid().cpu().numpy()[0])
        state_rows[e]=np.stack(scores);model.reset()
    from transformers.feature_extraction_utils import BatchFeature
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for k,(e,t,d) in enumerate(points):
        f,ids,a,raw=features[e,t]
        feat=BatchFeature({n:v.cuda() for n,v in f.items()});act=BatchFeature({n:v.cuda() for n,v in a.items()})
        ids=ids.cuda();raw=raw.cuda()
        row={"episode":e,"decision":t,"frame":t*8,"donor":d,"conditions":{}}
        with torch.autocast("cuda",dtype=torch.bfloat16):
            head=base.action_head;enc=head._encode_features(feat,act)
            def evaluate(cache,gain):
                seeded(args.seed+k*100)
                return inference_velocity(head,enc.backbone_features,enc.state_features,act.embodiment_id,feat,
                                          model.refinement,cache,model.memory.robot,gain)
            for mode in ("normal","empty","wrong"):
                if mode=="wrong" and d is None:continue
                model.reset();target="target"
                restore(model.memory.banks,target,snapshots[e,t])
                ctx={"episode_keys":[target],"timesteps":[t],"history_mode":mode}
                if mode=="wrong":
                    restore(model.memory.banks,"donor",snapshots[d,t]);ctx["donor_keys"]=["donor"]
                cache=model.prepare_cache(feat,ids,raw,ctx)
                gains=(0.,.25,.5,1.) if mode=="normal" else (1.,)
                for gain in gains:
                    out=evaluate(cache,gain)
                    if gain==0:
                        seeded(args.seed+k*100);ref=head.get_action(feat,act)["action_pred"]
                        if not torch.equal(ref,out["action_pred"]):raise RuntimeError("gamma0 baseline mismatch")
                    if mode=="empty":
                        seeded(args.seed+k*100);ref=head.get_action(feat,act)["action_pred"]
                        if not torch.equal(ref,out["action_pred"]):raise RuntimeError("empty baseline mismatch")
                    row["conditions"][f"{mode}_{gain:g}"]={"groups":metrics(out,act,gain),
                        "state_scores":cache.logits.float().sigmoid().cpu().tolist(),
                        "retrieval_scores":[None if z is None else [v.float().cpu().tolist() for v in z] for z in cache.scores]}
            model.reset()
        rows.append(row)
    summaries={}
    for i,name in enumerate(model.cfg.states):
        s=np.concatenate([state_rows[e][:,i] for e in sorted(state_rows)])
        y=np.concatenate([labels.data[e][0][::8,i] for e in sorted(state_rows)])
        q=np.concatenate([labels.data[e][1][::8,i] for e in sorted(state_rows)])
        if len(s)!=len(y):raise ValueError("state validation frame mismatch")
        summaries[name]=state_summary(s,y,q)
        events=[event_summary(state_rows[e][:,i],labels.data[e][0][::8,i],labels.data[e][1][::8,i]) for e in state_rows]
        summaries[name]["events"]={n:sum(x[n] for x in events) for n in events[0]}
    (args.output/"state_metrics.json").write_text(json.dumps(summaries,indent=2)+"\n")
    (args.output/"history_scale_diagnostics.json").write_text(json.dumps(rows,indent=2)+"\n")
def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ("checkpoint","dataset","labels","output"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    if args.execute:run(args)
    else:print(vars(args))
if __name__=="__main__":main()
