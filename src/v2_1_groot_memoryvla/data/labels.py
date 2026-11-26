"""Episode/frame-keyed supervision, quality weighting and unknown masking."""
import numpy as np
import torch
import torch.nn.functional as F

class Labels:
    def __init__(self,path,names):
        import pandas as pd
        self.names=tuple(names)
        columns=["episode_index","frame_index"]+[x for n in names for x in (n,n+"_quality")]
        df=pd.read_parquet(path,columns=columns)
        if df.duplicated(["episode_index","frame_index"]).any():raise ValueError("duplicate label key")
        self.data={}
        for eid,g in df.groupby("episode_index",sort=False):
            g=g.sort_values("frame_index")
            if not np.array_equal(g.frame_index.to_numpy(),np.arange(len(g))):raise ValueError("incomplete frame sequence")
            vals=np.stack([g[n].astype("Float32").to_numpy(dtype=np.float32,na_value=np.nan) for n in names],-1)
            quality=np.stack([g[n+"_quality"].to_numpy(dtype=np.int64) for n in names],-1)
            if not np.isin(quality,[0,1,2]).all() or not np.array_equal(np.isfinite(vals),quality>0) or not np.isin(vals[quality>0],[0,1]).all():
                raise ValueError("invalid label schema")
            self.data[int(eid)]=(vals,quality)
    def get(self,episode,frames):
        v,q=self.data[int(episode)];ix=np.asarray(frames,dtype=np.int64)
        if (ix<0).any() or (ix>=len(v)).any():raise ValueError("label frame outside episode")
        return torch.from_numpy(v[ix]),torch.from_numpy(q[ix])
    def class_weights(self,episodes,stride=8,weak_weight=.5):
        pos=np.zeros(len(self.names));neg=pos.copy()
        for eid in episodes:
            v,q=self.data[int(eid)];v=v[::stride];q=q[::stride]
            w=np.where(q==1,1.,np.where(q==2,weak_weight,0.))
            pos+=(w*(v==1)).sum(0);neg+=(w*(v==0)).sum(0)
        if (pos<=0).any() or (neg<=0).any():raise ValueError("selected training labels lack positive or negative evidence")
        return torch.tensor(np.clip(neg/pos,1,20),dtype=torch.float32)

def state_terms(logits,target,quality,pos_weight,weak_weight=.5):
    if logits.shape!=target.shape or target.shape!=quality.shape:raise ValueError("state shape mismatch")
    if not ((quality==0)|(quality==1)|(quality==2)).all():raise ValueError('invalid quality code')
    known=quality>0
    target=torch.where(known,target,0.).float()
    if not ((target==0)|(target==1)).all():raise ValueError("bad known targets")
    weight=torch.where(quality==1,1.,torch.where(quality==2,weak_weight,0.))
    loss=F.binary_cross_entropy_with_logits(logits.float(),target,pos_weight=pos_weight.float(),reduction="none")
    return (loss*weight).sum(0),known.sum(0)

def normalized_state_loss(numerator,count):
    valid=count>0
    return (numerator/count.clamp_min(1)).sum()/valid.sum().clamp_min(1)
