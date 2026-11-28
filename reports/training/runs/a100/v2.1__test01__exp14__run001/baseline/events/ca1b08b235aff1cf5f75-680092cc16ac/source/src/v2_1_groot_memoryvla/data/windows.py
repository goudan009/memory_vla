"""Audited state ordering, two independent sequential window slots, train-only statistics."""
import json
from pathlib import Path
import numpy as np
from v1_groot_memoryvla.data.dataset.episode_windows import EpisodeWindows
from v1_groot_memoryvla.data.convert_b1k import STATE_SLICES,ACTION_SLICES,RELATIVE_STATE

TAG="behavior_r1_pro"

def raw_from_states(states):
    q=np.concatenate([np.asarray(states[k],dtype=np.float32).reshape(-1) for k in STATE_SLICES])
    if q.shape!=(82,) or not np.isfinite(q).all():raise ValueError("raw82 observation invalid")
    return q

class Windows(EpisodeWindows):
    def get_group(self,index,sampling_seed=None):
        result=super().get_group(index,sampling_seed)
        q=np.concatenate([np.stack(self._cached_low_dim["state."+k].iloc[result["frame_indices"]].to_numpy()) for k in STATE_SLICES],axis=-1)
        if q.shape!=(len(result["steps"]),82) or not np.isfinite(q).all():raise ValueError("bad physical states")
        result["raw_q"]=q.astype(np.float32)
        # Avoid the V1 cache retaining all episodes indefinitely; active windows only.
        key=self.groups[index][:2]
        self.parquet_cache={key:self.parquet_cache[key]}
        return result

def make_split(dataset,seed=42):
    meta=[json.loads(l) for l in (Path(dataset)/"meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    ids=sorted(int(x["episode_index"]) for x in meta)
    if len(ids)!=200 or len(set(ids))!=200:raise ValueError("Task40 requires the audited 200 episodes")
    ids=np.random.default_rng(seed).permutation(ids).tolist()
    return {"task":40,"seed":seed,"train":sorted(ids[20:]),"validation":sorted(ids[:20]),
            "train_episodes_per_task":180,"validation_episodes_per_task":20,"validation_enabled":True}

def schedule(ds,seed,epoch):
    ids=np.array(sorted(ds.by_episode))
    np.random.default_rng(seed+epoch).shuffle(ids)
    queue=iter(ids.tolist())
    slots=[iter(ds.by_episode[e]) for e in [next(queue),next(queue)]]
    while slots:
        pair=[];next_slots=[]
        for slot in slots:
            try:i=next(slot)
            except StopIteration:
                try:slot=iter(ds.by_episode[next(queue)]);i=next(slot)
                except StopIteration:continue
            pair.append(i);next_slots.append(slot)
        if pair:yield pair
        slots=next_slots

class Stats:
    def __init__(self,shape):
        self.n=0;self.lo=np.full(shape,np.inf);self.hi=np.full(shape,-np.inf)
        self.s=np.zeros(shape);self.s2=np.zeros(shape)
    def add(self,x):
        x=np.asarray(x,dtype=np.float64)
        if not len(x):return
        if not np.isfinite(x).all():raise ValueError("nonfinite fit samples")
        self.n+=len(x);self.lo=np.minimum(self.lo,x.min(0));self.hi=np.maximum(self.hi,x.max(0))
        self.s+=x.sum(0);self.s2+=(x*x).sum(0)
    def done(self):
        if self.n<=0:raise ValueError("empty statistics")
        mean=self.s/self.n;std=np.sqrt(np.maximum(self.s2/self.n-mean**2,0))
        return {k:v.tolist() for k,v in dict(min=self.lo,max=self.hi,mean=mean,std=std).items()}

def fit_statistics(ds,processor):
    if processor.use_percentiles:raise ValueError("protocol requires min/max normalization")
    state={k:Stats((b-a,)) for k,(a,b) in STATE_SLICES.items()}
    action={k:Stats((b-a,)) for k,(a,b) in ACTION_SLICES.items()}
    relative={k:[Stats(action[k].lo.shape) for _ in range(32)] for k in RELATIVE_STATE}
    for eid in sorted(ds.by_episode):
        df=ds.loaders[0]._load_parquet_data(eid)
        q={k:np.stack(df["state."+k]) for k in state}
        a={k:np.stack(df["action."+k]) for k in action}
        for k in state:state[k].add(q[k])
        for k in action:action[k].add(a[k])
        for k,sk in RELATIVE_STATE.items():
            for h in range(32):
                end=len(df)-h
                if end>0:relative[k][h].add(a[k][h:]-q[sk][:end])
    result={"state":{k:v.done() for k,v in state.items()},"action":{k:v.done() for k,v in action.items()},
            "relative_action":{}}
    for k,seq in relative.items():
        ss=[v.done() for v in seq]
        result["relative_action"][k]={f:[s[f] for s in ss] for f in ss[0]}
    lo=np.concatenate([state[k].lo for k in STATE_SLICES]);hi=np.concatenate([state[k].hi for k in STATE_SLICES])
    return {TAG:result},lo,hi
