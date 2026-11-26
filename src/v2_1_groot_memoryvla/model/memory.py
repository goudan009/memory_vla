"""Causal dual banks, joint retrieval, and per-flow-step refinement."""
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from .operators import BottleneckSE, CrossTransformerBlock, GateFusion, TimestepEmbedder
from .embodiment import R1Pro, mlp
from ..config import check_scale

@dataclass
class Slot:
    start: int
    end: int
    count: int
    value: torch.Tensor
    q: torch.Tensor | None = None

@dataclass
class Cache:
    body: torch.Tensor
    cognition: torch.Tensor
    memory: torch.Tensor
    active: torch.Tensor
    logits: torch.Tensor
    scores: list

def cosine_distance(a,b):
    a,b=a.float(),b.float()
    dot=F.cosine_similarity(a,b,dim=-1,eps=1e-8)
    both=(a.norm(dim=-1)<1e-8)&(b.norm(dim=-1)<1e-8)
    return torch.where(both,0.,(1-dot).clamp(0,2)*.5).mean()

class Banks:
    def __init__(self,capacity,alpha,lo,hi):
        self.capacity=capacity;self.alpha=alpha;self.lo=lo;self.hi=hi
        self.p={};self.c={};self.prev={};self.last={}
    def reset(self,keys=None):
        for d in (self.p,self.c,self.prev,self.last):
            if keys is None:d.clear()
            else:
                for k in keys:d.pop(k,None)
    def distance(self,a,b):
        if a.q is None:return cosine_distance(a.value,b.value)
        ids=[0,1,2,9,*range(12,25),46,47,*range(48,55),76,77,*range(78,82)]
        delta=(a.q[ids]-b.q[ids]).abs()
        delta[3]=torch.atan2(delta[3].sin(),delta[3].cos()).abs()
        angular=torch.atan2(a.q[6:8],a.q[3:5])-torch.atan2(b.q[6:8],b.q[3:5])
        angular=torch.atan2(angular.sin(),angular.cos()).abs()/math.pi
        span=(self.hi[ids]-self.lo[ids]).float();span[3]=math.pi
        valid=span>1e-6
        physical=torch.cat(((delta[valid]/span[valid]).clamp(0,1),angular)).mean()
        return (1-self.alpha)*cosine_distance(a.value,b.value)+self.alpha*physical
    def merge_q(self,a,b,w):
        q=a.float()*w+b.float()*(1-w)
        e=torch.atan2(a[6:9]*w+b[6:9]*(1-w),a[3:6]*w+b[3:6]*(1-w))
        q[3:6]=e.cos();q[6:9]=e.sin()
        y=torch.atan2(w*a[9].sin()+(1-w)*b[9].sin(),w*a[9].cos()+(1-w)*b[9].cos())
        q[9]=y;q[10]=y.cos();q[11]=y.sin()
        q[25:32]=q[18:25].sin();q[32:39]=q[18:25].cos()
        q[55:62]=q[48:55].sin();q[62:69]=q[48:55].cos()
        rep=a if w>=.5 else b
        q[39:46]=rep[39:46];q[69:76]=rep[69:76]
        return q
    @torch.no_grad()
    def append(self,d,key,t,value,q=None):
        slots=d.setdefault(key,[])
        slots.append(Slot(t,t,1,value.detach().clone(),None if q is None else q.detach().float().clone()))
        while len(slots)>self.capacity:
            idx=int(torch.stack([self.distance(a,b) for a,b in zip(slots,slots[1:])]).argmin())
            a,b=slots[idx:idx+2];count=a.count+b.count;w=a.count/count
            slots[idx:idx+2]=[Slot(a.start,b.end,count,(a.value.float()*w+b.value.float()*(1-w)).to(a.value.dtype),
                                  None if a.q is None else self.merge_q(a.q,b.q,w))]
    def read(self,key,t,donor=None,empty=False):
        if t<=self.last.get(key,-1):raise ValueError("duplicate or non-increasing decision")
        if empty:return [],[],None
        rk=key if donor is None else donor
        p=[s for s in self.p.get(rk,[]) if s.end<t]
        c=[s for s in self.c.get(rk,[]) if s.end<t]
        prev=self.prev.get(rk)
        return p,c,prev[1] if prev is not None and prev[0]<t else None
    def write(self,key,t,v,q,c,prev):
        self.append(self.p,key,t,v,q);self.append(self.c,key,t,c)
        self.prev[key]=(t,prev.detach().clone());self.last[key]=t

class Memory(nn.Module):
    def __init__(self,cfg,lo,hi):
        super().__init__()
        self.cfg=cfg;d=256;dc=2048;ks=len(cfg.states)
        self.robot=R1Pro(lo,hi,cfg.prior)
        self.visual=BottleneckSE(2048,512,d)
        self.enhance=mlp(dc+d+ks+1,dc);self.state_head=mlp(dc,ks)
        self.cog_read=nn.ModuleList(CrossTransformerBlock(dc,"layernorm") for _ in range(2))
        self.cog_time=TimestepEmbedder(dc,dc//4);self.cog_gate=GateFusion(dc)
        self.task_c=nn.Linear(dc,d);self.task_v=nn.Linear(d,d)
        self.query=mlp(d+dc+d,d);self.key=mlp(d+82,d);self.value=mlp(d+82,d)
        self._banks=None;self.last_diagnostics=None
    @property
    def banks(self):
        if self._banks is None:
            self._banks=Banks(self.cfg.capacity,self.cfg.alpha,self.robot.lo,self.robot.hi)
        elif self._banks.lo.device!=self.robot.lo.device:
            if self._banks.last:raise RuntimeError("reset memory before changing device")
            self._banks=Banks(self.cfg.capacity,self.cfg.alpha,self.robot.lo,self.robot.hi)
        return self._banks
    def reset(self,keys=None):
        self.banks.reset(keys)
    def current(self,features,ids):
        visual=features["visual_features"];b,c,g,d=visual.shape
        if (c,g,d)!=(3,81,2048):raise ValueError("unexpected Eagle visual shape")
        v=self.visual(visual.reshape(b*c,g,d)).reshape(b,c*g,256)
        h=features["backbone_features"];att=features["backbone_attention_mask"].bool()
        pos=torch.arange(h.shape[1],device=h.device).expand(b,-1)
        eos=pos.masked_fill(~((ids==151645)&att),-1).max(1).values
        if (eos<0).any():raise ValueError("missing user EOS")
        return v,h[torch.arange(b,device=h.device),eos]
    def prepare(self,features,ids,q,context):
        if q.shape!=(ids.shape[0],82) or not torch.isfinite(q).all():raise ValueError("finite raw82 required")
        q=q.float();v,c=self.current(features,ids);vbar=v.mean(1);b=self.robot(q)
        keys=[tuple(k) if isinstance(k,list) else k for k in context["episode_keys"]]
        times=[int(t) for t in context["timesteps"]]
        if len(keys)!=len(q) or len(times)!=len(q) or min(times)<0:raise ValueError("invalid episode context")
        mode=context.get("history_mode","normal")
        if mode not in ("normal","empty","wrong","other_episode") or (self.training and mode!="normal"):
            raise ValueError("invalid history control")
        if context.get("sequence_start"):
            if times[0]!=0:raise ValueError('episode start must begin at t=0')
            self.reset(list(set(keys)))
        donors=context.get("donor_keys")
        if mode in ("wrong","other_episode"):
            if donors is None or len(donors)!=len(keys):raise ValueError("explicit matched donor keys required")
            donors=[tuple(k) if isinstance(k,list) else k for k in donors]
            if any(k==d for k,d in zip(keys,donors)):raise ValueError("self donor")
            reads=[self.banks.read(k,t,donor=d) for k,t,d in zip(keys,times,donors)]
        else:reads=None
        cs=[];ms=[];act=[];ls=[];scores=[]
        norm=lambda x:F.layer_norm(x.float(),(x.shape[-1],))
        for i,(k,t) in enumerate(zip(keys,times)):
            if t!=self.banks.last.get(k,-1)+1:
                raise ValueError('decision clock gap or missing episode reset')
            p,old,prev=reads[i] if reads is not None else self.banks.read(k,t,empty=mode=="empty")
            if reads is not None and t>0 and not (p or old):raise ValueError('wrong-history donor has no eligible prefix')
            pv=c.new_zeros(len(self.cfg.states)) if prev is None else prev.to(c.dtype)
            cv=c[i]+self.enhance(torch.cat((c[i],vbar[i],pv,c.new_tensor([float(prev is not None)]))))
            logits=self.state_head(cv);ct=cv
            if old:
                hist=torch.stack([s.value for s in old])[None]
                pe=self.cog_time(torch.tensor([s.end for s in old],device=c.device))[None]
                msg=cv[None,None]
                for block in self.cog_read:msg=block(msg,hist+pe,hist)
                ct=self.cog_gate(cv[None,None],msg)[0,0]
            if p:
                hv=torch.stack([s.value for s in p]).mean(1)
                hq=self.robot.normalize(torch.stack([s.q for s in p]))
                pair=torch.cat((hv,hq.to(hv.dtype)),-1)
                sk=norm(self.key(pair));values=self.value(pair)
                qq=norm(self.query(torch.cat((b[i],ct.expand(23,-1),vbar[i].expand(23,-1)),-1)))
                st=norm(self.task_c(ct))@norm(self.task_v(hv)).T/math.sqrt(256)
                sb=qq@sk.T/math.sqrt(256)
                weights=(st[None]+self.cfg.retrieval_lambda*sb).softmax(-1)
                mem=weights.to(values.dtype)@values
                scores.append((st.detach(),sb.detach(),weights.detach()))
            else:
                mem=b[i].new_zeros((23,256));scores.append(None)
            cs.append(ct);ms.append(mem);act.append(bool(p or old));ls.append(logits)
            self.banks.write(k,t,v[i],q[i],cv,logits.sigmoid())
        if context.get("sequence_end"):self.reset(list(set(keys)))
        cache=Cache(b,torch.stack(cs),torch.stack(ms),torch.tensor(act,device=q.device,dtype=torch.bool),torch.stack(ls),scores)
        self.last_diagnostics={"history_active":act,"state_scores":cache.logits.detach().float().sigmoid().cpu().tolist()} if context.get("memory_diagnostics") else None
        return cache

class Block(nn.Module):
    def __init__(self,d=256):
        super().__init__()
        self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,4,batch_first=True,dropout=0.)
        self.ff=nn.Sequential(nn.Linear(d,512),nn.GELU(),nn.Linear(512,d))
    def forward(self,x,mask=None):
        n=self.n1(x);x=x+self.attn(n,n,n,attn_mask=mask,need_weights=False)[0]
        return x+self.ff(self.n2(x))

class Refinement(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg;self.action=mlp(388,256);self.fusion=mlp(256+256+2048,256)
        self.local=Block() if cfg.prior=="topology" else None
        self.global_blocks=nn.ModuleList(Block() for _ in range(cfg.fusion_layers or 1))
        self.output=nn.Linear(256,50)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x,u0,t,cache,robot,scale=1.):
        scale=check_scale(scale)
        if x.shape!=u0.shape or x.shape[1:]!=(50,128):raise ValueError("expected H50 x128")
        if scale==0 or not bool(cache.active.any()):return u0,torch.zeros_like(u0)
        ids=torch.nonzero(cache.active,as_tuple=True)[0];cols=robot.columns
        xj=x[ids].index_select(-1,cols).transpose(1,2);uj=u0[ids].index_select(-1,cols).transpose(1,2)
        times=torch.as_tensor(t,device=x.device,dtype=torch.float32).reshape(-1)
        if times.numel()==1:times=times.expand(x.shape[0])
        if times.numel()!=x.shape[0]:raise ValueError("invalid flow times")
        f=torch.exp(-math.log(10000)*torch.arange(16,device=x.device)/16)
        te=times[ids,None]*f[None];te=torch.cat((te.cos(),te.sin()),-1)[:,None].expand(-1,23,-1)
        j=self.action(torch.cat((xj,uj,cache.body[ids],te.to(xj.dtype)),-1))
        z=self.fusion(torch.cat((j,cache.memory[ids],cache.cognition[ids,None].expand(-1,23,-1)),-1))
        if self.local is not None:z=self.local(z,robot.local_mask)
        for block in self.global_blocks:z=block(z)
        raw=self.output(z).transpose(1,2).to(u0.dtype)
        if not torch.isfinite(raw).all():raise FloatingPointError("nonfinite residual")
        delta=torch.zeros_like(u0).index_copy(0,ids,torch.zeros_like(u0[ids]).index_copy(-1,cols,raw))
        # Select only active rows/valid columns; preserve untouched values exactly.
        updated=u0[ids].index_select(-1,cols)+scale*raw
        return u0.index_copy(0,ids,u0[ids].index_copy(-1,cols,updated)),delta
