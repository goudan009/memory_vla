"""Explicit checkpoint runtime; detached history never part of the model weights."""
from pathlib import Path
import hashlib,random
import numpy as np
import torch
from .model.memory import Slot

def source_digest():
    root=Path(__file__).resolve().parent
    h=hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in (".py",".json") and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()

def export_runtime(model):
    if model is None:return None
    b=model.memory.banks
    def slots(d):
        return {k:[(s.start,s.end,s.count,s.value.detach().cpu(),None if s.q is None else s.q.detach().cpu()) for s in seq]
                for k,seq in d.items()}
    return {"p":slots(b.p),"c":slots(b.c),
            "prev":{k:(t,v.detach().cpu()) for k,(t,v) in b.prev.items()},"last":dict(b.last)}
def import_runtime(model,state):
    if model is None:
        if state is not None:raise ValueError("unexpected memory runtime")
        return
    if state is None:raise ValueError("missing memory runtime")
    model.reset();b=model.memory.banks;device=model.memory.robot.lo.device
    def slots(d):
        return {k:[Slot(a,z,n,v.to(device),None if q is None else q.to(device)) for a,z,n,v,q in seq] for k,seq in d.items()}
    b.p=slots(state["p"]);b.c=slots(state["c"])
    b.prev={k:(t,v.to(device)) for k,(t,v) in state["prev"].items()};b.last=dict(state["last"])
    for k,t in b.last.items():
        if any(s.end>t for s in b.p.get(k,[])+b.c.get(k,[])) or b.prev[k][0]!=t:
            raise ValueError("corrupt runtime chronology")
        if len(b.p.get(k,[]))>b.capacity or len(b.c.get(k,[]))>b.capacity:
            raise ValueError("corrupt runtime capacity")
def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all()}
def restore_rng(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
