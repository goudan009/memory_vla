"""Causal embodied history; no gradient through persisted observations."""
import torch
from torch import nn


class QueryGenerator(nn.Module):
    def __init__(self, vlm_dim, body_dim, dim):
        super().__init__()
        self.visual = nn.Linear(vlm_dim, body_dim)
        self.language = nn.Linear(vlm_dim, body_dim)
        self.body = nn.Linear(body_dim, body_dim)
        self.fuse = nn.Sequential(nn.Linear(3*body_dim,dim), nn.GELU(), nn.Linear(dim,dim))

    def forward(self, visual, language, body):
        return self.fuse(torch.cat((self.visual(visual),self.language(language),self.body(body)), -1))


class EmbodiedMemory(nn.Module):
    def __init__(self, dim=256, capacity=16):
        super().__init__()
        self.capacity = capacity
        self.q, self.k, self.v = [nn.Linear(dim,dim) for _ in range(3)]
        self.age = nn.Sequential(nn.Linear(2,dim),nn.Tanh(),nn.Linear(dim,dim))
        self.output = nn.Linear(dim,dim,bias=False)
        self.bank = {}

    def reset(self, keys=None):
        if keys is None: self.bank.clear()
        else:
            for key in keys: self.bank.pop(tuple(key) if isinstance(key,list) else key,None)

    def read_write(self, query, current, key, time, encoder):
        previous = self.bank.get(key, [])
        if previous and time <= previous[-1][0]:
            raise ValueError("Non-increasing history time; reset at Episode boundaries")
        if previous:
            # Re-encode detached past INPUTS. Detaching learned embeddings at
            # write time would leave the historical encoder without gradients.
            values = encoder(torch.stack([v for _,v in previous]).to(query))
            # Elapsed decision time, never future state or episode completion.
            ages = query.new_tensor([time-t for t,_ in previous]).log1p()
            pe = self.age(torch.stack((ages.sin(),ages.cos()),-1))
            # BF16 reduction + BF16 probabilities hid measured body-induced
            # query changes (~2e-5 in probabilities). Keep reduction/softmax
            # and weighted-value summation in FP32, like stable attention.
            scores = (self.q(query).float().unsqueeze(-2)*self.k(values+pe).float()).sum(-1)/query.shape[-1]**.5
            weights = scores.softmax(-1)
            retrieved = self.output((weights.unsqueeze(-1)*self.v(values)).sum(-2))
        else:
            retrieved = torch.zeros_like(query)
            weights = query.new_empty((*query.shape[:-1],0))
        times = [t for t,_ in previous]
        # Store BEFORE retrieval fusion: V2b/c with copied weights have exactly
        # the same historical contents; only their query changes.
        previous = previous+[(time,current.detach().clone())]
        if len(previous)>self.capacity:
            # V1-style adjacent similarity merge, keeping the later timestamp
            # so strict causal checks remain valid after consolidation.
            similarities = [torch.nn.functional.cosine_similarity(previous[i][1].float(),
                previous[i+1][1].float(),dim=-1).mean() for i in range(len(previous)-1)]
            i = int(torch.stack(similarities).argmax())
            previous[i:i+2] = [(previous[i+1][0],(previous[i][1]+previous[i+1][1])/2)]
        self.bank[key] = previous
        return retrieved, weights, times
