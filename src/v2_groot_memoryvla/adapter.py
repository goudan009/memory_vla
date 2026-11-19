"""V2 residuals at V1's post-Eagle / pre-vlln injection interface."""
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature
from .body import BodyEncoder
from .memory import QueryGenerator, EmbodiedMemory


class V2Adapter(nn.Module):
    def __init__(self, config, vlm_dim=2048, eos_token_id=151645):
        super().__init__()
        self.config, self.eos_token_id = config, eos_token_id
        self.diagnostics = []
        if config.body:
            self.body_encoder = BodyEncoder(config.body_dim,config.body_layers,config.body_heads)
            self.body_projection = nn.Linear(config.body_dim,vlm_dim,bias=False)
            self.body_gate = nn.Parameter(torch.tensor(config.injection_scale))
        if config.history:
            self.query = QueryGenerator(vlm_dim,config.body_dim,config.memory_dim)
            self.history_encoder = nn.Sequential(nn.Linear(config.body_dim+vlm_dim*2+31,config.memory_dim),
                nn.GELU(),nn.Linear(config.memory_dim,config.memory_dim))
            self.history = EmbodiedMemory(config.memory_dim,config.memory_capacity)
            self.memory_projection = nn.Linear(config.memory_dim,vlm_dim,bias=False)
            self.memory_gate = nn.Parameter(torch.tensor(config.injection_scale))

    def reset(self, keys=None):
        if self.config.history: self.history.reset(keys)

    def forward(self, original, fused, ids, context, graph=None, body=None, conditioning="residual"):
        self.diagnostics = []
        if not self.config.body: return fused
        if body is None or graph is None: raise ValueError("V2 requires mapped Body observations and URDF")
        h = original["backbone_features"]
        mask = original["backbone_attention_mask"].bool()
        eos = (ids==self.eos_token_id)&mask
        indices = torch.arange(h.shape[1],device=h.device).expand(h.shape[0],-1).masked_fill(~eos,-1).max(1).values
        if (indices<0).any(): raise ValueError("Missing user EOS")
        # direct replaces the full Eagle sequence with projected visual tokens
        # and one cognitive EOS. Original token indices no longer apply.
        if conditioning=="direct":
            condition_mask=fused["backbone_attention_mask"].bool() & ~fused["image_mask"].bool()
            if not torch.all(condition_mask.sum(1)==1):
                raise ValueError("Direct conditioning requires one cognitive token")
            targets=condition_mask.long().argmax(1)
        else:
            targets=indices
        encoded = self.body_encoder(graph,body)
        visual = original["visual_features"].mean((1,2))
        language = h[torch.arange(len(h),device=h.device),indices]
        pooled = encoded["pooled"]
        mode = context.get("mode")
        if self.training and mode not in {"group","train_sequence"}: raise ValueError("Ordered training metadata required")
        keys = [tuple(k) if isinstance(k,list) else k for k in context["episode_keys"]]
        steps = context["timesteps"]
        if len(keys)!=len(h) or len(steps)!=len(h): raise ValueError("Memory batch mismatch")
        group_starts=set()
        if mode=="group":
            length=int(context["sequence_length"])
            lengths=context.get("group_lengths",[length]*(len(h)//max(length,1)))
            if not lengths or min(lengths)<1 or sum(lengths)!=len(h):
                raise ValueError("Incomplete memory groups")
            offset=0
            for length in lengths:
                group_starts.add(offset); offset+=length
            self.reset()
        if mode=="train_sequence":
            if len(set(keys))!=1: raise ValueError("Chunk crosses Episode")
            if context["sequence_start"]: self.reset(keys)
            elif self.config.history and keys[0] not in self.history.bank: raise ValueError("Missing sequence_start")
        updates = []
        for i,key in enumerate(keys):
            if mode=="group" and i in group_starts: self.reset()
            elif mode=="group" and key!=keys[i-1]: raise ValueError("Group crosses Episode")
            delta = self.body_gate.tanh()*self.body_projection(pooled[i])
            diagnostic = {"body":pooled[i]}
            if self.config.history:
                qbody = pooled[i] if self.config.variant=="v2c" else torch.zeros_like(pooled[i])
                query = self.query(visual[i],language[i],qbody)
                # Joint count independent descriptor of the last command. Its
                # structure-weighted moments retain joint type/axis/origin
                # context; the trainable historical encoder runs on READ.
                g = graph[i] if isinstance(graph,(list,tuple)) else graph
                n = len(g.names)
                action = body["previous_action"][i,:n]*body["action_mask"][i,:n]
                descriptor = (g.static.to(action.device)*action.unsqueeze(-1)).sum(0)/body["action_mask"][i,:n].sum().clamp_min(1)
                command = torch.cat((descriptor,body["base"][i,6:10]),-1)
                current = torch.cat((pooled[i],visual[i],language[i],command),-1)
                if mode=="empty":
                    retrieved=torch.zeros_like(query)
                    weights=query.new_empty(0)
                    times=[]
                else:
                    retrieved, weights, times = self.history.read_write(query,current,key,int(steps[i]),self.history_encoder)
                delta = delta+self.memory_gate.tanh()*self.memory_projection(retrieved)
                diagnostic.update(query=query,attention=weights,history_times=times)
            row = fused["backbone_features"][i].clone()
            row[targets[i]] = row[targets[i]]+delta
            updates.append(row)
            self.diagnostics.append(diagnostic)
        if mode=="group" or (mode=="train_sequence" and context["sequence_end"]): self.reset(keys if mode!="group" else None)
        return BatchFeature(data={**fused,"backbone_features":torch.stack(updates)})
