"""Load the unchanged no-memory GR00T backbone; store V2.1 as a strict sidecar."""
import json
from pathlib import Path
import hashlib
import torch
from torch import nn
from ..config import Config,check_scale
from .memory import Memory,Refinement
from .velocity import training_velocity,inference_velocity

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Integrated(nn.Module):
    def __init__(self,base,cfg,lo,hi):
        super().__init__()
        if base.memory is not None or base.action_head.openvla_memory or base.action_head.memory_fusion_mode!="legacy":
            raise ValueError("V2.1 requires an unmodified no-memory base")
        if (base.action_head.action_horizon,base.action_head.action_dim)!=(50,128):
            raise ValueError("base checkpoint incompatible")
        self.base=base;self.cfg=cfg
        self.base.requires_grad_(False).eval()
        self.base.backbone.return_visual_features=True
        self.base.backbone.return_preprojected_vision=False
        self.memory=Memory(cfg,lo,hi);self.refinement=Refinement(cfg)
        self.residual_scale=cfg.residual_scale
    def train(self,mode=True):
        super().train(mode)
        self.base.eval()
        return self
    def reset(self,keys=None):
        self.memory.reset(keys)
    def prepare_cache(self,features,ids,q,context):
        return self.memory.prepare(features,ids,q,context)
    def loss_from_features(self,features,action,cache):
        return training_velocity(self.base.action_head,features,action,self.refinement,cache,self.memory.robot)
    @torch.no_grad()
    def get_action(self,inputs,raw_q,context,initial_noise=None):
        backbone,action=self.base.prepare_input(inputs)
        features=self.base.backbone(backbone)
        cache=self.prepare_cache(features,backbone["input_ids"],raw_q.to(self.base.device,dtype=torch.float32),context)
        head=self.base.action_head
        enc=head._encode_features(features,action)
        return inference_velocity(head,enc.backbone_features,enc.state_features,action.embodiment_id,features,
                                  self.refinement,cache,self.memory.robot,check_scale(self.residual_scale),initial_noise)
    def save_sidecar(self,directory):
        directory=Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        asset=Path(__file__).resolve().parents[1]/"assets/r1pro.json"
        spec={"version":"2.1","config":self.cfg.as_dict(),"asset_sha256":sha256(asset)}
        (directory/"v2_1.json").write_text(json.dumps(spec,indent=2)+"\n")
        torch.save({"memory":self.memory.state_dict(),"refinement":self.refinement.state_dict()},directory/"v2_1.pt")
    @classmethod
    def load_sidecar(cls,base,directory,device):
        directory=Path(directory);spec=json.loads((directory/"v2_1.json").read_text())
        if spec["version"]!="2.1" or spec["asset_sha256"]!=sha256(Path(__file__).resolve().parents[1]/"assets/r1pro.json"):
            raise ValueError("checkpoint/robot mapping mismatch")
        weights=torch.load(directory/"v2_1.pt",map_location="cpu",weights_only=True)
        cfg=Config(**spec["config"])
        model=cls(base,cfg,weights["memory"]["robot.lo"],weights["memory"]["robot.hi"])
        model.memory.load_state_dict(weights["memory"],strict=True)
        model.refinement.load_state_dict(weights["refinement"],strict=True)
        return model.to(device=device).eval()
