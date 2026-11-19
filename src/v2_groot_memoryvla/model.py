"""Composition around V1; shared training/inference fusion, no V1 monkey patch."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import torch
from torch import nn
from .adapter import V2Adapter
from .config import V2Config
from .data import BodyMapper


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_memory_config(saved, config):
    """Preserve learned V1 architecture; use the current protocol for fresh GR00T."""
    if config.variant=="baseline": return config
    trained=saved.get("memory_enabled",False)
    conditioning=saved.get("memory_conditioning","residual") if trained else "direct"
    capacity=saved.get("memory_length",16) if trained else 64
    if trained:
        if config.memory_conditioning not in {None,conditioning}:
            raise ValueError("Cannot change conditioning of a trained V1 checkpoint")
        if config.memory_length not in {None,capacity}:
            raise ValueError("Cannot change memory capacity of a trained V1 checkpoint")
    return replace(config,memory_conditioning=config.memory_conditioning or conditioning,
                   memory_length=config.memory_length or capacity)


class V2Model(nn.Module):
    def __init__(self, base, config, mapper=None, source=None):
        super().__init__()
        self.base, self.config, self.mapper = base,config,mapper
        self.source = str(Path(source).resolve()) if source else None
        self.extension = V2Adapter(config,base.config.backbone_embedding_dim,
                                   base.config.memory_eos_token_id)
        if config.variant != "baseline" and base.memory is None:
            raise ValueError("V1/V2 requires V1 memory in the supplied base")
        if config.body and mapper is None: raise ValueError("URDF mapping required")

    @classmethod
    def from_v1(cls, checkpoint, config, urdf=None, mapping=None):
        from v1_groot_memoryvla.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
        saved = json.loads((Path(checkpoint)/"config.json").read_text())
        config = resolve_memory_config(saved,config)
        memory = config.variant != "baseline"
        if config.variant=="baseline" and saved.get("memory_enabled",False):
            raise ValueError("Baseline needs the original no-memory checkpoint")
        memory_options=dict(memory_conditioning=config.memory_conditioning,memory_length=config.memory_length) if memory else {}
        base,info = Gr00tN1d6.from_pretrained(checkpoint,memory_enabled=memory,**memory_options,
            tune_llm=False,tune_visual=False,tune_top_llm_layers=0,
            backbone_trainable_params_fp32=False,torch_dtype=torch.float32,output_loading_info=True)
        missing = info["missing_keys"]
        if info["unexpected_keys"] or info.get("mismatched_keys") or any(not k.startswith("memory.") for k in missing):
            raise ValueError(f"Incompatible V1 checkpoint: {info}")
        if saved.get("memory_enabled",False) and missing:
            raise ValueError("Trained V1 parameters were not restored")
        base.backbone.requires_grad_(False).bfloat16().eval()
        if memory: base.backbone.return_visual_features=True
        model = cls(base,config,BodyMapper(urdf,**({"mapping":mapping} if mapping else {})) if config.body else None,checkpoint)
        model.loading_info = info
        return model

    def train(self, mode=True):
        super().train(mode)
        self.base.backbone.eval()  # Cached Eagle must remain deterministic.
        return self

    def reset(self, keys=None):
        if self.base.memory is not None: self.base.memory.reset(keys)
        self.extension.reset(keys)

    def fuse(self, features, ids, context, body=None):
        original = features
        if self.config.variant!="baseline": features = self.base.memory(features,ids,context)
        return self.extension(original,features,ids,context,
            self.mapper.graph if self.mapper else None,body,
            conditioning=self.base.memory.conditioning if self.base.memory is not None else "residual")

    def forward_cached(self, features, ids, context, actions, body=None, predict=False):
        features = self.fuse(features,ids,context,body)
        head = self.base.action_head
        return head.get_action(features,actions) if predict else head(features,actions)

    def forward(self, inputs, predict=False):
        inputs = dict(inputs)
        body = inputs.pop("body",None)
        context = inputs.pop("memory_context",None)
        if body is not None and self.config.body:
            device = next(self.extension.parameters()).device
            body = {k:v.to(device) for k,v in body.items()}
        backbone,actions = self.base.prepare_input(inputs)
        with torch.no_grad(): features = self.base.backbone(backbone)
        return self.forward_cached(features,backbone["input_ids"],context,actions,body,predict)

    def get_action(self, inputs):
        return self.forward(inputs,predict=True)

    def save(self, directory, processor=None):
        """Full portable V1 model + new modules; never overwrite the source."""
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=False)
        self.base.save_pretrained(directory/"base")
        if processor is not None: processor.save_pretrained(directory/"base")
        elif self.source:
            for name in ("processor_config.json","statistics.json","embodiment_id.json"):
                source = Path(self.source)/name
                if source.exists(): (directory/"base"/name).write_bytes(source.read_bytes())
        self.config.save(directory/"v2_config.json")
        torch.save(self.extension.state_dict(),directory/"extension.pt")
        if self.mapper:
            (directory/"robot.urdf").write_bytes(Path(self.mapper.graph.urdf).read_bytes())
            (directory/"mapping.json").write_bytes(Path(self.mapper.mapping).read_bytes())

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        config = V2Config.load(directory/"v2_config.json")
        model = cls.from_v1(directory/"base",config,directory/"robot.urdf",directory/"mapping.json")
        model.extension.load_state_dict(torch.load(directory/"extension.pt",map_location="cpu",weights_only=True),strict=True)
        return model

    def save_extension(self, directory):
        """Small checkpoint allowed only when every reused V1 weight is frozen.

        Stores an immutable base reference instead of copying 6+GB. Reload
        verifies base shard hashes; changed V1 training needs full save().
        """
        if any(p.requires_grad for p in self.base.parameters()):
            raise ValueError("Use full save when training V1 parameters")
        if getattr(self,"loading_info",{}).get("missing_keys"):
            raise ValueError("Use full save for newly initialized V1 memory")
        if self.source is None: raise ValueError("Missing base checkpoint reference")
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=False)
        self.config.save(directory/"v2_config.json")
        torch.save(self.extension.state_dict(),directory/"extension.pt")
        metadata = {}
        names = ["config.json","model.safetensors.index.json"]
        names += sorted(set(json.loads((Path(self.source)/names[1]).read_text())["weight_map"].values()))
        for name in names:
            p = Path(self.source)/name
            metadata[name] = file_digest(p)
        (directory/"manifest.json").write_text(json.dumps({"base":self.source,"metadata":metadata,
            "urdf":self.mapper.graph.urdf,"mapping":self.mapper.mapping,
            "urdf_sha256":self.mapper.graph.sha256,
            "mapping_sha256":hashlib.sha256(Path(self.mapper.mapping).read_bytes()).hexdigest()},indent=2))

    @classmethod
    def load_extension(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory/"manifest.json").read_text())
        for name, digest in manifest["metadata"].items():
            if file_digest(Path(manifest["base"])/name)!=digest:
                raise ValueError("Base checkpoint metadata changed")
        for name in ("urdf","mapping"):
            if hashlib.sha256(Path(manifest[name]).read_bytes()).hexdigest()!=manifest[name+"_sha256"]:
                raise ValueError("Robot definition changed")
        model = cls.from_v1(manifest["base"],V2Config.load(directory/"v2_config.json"),manifest["urdf"],manifest["mapping"])
        model.extension.load_state_dict(torch.load(directory/"extension.pt",weights_only=True,map_location="cpu"),strict=True)
        return model
