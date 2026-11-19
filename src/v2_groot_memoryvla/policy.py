"""Original GR00T schemas/processor with explicit executed-action feedback.

This policy is opt-in. Existing V1 servers/evaluators are not changed.
"""
from pathlib import Path
import numpy as np
import torch
from v1_groot_memoryvla.policy.gr00t_policy import Gr00tPolicy,Gr00tSimPolicyWrapper
from v1_groot_memoryvla.policy.policy import BasePolicy
from v1_groot_memoryvla.policy.memory_state import MemoryState
from v1_groot_memoryvla.data.embodiment_tags import EmbodimentTag
from v1_groot_memoryvla.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from .model import V2Model


class V2Policy(Gr00tPolicy):
    def __init__(self, model, processor, device="cuda", strict=True):
        BasePolicy.__init__(self,strict=strict)
        self.model = model.to(device).eval()
        self.processor = processor
        self.processor.eval()
        self.embodiment_tag = EmbodimentTag.BEHAVIOR_R1_PRO
        self.modality_configs = processor.get_modality_configs()[self.embodiment_tag.value]
        self.language_key = self.modality_configs["language"].modality_keys[0]
        self.collate_fn = processor.collator
        self.memory_state = MemoryState(model)

    @classmethod
    def load(cls, checkpoint, device="cuda"):
        path=Path(checkpoint)
        model = V2Model.load_extension(path) if (path/"manifest.json").exists() else V2Model.load(path)
        processor=Gr00tN1d6Processor.from_pretrained(model.source or path/"base")
        return cls(model,processor,device)

    def _get_action(self, observation, options=None):
        options=options or {}
        observations=self._unbatch_observation(observation)
        count=len(observations)
        context=self.memory_state.context(count,options)
        body=None
        if self.model.config.body:
            # Timestamp at image/state capture, not wall-clock receipt time.
            times=options["observation_timestamps"]
            actions=options.get("executed_actions",[None]*count)
            action_times=options.get("executed_action_timestamps",[None]*count)
            extra=options.get("body_state",[{} for _ in range(count)])
            if any(len(v)!=count for v in (times,actions,action_times,extra)):
                raise ValueError("Body feedback batch mismatch")
            records=[]
            for i,obs in enumerate(observations):
                if context["timesteps"][i]>0 and actions[i] is None:
                    raise ValueError("Missing last executed command; predicted chunks cannot substitute")
                fields={k:np.asarray(v)[-1] for k,v in obs["state"].items()}
                # Optional measured qvel from the SAME captured observation.
                # Mask unavailable velocity; never infer it from future frames.
                fields.update(extra[i])
                records.append(self.model.mapper.observation(fields,actions[i],times[i],action_times[i]))
            body={k:torch.from_numpy(np.stack([r[k] for r in records])) for k in records[0]}
        steps=[self.processor([{"type":"episode_step","content":self._to_vla_step_data(o)}]) for o in observations]
        batch=self.collate_fn(steps)["inputs"]
        batch.update(memory_context=context,body=body)
        with torch.inference_mode(),torch.autocast(next(self.model.parameters()).device.type,
                dtype=torch.bfloat16,enabled=next(self.model.parameters()).is_cuda):
            pred=self.model.get_action(batch)["action_pred"].float().cpu().numpy()
        states={k:np.stack([o["state"][k] for o in observations]) for k in self.modality_configs["state"].modality_keys}
        result=self.processor.decode_action(pred,self.embodiment_tag,states)
        self.memory_state.advance(context)
        return {k:v.astype(np.float32) for k,v in result.items()},{}


class V2SimPolicyWrapper(Gr00tSimPolicyWrapper):
    """Preserve measured velocity omitted by V1's nested modality selection."""
    def _get_action(self, observation, options=None):
        options=dict(options or {})
        if self.policy.model.config.body:
            batch=next(v for k,v in observation.items() if k.startswith("video.")).shape[0]
            keys={g["velocity"] for g in self.policy.model.mapper.spec["groups"]}
            # Values come from the SAME incoming capture, not external truth.
            options["body_state"]=[{k:observation["state."+k][i,-1]
                for k in keys if "state."+k in observation} for i in range(batch)]
        return super()._get_action(observation,options)
