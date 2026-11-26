"""GR00T V2 integration: loss-only labels, frozen expert, original memory injection."""

import torch
from transformers import AutoConfig, AutoModel, PreTrainedModel

from v1_groot_memoryvla.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6 as V1Model
from ..config import Gr00tN1d6Config, adapter_options
from ..training import configure_trainable_parameters
from .adapter import MemoryAdapter
from .state import masked_state_loss


class Gr00tN1d6(V1Model):
    config_class = Gr00tN1d6Config

    @classmethod
    def from_pretrained(cls, *args, initialize_from_v1=False, **kwargs):
        """Audit the actual HF load, not the temporary adapter constructor.

        V2 reloads require every parameter. Explicit V1 initialization permits
        exactly the new state parameters to be absent, and nothing else.
        """
        return_info = kwargs.pop("output_loading_info", False)
        model, info = super().from_pretrained(*args, output_loading_info=True, **kwargs)
        state_keys = {name for name in model.state_dict() if ".state_head." in name
                      or ".state_query." in name or ".state_gate." in name}
        expected_missing = state_keys if initialize_from_v1 else set()
        if (set(info["missing_keys"]) != expected_missing or info["unexpected_keys"]
                or info.get("mismatched_keys") or info.get("error_msgs")):
            raise ValueError(f"Incompatible checkpoint for {'V1 initialization' if initialize_from_v1 else 'V2 reload'}: {info}")
        return (model, info) if return_info else model

    def __init__(self, config, transformers_loading_kwargs=None):
        if not config.memory_enabled:
            raise ValueError("V2 requires a compatible V1 memory adapter")
        super().__init__(config, transformers_loading_kwargs=(
            {"trust_remote_code": True} if transformers_loading_kwargs is None else transformers_loading_kwargs))
        self.memory = MemoryAdapter.from_v1(self.memory, **adapter_options(config))
        configure_trainable_parameters(self)
        self.train(self.training)

    @classmethod
    def from_v1_model(cls, model, *, state_labels, state_loss_weight=None, **state_config):
        """Reuse loaded backbone/expert objects and copy V1 memory weights.

        Ownership of backbone/action_head transfers to the V2 wrapper: their
        parameters become frozen. This avoids loading a second multi-GB base.
        Start V2 episodes from their own observation prefixes (no copied history).
        """
        if model.memory is None:
            raise ValueError("The V1 initialization must contain memory weights")
        allowed = {"memory_state_guidance", "memory_event_low", "memory_event_high", "memory_recent_slots"}
        if set(state_config) - allowed:
            raise ValueError("Only V2 state configuration may be overridden: "
                             + str(sorted(set(state_config) - allowed)))
        config_dict = model.config.to_dict()
        config_dict.pop("model_type", None)
        config_dict.update(memory_state_labels=state_labels,
                           memory_state_loss_weight=state_loss_weight, **state_config)
        config = Gr00tN1d6Config(**config_dict)
        result = cls.__new__(cls)
        PreTrainedModel.__init__(result, config)
        result.backbone, result.action_head = model.backbone, model.action_head
        result.collator = model.collator
        result.memory = MemoryAdapter.from_v1(model.memory, **adapter_options(config))
        configure_trainable_parameters(result)
        result.train(model.training)
        return result

    def train(self, mode=True):
        super().train(mode)
        # Frozen dropout/normalization must not change the baseline branches.
        for name in ("backbone", "action_head"):
            module = getattr(self, name, None)
            if module is not None:
                module.eval()
        if self.memory is not None:
            self.memory.perceptual.eval()
            self.memory.perceptual_encoder.eval()
        return self

    def prepare_input(self, inputs):
        # Physical truth is not visible to either the VLM or the action expert.
        clean = {k: v for k, v in inputs.items() if k not in ("state_targets", "state_known")}
        return super().prepare_input(clean)

    def forward(self, inputs):
        if ("state_targets" in inputs) != ("state_known" in inputs):
            raise ValueError("Physical targets and known mask must be provided together")
        if self.training and ("state_targets" not in inputs or self.config.memory_state_loss_weight is None):
            raise ValueError("Joint training requires audited labels/mask and a calibrated state loss weight")
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        features = self._backbone_with_memory(backbone_inputs, inputs.get("memory_context"))
        # No no_grad here: action loss must backpropagate through the frozen expert.
        outputs = self.action_head(features, action_inputs)
        action_loss = outputs["loss"]
        state_loss = (masked_state_loss(features["state_logits"], inputs["state_targets"], inputs["state_known"])
                      if "state_targets" in inputs else features["state_logits"].float().sum() * 0)
        outputs["action_loss_unreduced"] = outputs["action_loss"]
        outputs.update(action_loss=action_loss, state_loss=state_loss,
                       state_logits=features["state_logits"], state_feature=features["state_feature"],
                       retrieved_memory=features["retrieved_memory"], memory_gate=features["memory_gate"])
        outputs["loss"] = action_loss + (self.config.memory_state_loss_weight or 0.) * state_loss
        return outputs

    def get_action(self, inputs):
        # V1's legacy whole-bank diagnostic helper does not snapshot V2 event
        # state. Use explicit cognitive_history_mode + V2 runtime snapshots.
        if inputs.get("memory_context", {}).get("memory_action_controls"):
            raise ValueError("Use V2 cognitive_history_mode with V2 runtime snapshots")
        return super().get_action(inputs)


AutoConfig.register("Gr00tN1d6V2", Gr00tN1d6Config)
AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
