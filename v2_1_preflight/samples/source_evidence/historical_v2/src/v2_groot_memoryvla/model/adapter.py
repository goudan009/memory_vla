"""Keep the V1 perceptual and EOS injection graph, replacing cognition only."""

from copy import deepcopy

import torch

from v1_groot_memoryvla.model.memory.adapter import MemoryAdapter as V1MemoryAdapter
from .bank import CognitiveMemoryBank


class MemoryAdapter(V1MemoryAdapter):
    def __init__(self, *args, state_labels, guidance_enabled=True, event_low=.25,
                 event_high=.75, recent_slots=2, **kwargs):
        super().__init__(*args, **kwargs)
        if self.conditioning != "residual" or self.expose_cognitive_history or hasattr(self, "cognitive_fusion"):
            raise ValueError("V2 experiment01 requires the V1 legacy residual injection path")
        old = self.cognitive
        dim = old.retrieval_blocks[0].q_proj.in_features
        self.cognitive = CognitiveMemoryBank(dim, state_labels, old.capacity,
            len(old.retrieval_blocks), old.retrieval_blocks[0].qk_norm, old.update_fused,
            guidance_enabled, event_low, event_high, recent_slots)
        result = self.cognitive.load_state_dict(old.state_dict(), strict=False)
        if result.unexpected_keys:
            raise ValueError(str(result))
        for module in self.cognitive.modules():
            module._memory_v1 = True

    @classmethod
    def from_v1(cls, adapter, **state_options):
        """Copy an already-loaded V1 adapter without touching backbone/action weights."""
        if type(adapter) is not V1MemoryAdapter:
            raise TypeError("Expected an unmodified V1 MemoryAdapter")
        options = dict(vlm_dim=adapter.cognitive.retrieval_blocks[0].q_proj.in_features,
            perceptual_dim=adapter.perceptual.retrieval_blocks[0].q_proj.in_features,
            capacity=adapter.cognitive.capacity, retrieval_layers=len(adapter.cognitive.retrieval_blocks),
            group_size=adapter.group_size, eos_token_id=adapter.eos_token_id,
            conditioning=adapter.conditioning,
            cognitive_qk_norm=adapter.cognitive.retrieval_blocks[0].qk_norm,
            residual_identity_init=adapter.residual_identity_init,
            cognitive_residual_projection=hasattr(adapter, "cognitive_residual"),
            cognitive_residual_scope=adapter.cognitive_residual_scope,
            residual_readout=adapter.residual_readout,
            visual_dim=adapter.perceptual_encoder.reduce.in_channels)
        if adapter.expose_cognitive_history or hasattr(adapter, "cognitive_fusion"):
            raise ValueError("Only legacy V1 adapters can initialize this V2 method")
        result = cls(**options, **state_options)
        result.load_v1_state_dict(adapter.state_dict())
        result.train(adapter.training)
        return result

    def load_v1_state_dict(self, state):
        """Only the newly introduced head/Q/gate parameters may be absent."""
        new = {n for n in self.state_dict() if ".state_head." in n or
               ".state_query." in n or ".state_gate." in n}
        expected = set(self.state_dict()) - new
        if set(state) != expected:
            raise ValueError(f"Incompatible V1 checkpoint: missing={sorted(expected-set(state))}, "
                             f"unexpected={sorted(set(state)-expected)}")
        # Assign cloned tensors so every original dtype/device/value survives,
        # including a scalar with a different dtype from the memory networks.
        # Initialization happens before the caller constructs its optimizer.
        self.load_state_dict({name: value.detach().clone() for name, value in state.items()},
                             strict=False, assign=True)
        anchor = self.cognitive.retrieval_blocks[0].q_proj.weight
        self.cognitive.state_head.to(device=anchor.device, dtype=anchor.dtype)
        # Reinitializing from V1 must discard any previously learned guidance.
        for block in self.cognitive.retrieval_blocks:
            anchor = block.q_proj.weight
            block.state_query.to(device=anchor.device, dtype=anchor.dtype)
            torch.nn.init.zeros_(block.state_query.weight)
        gate = self.cognitive.gate_fusion_blocks
        gate.state_gate.to(device=gate.proj.weight.device, dtype=gate.proj.weight.dtype)
        torch.nn.init.zeros_(gate.state_gate.weight)
        self.reset()

    @torch.no_grad()
    def _matched_cognitive_histories(self, keys, timesteps):
        """Wrong-content control with fixed slot count, ages and per-slot RMS.

        Donor selection and rescaling are diagnostic-only. They do not alter
        normal memory, learned weights, or the training/retrieval method.
        """
        overrides, matches = {}, []
        for index, (key, timestep) in enumerate(zip(keys, timesteps)):
            reference = [(t, f) for t, f in self.cognitive.bank.get(key, []) if t < int(timestep)]
            count = len(reference)
            entry = dict(episode_key=key, donor_key=None, slots=count,
                         read_timesteps=[t for t, _ in reference], donor_timesteps=[],
                         scale_factors=[], reference_rms=[], donor_rms=[], matched_rms=[])
            if not count:
                overrides[(key, int(timestep))] = []
                matches.append(entry)
                continue
            target = torch.stack([f for _, f in reference])
            target_rms = target.float().square().mean(dim=(-2, -1)).sqrt()
            for donor_key in keys[index + 1:] + keys[:index]:
                donor = [(t, f) for t, f in self.cognitive.bank.get(donor_key, []) if t < int(timestep)]
                if len(donor) < count:
                    continue
                donor = donor[-count:]
                source = torch.stack([f for _, f in donor]).to(target)
                source_rms = source.float().square().mean(dim=(-2, -1)).sqrt()
                if ((source_rms == 0) & (target_rms > 0)).any():
                    continue  # A zero vector cannot be scaled to a nonzero reference.
                scales = target_rms / torch.where(source_rms > 0, source_rms, torch.ones_like(source_rms))
                values = (source.float() * scales[:, None, None]).to(target.dtype)
                overrides[(key, int(timestep))] = [(t, value.detach().clone())
                    for (t, _), value in zip(reference, values)]
                entry.update(donor_key=donor_key, donor_timesteps=[t for t, _ in donor],
                    scale_factors=scales.cpu().tolist(), reference_rms=target_rms.cpu().tolist(),
                    donor_rms=source_rms.cpu().tolist(),
                    matched_rms=values.float().square().mean(dim=(-2, -1)).sqrt().cpu().tolist())
                break
            else:
                raise ValueError(f"No matched donor with {count} past slots and usable scale for {key}")
            matches.append(entry)
        return overrides, matches

    def forward(self, backbone_output, input_ids, context):
        if context.get("history_mode", "normal") != "normal":
            raise ValueError("Use cognitive_history_mode so perceptual history stays fixed")
        mode = context.get("cognitive_history_mode", "normal")
        if mode not in ("normal", "empty", "other_episode"):
            raise ValueError("Unknown cognitive history intervention")
        if self.training and mode != "normal":
            raise ValueError("History interventions are inference-only")
        keys = [tuple(k) if isinstance(k, list) else k for k in context["episode_keys"]]
        bank = self.cognitive
        overrides, matches = None, []
        if mode == "other_episode":
            if len(keys) < 2 or len(set(keys)) != len(keys):
                raise ValueError("Wrong history needs distinct concurrent episodes")
            overrides, matches = self._matched_cognitive_histories(keys, context["timesteps"])
        elif mode == "empty":
            overrides = {(key, int(t)): [] for key, t in zip(keys, context["timesteps"])}
        self.last_cognitive_history_control = dict(mode=mode, matches=matches)
        bank.records, bank.read_overrides = [], overrides
        try:
            output = super().forward(backbone_output, input_ids, context)
            if self.last_diagnostics is not None:
                self.last_diagnostics["cognitive_history_control"] = self.last_cognitive_history_control
            if bank.records:
                for name in ("state_logits", "state_feature", "retrieved_memory", "memory_gate"):
                    output[name] = torch.cat([r[name] for r in bank.records], dim=0).squeeze(1)
                output["state_event"] = [r["event"] for r in bank.records]
            else:  # Explicit whole-memory empty control in the original V1 adapter.
                h = backbone_output["backbone_features"]
                eos = (input_ids == self.eos_token_id) & backbone_output["backbone_attention_mask"]
                indices = torch.arange(h.shape[1], device=h.device).expand_as(eos).masked_fill(~eos, -1).max(1).values
                current = h[torch.arange(h.shape[0], device=h.device), indices]
                output["state_logits"] = bank.state_head(current)
                output["state_feature"] = output["state_logits"].sigmoid()
                output["retrieved_memory"] = current
                output["memory_gate"] = torch.ones_like(current)
                output["state_event"] = [False] * h.shape[0]
            return output
        finally:
            bank.read_overrides = None
            bank.records = None

    def runtime_state_dict(self):
        return dict(cognitive=self.cognitive.runtime_state_dict(),
            perceptual=deepcopy({name: getattr(self.perceptual, name)
                                for name in ("bank", "last_written", "merge_count")}))

    def load_runtime_state_dict(self, state):
        self.cognitive.load_runtime_state_dict(state["cognitive"])
        parameter = next(self.perceptual.parameters())
        restored = deepcopy(state["perceptual"])
        restored["bank"] = {key: [(t, f.detach().to(device=parameter.device).clone())
                                  for t, f in slots] for key, slots in restored["bank"].items()}
        for name, value in restored.items():
            setattr(self.perceptual, name, value)
