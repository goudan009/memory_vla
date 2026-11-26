"""V1 read/fuse/write ordering with V2 state-conditioned Q, gate and retention."""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from v1_groot_memoryvla.model.memory.bank import CrossTransformerBlock, GateFusion, MemoryBank
from .state import StateHead


def zero_projection(inputs, outputs):
    layer = nn.Linear(inputs, outputs, bias=False)
    layer._memory_v1 = True
    layer._memory_zero_output = True  # Also respected by HF missing-weight initialization.
    nn.init.zeros_(layer.weight)
    return layer


class StateQueryBlock(CrossTransformerBlock):
    def __init__(self, dim, states, qk_norm):
        super().__init__(dim, qk_norm)
        self.state_query = zero_projection(states, dim)
        self.state_feature = None

    def forward(self, query, k, v, return_value_message=False, capture_attention=False):
        if self.state_feature is None:
            return super().forward(query, k, v, return_value_message, capture_attention)
        q_input = F.layer_norm(query, (query.shape[-1],)) if self.qk_norm == "layernorm" else query
        k_input = F.layer_norm(k, (k.shape[-1],)) if self.qk_norm == "layernorm" else k
        # State is added AFTER q_proj; the query residual below stays exactly V1.
        q = self.q_proj(q_input) + self.state_query(self.state_feature)
        k = self.k_proj(k_input)
        values = F.linear(v, self.v_proj.weight) if return_value_message else self.v_proj(v)
        attended = F.scaled_dot_product_attention(q, k, values, dropout_p=0., is_causal=False)
        self.last_attention = None
        if capture_attention:
            with torch.no_grad():
                weights = ((q.detach().float() @ k.detach().float().transpose(-2, -1))
                           / math.sqrt(q.shape[-1])).softmax(-1)
                self.last_attention = weights.mean(dim=(0, 1)).cpu().tolist()
        message = attended
        if return_value_message:
            attended = attended + self.v_proj.bias.to(attended.dtype)
        x = self.attn_norm(query + attended)
        result = self.ffn_norm(x + self.ffn(x))
        return (result, message) if return_value_message else result


class StateGateFusion(GateFusion):
    def __init__(self, dim, states):
        super().__init__(dim)
        self.state_gate = zero_projection(states, dim)
        self.state_feature = None
        self.retrieved_memory = None
        self.memory_gate = None

    def forward(self, current, retrieved, value_message=None, capture=False):
        gate_logits = self.proj(torch.cat((current, retrieved), dim=-1))
        if self.state_feature is not None:
            gate_logits = gate_logits + self.state_gate(self.state_feature)
        gate = torch.sigmoid(gate_logits)
        self.retrieved_memory, self.memory_gate = retrieved, gate
        self.last_gate_mean = float(gate.detach().float().mean()) if capture else None
        fused = gate * current + (1 - gate) * retrieved
        return fused if value_message is None else (fused, (1 - gate) * value_message)


class CognitiveMemoryBank(MemoryBank):
    def __init__(self, dim, labels, capacity=64, retrieval_layers=2, qk_norm="layernorm",
                 update_fused=True, guidance_enabled=True, event_low=.25, event_high=.75,
                 recent_slots=2):
        if capacity < 1 or retrieval_layers < 1:
            raise ValueError("Memory capacity and retrieval layers must be positive")
        if not 0 <= event_low < event_high <= 1 or not 0 <= recent_slots <= capacity:
            raise ValueError("Invalid hysteresis thresholds or ordinary-history reserve")
        super().__init__(dim, capacity, retrieval_layers, qk_norm, update_fused)
        self.state_head = StateHead(dim, labels)
        states = len(self.state_head.labels)
        self.retrieval_blocks = nn.ModuleList(StateQueryBlock(dim, states, qk_norm)
                                             for _ in range(retrieval_layers))
        self.gate_fusion_blocks = StateGateFusion(dim, states)
        self.guidance_enabled = guidance_enabled
        self.event_low, self.event_high, self.recent_slots = event_low, event_high, recent_slots
        self.event_states, self.event_slots, self.eviction_count = {}, {}, {}
        self.records = None  # Adapter-owned collector; no unbounded stream graphs.
        self.last_record = None
        self.read_overrides = None
        self._pending_event = False

    def reset(self, keys=None):
        keys = None if keys is None else list(keys)
        super().reset(keys)
        for store in (self.event_states, self.event_slots, self.eviction_count):
            if keys is None:
                store.clear()
            else:
                for key in keys:
                    store.pop(key, None)

    @torch.no_grad()
    def _event(self, key, probabilities):
        p = probabilities.detach().float().reshape(-1).cpu()
        previous = self.event_states.get(key, torch.full_like(p, -1, dtype=torch.int8))
        state = previous.clone()
        state[p <= self.event_low] = 0
        state[p >= self.event_high] = 1
        changed = bool(((previous >= 0) & (state != previous)).any())
        self.event_states[key] = state
        return changed

    def read_fuse_write(self, current, key, timestep, **kwargs):
        # Reject chronology errors before changing hysteresis/runtime state.
        if self.read_overrides is not None:
            kwargs["history_override"] = self.read_overrides[(key, int(timestep))]
        history = kwargs.get("history_override")
        history = self.bank.get(key, []) if history is None else history
        if any(t >= timestep for t, _ in history) or timestep <= self.last_written.get(key, -1):
            raise ValueError("Cognitive memory requires strictly past reads and increasing writes")
        logits = self.state_head(current)
        probabilities = logits.sigmoid()
        state = probabilities if self.guidance_enabled else None
        self._pending_event = self._event(key, probabilities) if self.guidance_enabled else False
        for block in self.retrieval_blocks:
            block.state_feature = state
        gate = self.gate_fusion_blocks
        gate.state_feature = state
        gate.retrieved_memory, gate.memory_gate = current, torch.ones_like(current)
        try:
            result = super().read_fuse_write(current, key, timestep, **kwargs)
            self.last_record = dict(state_logits=logits, state_feature=probabilities,
                                    retrieved_memory=gate.retrieved_memory,
                                    memory_gate=gate.memory_gate, event=self._pending_event)
            if self.records is not None:
                self.records.append(self.last_record)
            if self.last_diagnostics is not None:
                self.last_diagnostics.update(event=self._pending_event,
                    event_timesteps=sorted(self.event_slots.get(key, set())),
                    evictions_after_write=self.eviction_count.get(key, 0))
            return result
        finally:
            for block in self.retrieval_blocks:
                block.state_feature = None
            gate.state_feature = None
            gate.retrieved_memory = gate.memory_gate = None
            self._pending_event = False

    @torch.no_grad()
    def _write(self, key, timestep, fused):
        if not self.guidance_enabled:
            # Exact V1 append + adjacent cosine merge, including a full bank.
            super()._write(key, timestep, fused)
            retained = {t for t, _ in self.bank[key]}
            self.event_slots[key] = self.event_slots.get(key, set()) & retained
            return
        history = self.bank.setdefault(key, [])
        events = self.event_slots.setdefault(key, set())
        history.append((int(timestep), fused.detach().clone()))
        self.last_written[key] = int(timestep)
        if self._pending_event:
            events.add(int(timestep))
        while len(history) > self.capacity:
            # Prefer latest events, but reserve recent ordinary observations.
            # Remaining capacity holds recent history. No new slots/network/loss.
            event_budget = self.capacity - self.recent_slots
            protected_events = set(sorted(events, reverse=True)[:event_budget])
            ordinary = [t for t, _ in history if t not in events]
            protected = protected_events | set(ordinary[-self.recent_slots:] if self.recent_slots else [])
            drop = next(i for i, (t, _) in enumerate(history) if t not in protected)
            t, _ = history.pop(drop)
            events.discard(t)
            self.eviction_count[key] = self.eviction_count.get(key, 0) + 1

    def runtime_state_dict(self):
        """Separate detached runtime snapshot, not included in neural checkpoints."""
        return deepcopy({name: getattr(self, name) for name in (
            "bank", "merge_count", "last_written", "event_states", "event_slots", "eviction_count")})

    def load_runtime_state_dict(self, state):
        parameter = next(self.parameters())
        restored = deepcopy(state)
        for key, slots in restored["bank"].items():
            if len(slots) > self.capacity:
                raise ValueError("Runtime snapshot exceeds memory capacity")
            restored["bank"][key] = [(t, f.detach().to(device=parameter.device).clone())
                                     for t, f in slots]
        self.reset()
        for name in ("bank", "merge_count", "last_written", "event_states", "event_slots", "eviction_count"):
            setattr(self, name, restored[name])
