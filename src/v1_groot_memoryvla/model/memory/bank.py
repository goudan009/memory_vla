"""MemoryVLA operators adapted from references_code/memoryvla/vla/memory_vla.py.

Retains its single-head cross attention, K-only time encoding, gate and
adjacent cosine merge. Runtime keys replace the reference's global episode
counter so independent evaluation clients can share one policy server.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size),
                                 nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, t):
        parameter = next(self.parameters())
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(
            half, device=parameter.device, dtype=torch.float32) / half)
        args = t.to(parameter.device).float()[:, None] * frequencies[None]
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding.to(parameter.dtype))


class CrossTransformerBlock(nn.Module):
    def __init__(self, feature_dim, qk_norm="none"):
        super().__init__()
        if qk_norm not in ("none", "layernorm"):
            raise ValueError(f"Unsupported qk_norm: {qk_norm}")
        self.qk_norm = qk_norm
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(nn.Linear(feature_dim, feature_dim * 4), nn.GELU(),
                                 nn.Linear(feature_dim * 4, feature_dim))
        self.ffn_norm = nn.LayerNorm(feature_dim)

    def forward(self, query, k, v, return_value_message=False, capture_attention=False):
        # GR00T EOS magnitudes can saturate the reference's post-norm attention.
        # Normalize only retrieval inputs, including K's time encoding. Preserve
        # the original values and residual; no new learned checkpoint tensors.
        q_input = F.layer_norm(query, (query.shape[-1],)) if self.qk_norm == "layernorm" else query
        k_input = F.layer_norm(k, (k.shape[-1],)) if self.qk_norm == "layernorm" else k
        values = F.linear(v, self.v_proj.weight) if return_value_message else self.v_proj(v)
        q_projected, k_projected = self.q_proj(q_input), self.k_proj(k_input)
        attended = F.scaled_dot_product_attention(q_projected, k_projected,
                                                  values, dropout_p=0.0, is_causal=False)
        self.last_attention = None
        if capture_attention:
            with torch.no_grad():
                scores = q_projected.detach().float() @ k_projected.detach().float().transpose(-2,-1)
                weights = (scores / math.sqrt(q_projected.shape[-1])).softmax(-1)
                self.last_attention = weights.mean(dim=(0,1)).cpu().tolist()
        message = attended
        if return_value_message:
            # The residual readout must vanish for zero historical values. Add
            # V's bias only to the regular transformer path used for fusion.
            attended = attended + self.v_proj.bias.to(attended.dtype)
        x = self.attn_norm(query + attended)
        result = self.ffn_norm(x + self.ffn(x))
        return (result, message) if return_value_message else result


class BottleneckSE(nn.Module):
    def __init__(self, c_in, c_mid, c_out):
        super().__init__()
        self.reduce = nn.Conv2d(c_in, c_mid, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.excite = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_mid, c_mid // 16, 1),
                                    nn.ReLU(), nn.Conv2d(c_mid // 16, c_mid, 1), nn.Sigmoid())
        self.expand = nn.Conv2d(c_mid, c_out, 1, bias=False)

    def forward(self, x):
        b, n, c = x.shape
        side = math.isqrt(n)
        if side * side != n:
            raise ValueError(f"BottleneckSE needs one square camera grid, got {n} tokens")
        x = x.reshape(b, side, side, c).permute(0, 3, 1, 2)
        z = self.act(self.reduce(x))
        return self.expand(z * self.excite(z)).flatten(2).transpose(1, 2)


class GateFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim)
        self.proj._memory_gate_proj = True
        nn.init.normal_(self.proj.weight, std=1e-3)
        nn.init.normal_(self.proj.bias, std=1e-3)

    def forward(self, current, retrieved, value_message=None, capture=False):
        scale = torch.sigmoid(self.proj(torch.cat((current, retrieved), dim=-1)))
        self.last_gate_mean = float(scale.detach().float().mean()) if capture else None
        fused = scale * current + (1 - scale) * retrieved
        return fused if value_message is None else (fused, (1 - scale) * value_message)


class MemoryBank(nn.Module):
    def __init__(self, dim, capacity=16, retrieval_layers=2, qk_norm="none"):
        super().__init__()
        self.capacity = capacity
        self.retrieval_blocks = nn.ModuleList(CrossTransformerBlock(dim, qk_norm) for _ in range(retrieval_layers))
        self.timestep_encoder = TimestepEmbedder(dim, dim // 4)
        self.gate_fusion_blocks = GateFusion(dim)
        self.bank = {}
        self.merge_count = {}
        self.last_written = {}

    def reset(self, keys=None):
        if keys is None:
            self.bank.clear()
            self.merge_count.clear()
            self.last_written.clear()
        else:
            for key in keys:
                self.bank.pop(key, None)
                self.merge_count.pop(key, None)
                self.last_written.pop(key, None)

    def read_fuse_write(self, current, key, timestep, return_value_message=False, history_override=None,
                        capture_diagnostics=False, capture_attention=False):
        """Read strictly past features, fuse, then detach/clone the fused write."""
        history = self.bank.get(key, []) if history_override is None else history_override
        self.last_diagnostics = None
        if capture_diagnostics:
            self.last_diagnostics = dict(history_timesteps=[t for t,_ in history],
                history_slots=len(history), gate_current_mean=None,
                timestamp_semantics='槽位保留的代表时间戳；合并后不是完整原始时间范围')
        if any(t >= timestep for t, _ in history):
            raise ValueError('Memory retrieval may read only strictly past entries')
        if timestep <= self.last_written.get(key, -1):
            raise ValueError(f"Non-increasing memory timestep for {key}: {timestep}")
        if history:
            n, d = current.shape[-2:]
            memory = torch.stack([feat for _, feat in history]).reshape(1, -1, d)
            times = torch.tensor([t for t, _ in history], device=current.device)
            pe = self.timestep_encoder(times).unsqueeze(0).repeat_interleave(n, dim=1)
            retrieved = current
            for index, block in enumerate(self.retrieval_blocks):
                if return_value_message and index == len(self.retrieval_blocks)-1:
                    retrieved, message = block(retrieved, memory + pe, memory, return_value_message=True,
                                                capture_attention=capture_attention)
                else:
                    retrieved = block(retrieved, memory + pe, memory,capture_attention=capture_attention)
            if return_value_message:
                fused, message = self.gate_fusion_blocks(current, retrieved, message,capture=capture_diagnostics)
            else:
                fused = self.gate_fusion_blocks(current, retrieved,capture=capture_diagnostics)
            if capture_diagnostics:
                self.last_diagnostics['gate_current_mean'] = self.gate_fusion_blocks.last_gate_mean
                if capture_attention:
                    token_weights = self.retrieval_blocks[-1].last_attention
                    self.last_diagnostics['retrieval_attention'] = dict(
                        slot_timesteps=[t for t,_ in history],
                        slot_mass=[sum(token_weights[i*n:(i+1)*n]) for i in range(len(history))],
                        semantics='末层Q/K以FP32重算softmax后按槽位聚合；非SDPA内核逐位注意力，不证明因果')
        else:
            # Gate(x,x) is mathematically x; bypass rounding so the empty bank
            # has bitwise zero residual, including in BF16.
            fused = current
            message = torch.zeros_like(current)
        self._write(key, timestep, fused.squeeze(0))
        if capture_diagnostics:
            self.last_diagnostics.update(slots_after_write=len(self.bank[key]),
                merges_after_write=self.merge_count.get(key,0))
        return (fused, message) if return_value_message else fused

    @torch.no_grad()
    def _write(self, key, timestep, fused):
        history = self.bank.setdefault(key, [])
        history.append((int(timestep), fused.detach().clone()))
        self.last_written[key] = int(timestep)
        while len(history) > self.capacity:
            similarities = torch.stack([F.cosine_similarity(a[1], b[1], dim=-1).mean()
                                         for a, b in zip(history, history[1:])])
            index = int(similarities.argmax())
            t, first = history[index]
            history[index] = (t, ((first + history[index + 1][1]) * 0.5).detach().clone())
            history.pop(index + 1)
            self.merge_count[key] = self.merge_count.get(key, 0) + 1
