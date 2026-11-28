"""Unchanged V1 MemoryVLA operators (model/memory/bank.py), isolated for CPU testing."""
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


