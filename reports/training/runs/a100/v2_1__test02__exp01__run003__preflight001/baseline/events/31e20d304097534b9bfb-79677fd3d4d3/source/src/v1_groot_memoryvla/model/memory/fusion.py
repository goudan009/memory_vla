"""Opt-in, zero-initialized cognitive residuals for Experiment 10."""

import torch
from torch import nn
from torch.nn import functional as F


RESIDUAL_LIMIT = 0.1


def _init_fusion(module, output):
    # Also initialize these when HF loads them as missing from a G0 checkpoint.
    for child in module.modules():
        child._memory_v1 = True
    output._memory_zero_output = True
    nn.init.zeros_(output.weight)


class CognitiveGatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(2 * dim, dim)
        self.value = nn.Linear(dim, dim, bias=False)
        _init_fusion(self, self.value)

    def forward(self, current, message):
        # A bias-free value projection keeps an empty/zero message inert.
        current = F.layer_norm(current, (current.shape[-1],))
        message = F.layer_norm(message, (message.shape[-1],))
        gate = self.gate(torch.cat((current, message), dim=-1)).sigmoid()
        return RESIDUAL_LIMIT * gate * self.value(message).tanh()


class CognitiveActionAttention(nn.Module):
    def __init__(self, action_dim, memory_dim):
        super().__init__()
        width = min(256, action_dim)
        self.query = nn.Linear(action_dim, width, bias=False)
        self.key = nn.Linear(memory_dim, width, bias=False)
        self.value = nn.Linear(memory_dim, width, bias=False)
        self.output = nn.Linear(width, action_dim, bias=False)
        _init_fusion(self, self.output)

    def forward(self, hidden_states, *, slots, mask, action_tokens):
        actions = hidden_states[:, -action_tokens:]
        query = self.query(F.layer_norm(actions, (actions.shape[-1],)))
        slots = slots.masked_fill(~mask[..., None], 0)
        slots = F.layer_norm(slots, (slots.shape[-1],))
        # Give empty rows a zero key/value to keep attention defined, then
        # explicitly suppress their residual.
        valid = mask.any(dim=-1, keepdim=True)
        safe_mask = torch.cat((mask[:, :1] | ~valid, mask[:, 1:]), dim=-1)
        attended = F.scaled_dot_product_attention(
            query.unsqueeze(1), self.key(slots).unsqueeze(1), self.value(slots).unsqueeze(1),
            attn_mask=safe_mask[:, None, None, :],
        ).squeeze(1)
        residual = RESIDUAL_LIMIT * self.output(attended).tanh()
        residual = residual.masked_fill(~valid[..., None], 0)
        return torch.cat((hidden_states[:, :-action_tokens], actions + residual), dim=1)
