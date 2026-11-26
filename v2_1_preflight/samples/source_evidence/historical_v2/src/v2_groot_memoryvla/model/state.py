"""Current-observation state prediction and loss-only physical supervision."""

import torch
from torch import nn
from torch.nn import functional as F


PHYSICAL_LABELS = ("door_open", "bag_inside", "started")


def validate_labels(labels):
    labels = tuple(labels)
    if not labels or len(set(labels)) != len(labels) or set(labels) - set(PHYSICAL_LABELS):
        raise ValueError("Select unique, audited physical state labels from " + str(PHYSICAL_LABELS))
    return labels


class StateHead(nn.Module):
    """One small head on the *pre-retrieval* current user EOS, never on actions."""

    def __init__(self, dim, labels):
        super().__init__()
        self.labels = validate_labels(labels)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, len(self.labels))

    def forward(self, current):
        return self.proj(self.norm(current))


def masked_state_loss(logits, targets, known):
    """Unknown entries may contain NaNs: select BEFORE computing BCE."""
    if logits.shape != targets.shape or logits.shape != known.shape:
        raise ValueError("State logits, targets and known mask must have identical shapes")
    if known.dtype != torch.bool:
        raise ValueError("State known mask must be boolean")
    targets = targets.to(device=logits.device, dtype=torch.float32)
    known = known.to(logits.device)
    selected = targets[known]
    if not torch.isfinite(selected).all() or ((selected < 0) | (selected > 1)).any():
        raise ValueError("Known physical state targets must be finite probabilities in [0, 1]")
    if not known.any():
        return logits.float().sum() * 0
    return F.binary_cross_entropy_with_logits(logits.float()[known], selected)
