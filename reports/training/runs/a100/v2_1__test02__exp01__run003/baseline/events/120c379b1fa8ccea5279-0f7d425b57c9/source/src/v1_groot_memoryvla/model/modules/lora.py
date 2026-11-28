"""Linear LoRA, W x + alpha/r B A dropout(x), with stable base weight names."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    def __init__(self, base, rank, alpha, dropout):
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.rank, self.scaling = rank, alpha / rank
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, self.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        update = F.linear(F.linear(self.lora_dropout(x).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return base + update.to(base.dtype) * self.scaling


def install_lora(model, specification):
    """Exact paths are stored in config and recreated before HF loads weights."""
    for path in specification.get('modules', []):
        parent_path, _, leaf = path.rpartition('.')
        parent = model.get_submodule(parent_path) if parent_path else model
        base = parent._modules[leaf]
        if isinstance(base, LoRALinear):
            continue
        if not isinstance(base, nn.Linear):
            raise ValueError('LoRA target is not Linear: ' + path)
        parent._modules[leaf] = LoRALinear(base, specification['rank'], specification['alpha'], specification['dropout'])
