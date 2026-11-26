"""CPU checks for selecting real engineering inputs; no GPU job here."""

from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch

from v2_groot_memoryvla.training_smoke import CachedBackbone, select_groups, smoke_inputs
from v2_groot_memoryvla.model.gr00t_n1d6 import Gr00tN1d6
from .test_state_memory import (adapter, observation, TinyBackbone, tiny_config, tiny_inputs,
                               v1_model, processing_gr00t_n1d6)


def test_real_groups_take_different_episodes_instead_of_adjacent_windows():
    dataset = SimpleNamespace(groups=[(0, 400010, 0), (0, 400010, 24),
                                     (0, 400020, 0), (0, 400030, 0)])
    assert select_groups(dataset) == [0, 2]


def test_insufficient_official_episodes_rejected():
    with pytest.raises(ValueError, match='two distinct official'):
        select_groups(SimpleNamespace(groups=[(0, 400010, 0), (0, 400010, 24)]))


def test_smoke_uses_training_sequence_protocol_and_resets_at_end():
    _, memory = adapter()
    memory.train()
    features, ids = observation()
    for t in range(3):
        inputs = smoke_inputs(dict(features=features, input_ids=ids, actions={}, episode='a', timestep=t),
                              sequence_start=t == 0, sequence_end=t == 2)
        memory(features, ids, inputs['memory_context'])
        if t == 1:
            assert len(memory.cognitive.bank['a']) == 2
    assert not memory.cognitive.bank and not memory.perceptual.bank


def test_cached_backbone_preserves_bf16_input_preparation_with_fp32_expert():
    with patch.object(v1_model, 'get_backbone_cls', lambda _: TinyBackbone), \
         patch.object(processing_gr00t_n1d6, 'Gr00tN1d6DataCollator', lambda **kwargs:None):
        model = Gr00tN1d6(tiny_config()).train()
    model.backbone = CachedBackbone(device='cpu')
    data = tiny_inputs()
    prepared, _ = model.prepare_input(data)
    assert prepared['backbone_features'].dtype == torch.bfloat16
    with torch.autocast('cpu', dtype=torch.bfloat16):
        outputs = model(data)
    outputs['loss'].backward()
    assert model.memory.cognitive.state_head.proj.weight.grad is not None
    assert not any(p.requires_grad for p in model.backbone.parameters())
