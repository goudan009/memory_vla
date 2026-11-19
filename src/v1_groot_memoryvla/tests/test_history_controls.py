"""Inference interventions change only retrieval, preserving each stream's writes."""
from copy import deepcopy

import pytest
import torch

from v1_groot_memoryvla.model.memory.adapter import MemoryAdapter
from v1_groot_memoryvla.tests.test_memory_v1 import features


def populated():
    torch.manual_seed(81)
    memory = MemoryAdapter(vlm_dim=32, perceptual_dim=32, conditioning='residual',
        cognitive_qk_norm='layernorm', residual_identity_init=True,
        cognitive_residual_projection=True, residual_readout='value').eval()
    with torch.no_grad():
        memory.perceptual_residual.weight.copy_(torch.eye(32)*.1)
        memory.cognitive_residual.weight.copy_(torch.eye(32)*.1)
        for bank, n in ((memory.perceptual, 12), (memory.cognitive, 1)):
            bank._write('a', 0, torch.randn(n, 32))
            bank._write('b', 0, torch.randn(n, 32)*3)
    return memory


def test_other_episode_history_matches_swapped_read_and_original_writes():
    memory=populated()
    reference=deepcopy(memory)
    sample,ids=features(2)
    for bank in (reference.perceptual, reference.cognitive):
        bank.bank['a'],bank.bank['b']=bank.bank['b'],bank.bank['a']
    expected=reference(sample,ids,dict(episode_keys=['a','b'],timesteps=[1,1]))
    original={label: list(bank.bank['a']) for label,bank in
              [('per',memory.perceptual),('cog',memory.cognitive)]}
    actual=memory(sample,ids,dict(episode_keys=['a','b'],timesteps=[1,1],history_mode='other_episode'))
    torch.testing.assert_close(actual.backbone_features,expected.backbone_features,rtol=0,atol=0)
    for label,bank in [('per',memory.perceptual),('cog',memory.cognitive)]:
        assert torch.equal(bank.bank['a'][0][1],original[label][0][1])
        assert len(bank.bank['a'])==len(bank.bank['b'])==2
        assert bank.last_written=={'a':1,'b':1}
    assert memory.last_history_control['mode']=='other_episode'
    assert memory.last_history_control['donor_keys']==['b','a']


def test_other_episode_snapshot_is_independent_of_batch_processing_order():
    memory=populated(); reverse=deepcopy(memory)
    sample,ids=features(2)
    actual=memory(sample,ids,dict(episode_keys=['a','b'],timesteps=[1,1],history_mode='other_episode'))
    flipped={k:v.flip(0) for k,v in sample.items()}
    expected=reverse(flipped,ids.flip(0),dict(episode_keys=['b','a'],timesteps=[1,1],history_mode='other_episode'))
    torch.testing.assert_close(actual.backbone_features,expected.backbone_features.flip(0),rtol=0,atol=0)


def test_empty_history_control_keeps_current_features_and_does_not_write():
    memory=populated();sample,ids=features(2)
    actual=memory(sample,ids,dict(episode_keys=['a','b'],timesteps=[1,1],history_mode='empty'))
    assert torch.equal(actual.backbone_features,sample.backbone_features)
    assert all(bank.last_written=={'a':0,'b':0} for bank in (memory.perceptual,memory.cognitive))


def test_controls_reject_unknown_modes_and_unpaired_wrong_history():
    memory=populated();sample,ids=features(1)
    for mode in ('other_episode','nonsense'):
        with pytest.raises(ValueError):
            memory(sample,ids,dict(episode_keys=['a'],timesteps=[1],history_mode=mode))
