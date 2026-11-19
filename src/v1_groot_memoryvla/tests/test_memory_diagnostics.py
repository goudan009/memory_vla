from copy import deepcopy

import torch

from v1_groot_memoryvla.tests.test_history_controls import populated
from v1_groot_memoryvla.tests.test_memory_v1 import features


def test_diagnostics_preserve_outputs_and_capture_actual_donor_slots():
    memory=populated(); reference=deepcopy(memory)
    sample,ids=features(2)
    context=dict(episode_keys=['a','b'],timesteps=[1,1],history_mode='other_episode')
    with torch.no_grad():
        expected=reference(sample,ids,context)
        actual=memory(sample,ids,dict(context,memory_diagnostics=True))
    assert torch.equal(actual.backbone_features,expected.backbone_features)
    rows=memory.last_diagnostics['observations']
    assert len(rows)==2 and rows[0]['read_source']=='b' and rows[0]['write_source']=='a'
    for name in ['perceptual','cognitive']:
        bank=rows[0][name]
        assert bank['history_timesteps']==[0] and bank['slots_after_write']==2
        assert 0<=bank['gate_current_mean']<=1
    attention=rows[0]['cognitive']['retrieval_attention']
    assert attention['slot_timesteps']==[0]
    assert abs(sum(attention['slot_mass'])-1)<1e-5
    assert rows[0]['residual_rms']>0
    for bank,other in [(memory.perceptual,reference.perceptual),(memory.cognitive,reference.cognitive)]:
        assert bank.last_written==other.last_written
        for key in bank.bank:
            assert all(torch.equal(a[1],b[1]) for a,b in zip(bank.bank[key],other.bank[key]))


def test_empty_control_reports_no_read_and_no_write_without_stale_diagnostics():
    memory=populated();sample,ids=features(2)
    context=dict(episode_keys=['a','b'],timesteps=[1,1],history_mode='empty')
    with torch.no_grad():memory(sample,ids,dict(context,memory_diagnostics=True))
    rows=memory.last_diagnostics['observations']
    assert all(r['history_read_enabled'] is False and r['residual_rms']==0 for r in rows)
    assert all(r['perceptual']['history_timesteps']==[] for r in rows)
    assert memory.perceptual.last_written=={'a':0,'b':0}
    with torch.no_grad():memory(sample,ids,context)
    assert memory.last_diagnostics is None
