import pytest
import numpy as np
import torch

from v2_groot_memoryvla.task40_evaluation import binary_scores, compare, matched_wrong, off_history, reference_actions
from v2_groot_memoryvla.training_smoke import file_hash


def test_certified_manifest_string_path_is_hashable(tmp_path):
    import hashlib
    artifact = tmp_path / 'label.npz'
    artifact.write_bytes(b'recorded-state-label')
    manifest_entry = {'artifact': str(artifact)}
    assert file_hash(manifest_entry['artifact']) == hashlib.sha256(artifact.read_bytes()).hexdigest()


def runtime():
    return dict(cognitive=dict(bank={
        'target': [(1, torch.tensor([[3.,4.]])), (4, torch.tensor([[1.,2.]]))],
        'donor': [(2, torch.tensor([[1.,-1.]])), (3, torch.tensor([[2.,-1.]]))],
        'off': [(0, torch.tensor([[9.,8.]]))]},
        last_written={'target':4,'donor':4,'off':4}, merge_count={'off':2},
        event_states={'target':torch.tensor([1],dtype=torch.int8)},
        event_slots={'target':{4}}, eviction_count={'target':8}),
        perceptual=dict(bank={'target':[(4,torch.tensor([[7.]],dtype=torch.bfloat16))]},
                        last_written={'target':4}, merge_count={}))


def test_wrong_history_changes_content_preserves_times_slots_scale_and_other_banks():
    source = runtime()
    result, evidence = matched_wrong(source, 'target', 'donor')
    assert [t for t,_ in result['cognitive']['bank']['target']] == [1,4]
    assert evidence['rms_max_abs'] < 1e-6
    assert not torch.equal(result['cognitive']['bank']['target'][0][1],source['cognitive']['bank']['target'][0][1])
    assert torch.equal(result['perceptual']['bank']['target'][0][1],source['perceptual']['bank']['target'][0][1])
    source['cognitive']['bank']['donor'] = []
    with pytest.raises(ValueError, match='insufficient'):
        matched_wrong(source, 'target', 'donor')


def test_guidance_off_uses_own_rebuilt_cognition_with_target_perception():
    source = runtime()
    result = off_history(source, 'target', 'off')
    assert result['cognitive']['bank']['target'][0][0] == 0
    assert torch.equal(result['cognitive']['bank']['target'][0][1],torch.tensor([[9.,8.]]))
    assert result['cognitive']['merge_count']['target'] == 2
    assert 'target' not in result['cognitive']['event_states']
    assert 'target' not in result['cognitive']['event_slots']
    assert result['perceptual']['bank']['target'][0][1].dtype == torch.bfloat16
    assert source['cognitive']['event_slots']['target'] == {4}


def test_state_scores_expose_constant_negative_failure_on_imbalanced_labels():
    truth = [0]*99+[1]
    baseline = binary_scores(truth, [0]*100)
    assert baseline['accuracy'] == .99 and baseline['f1'] == 0
    perfect = binary_scores(truth, truth)
    assert perfect['f1'] == 1 and perfect['tp'] == 1 and perfect['tn'] == 99


def test_action_comparison_excludes_padding_and_keeps_executed_horizon():
    original = np.zeros((1,50,4), dtype=np.float32)
    alternative = original.copy()
    alternative[:,:,2:] = 100
    alternative[:,8:32,0] = 1
    result = compare({'normalized_actions':original}, {'normalized_actions':alternative},
                     np.zeros((32,4)), np.ones((32,4)), {'base':(0,2)})
    assert result['normalized_actions']['mae'] == .375
    assert result['action_parts']['base']['mae_first8'] == 0
    assert result['action_parts']['base']['mae'] == .375


def test_bf16_cached_reference_is_usable_by_action_diagnostics():
    actions = {'action': torch.full((1, 32, 4), 2., dtype=torch.bfloat16),
               'action_mask': torch.ones((1, 32, 4), dtype=torch.bfloat16)}
    reference, mask = reference_actions(actions, 0)
    normal = {'normalized_actions': np.full((1, 50, 4), 2., dtype=np.float32)}
    alternative = {'normalized_actions': np.zeros((1, 50, 4), dtype=np.float32)}
    result = compare(normal, alternative, reference, mask, {'base': (0, 2)})
    assert result['action_parts']['base']['normal_reference_mae'] == 0.
    assert result['action_parts']['base']['alternative_reference_mae'] == 2.
