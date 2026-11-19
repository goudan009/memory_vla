"""PyTorch group-readout behavior, independent of any assumed robot layout."""

import pytest
import torch

from vlamemory.modulation import GroupActionModulator


def _tiny_modulator():
    return GroupActionModulator(memory_dim=6, action_hidden_dim=6, action_dim=6,
                               group_indices=((0, 2), (1, 3)), passthrough_indices=(4, 5))


def test_zero_initialized_modulator_preserves_original_action_head():
    module = _tiny_modulator()
    hidden = torch.tensor([[[1., 2., 3., 4., 5., 6.]]])
    actual = module.readout(hidden, torch.ones(1, 2, 6), lambda h: h, has_history=True)
    torch.testing.assert_close(actual, hidden, atol=0, rtol=0)


def test_group_context_changes_only_its_own_dims_and_keeps_padding():
    module = _tiny_modulator()
    with torch.no_grad():
        module.modulation_proj.weight[6, 0] = 3.
        module.modulation_proj.weight[7, 1] = 4.
    hidden = torch.tensor([[[1., 2., 3., 4., 5., 6.]]])
    contexts = torch.tensor([[[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]]])
    actual = module.readout(hidden, contexts, lambda h: h, has_history=True)
    torch.testing.assert_close(actual, torch.tensor([[[4., 6., 3., 4., 5., 6.]]]))


def test_empty_history_is_exact_identity_even_after_training_modulator():
    module = _tiny_modulator()
    with torch.no_grad():
        module.modulation_proj.bias.fill_(1)
    hidden = torch.tensor([[[1., 2., 3., 4., 5., 6.]]])
    actual = module.readout(hidden, torch.zeros(1, 2, 6), lambda h: h, has_history=False)
    torch.testing.assert_close(actual, hidden, atol=0, rtol=0)


def test_group_mapping_rejects_overlap_and_unassigned_dimensions():
    with pytest.raises(ValueError, match="overlap"):
        GroupActionModulator(memory_dim=6, action_hidden_dim=6, action_dim=6,
                             group_indices=((0, 1), (1, 2)), passthrough_indices=(3, 4, 5))
    with pytest.raises(ValueError, match="cover"):
        GroupActionModulator(memory_dim=6, action_hidden_dim=6, action_dim=6,
                             group_indices=((0, 1), (2, 3)), passthrough_indices=(4,))


def test_group_readout_remains_differentiable_and_accepts_embodiment_closure():
    module = _tiny_modulator()
    hidden = torch.ones(2, 3, 6, requires_grad=True)
    contexts = torch.ones(2, 2, 6)
    embodiment_gain = torch.tensor([1., 2.])[:, None, None]
    actual = module.readout(hidden, contexts, lambda h: h * embodiment_gain,
                            has_history=torch.tensor([True, False]))
    actual.sum().backward()
    assert module.modulation_proj.weight.grad.abs().sum() > 0
    torch.testing.assert_close(actual[1], torch.full((3, 6), 2.))
