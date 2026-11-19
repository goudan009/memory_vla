"""Residual adapters start as an identity even when history is nonempty."""
import torch
import pytest

from v1_groot_memoryvla.model.memory.adapter import MemoryAdapter
from v1_groot_memoryvla.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from v1_groot_memoryvla.tests.test_memory_v1 import features


def adapter():
    return MemoryAdapter(vlm_dim=32, perceptual_dim=32, conditioning="residual",
                         cognitive_qk_norm="layernorm", residual_identity_init=True).eval()


def test_nonempty_history_preserves_all_original_features_at_initialization():
    torch.manual_seed(42)
    memory = adapter()
    original, ids = features(3)
    for t in range(3):
        sample = {k: v[t:t + 1] for k, v in original.items()}
        result = memory(sample, ids[t:t + 1], {"episode_keys": ["a"], "timesteps": [t]})
        assert torch.equal(result.backbone_features, sample["backbone_features"])
        assert result.image_mask is sample["image_mask"]
        assert result.backbone_attention_mask is sample["backbone_attention_mask"]
    assert len(memory.cognitive.bank["a"]) == 3


def test_huggingface_missing_weight_initialization_keeps_zero_residual():
    memory = adapter()
    for module in memory.modules():
        Gr00tN1d6._init_weights(None, module)
    assert torch.count_nonzero(memory.perceptual_residual.weight) == 0
    assert memory.cognitive_residual_scale.item() == 0


def test_retrieval_gradients_wake_up_after_identity_outputs_begin_learning():
    torch.set_num_threads(2)
    torch.manual_seed(43)
    memory = adapter()
    optimizer = torch.optim.AdamW(memory.parameters(), lr=1e-3)
    original, ids = features(3)
    target = torch.randn_like(original.backbone_features)
    for step in range(4):
        memory.reset()
        optimizer.zero_grad(set_to_none=True)
        outputs = []
        for t in range(3):
            sample = {k: v[t:t + 1] for k, v in original.items()}
            outputs.append(memory(sample, ids[t:t + 1], {
                "episode_keys": ["a"], "timesteps": [t]}).backbone_features)
        (torch.cat(outputs) - target).square().mean().backward()
        if step == 3:
            for bank in (memory.perceptual, memory.cognitive):
                for block in bank.retrieval_blocks:
                    for name in ("q_proj", "k_proj", "v_proj"):
                        grad = getattr(block, name).weight.grad
                        assert grad is not None and torch.isfinite(grad).all()
                        assert grad.norm() > 0
        optimizer.step()


def test_legacy_residual_state_dictionary_has_no_new_parameters():
    legacy = MemoryAdapter(vlm_dim=32, perceptual_dim=32, conditioning="residual")
    assert "cognitive_residual_scale" not in legacy.state_dict()


@pytest.mark.parametrize("scope", ["eos", "text"])
def test_projected_cognitive_residual_starts_identical_and_learns_a_direction(scope):
    torch.manual_seed(43)
    memory = MemoryAdapter(vlm_dim=32, perceptual_dim=32, conditioning="residual",
        cognitive_qk_norm="layernorm", residual_identity_init=True,
        cognitive_residual_projection=True, cognitive_residual_scope=scope).eval()
    for module in memory.modules():
        Gr00tN1d6._init_weights(None, module)
    assert torch.count_nonzero(memory.cognitive_residual.weight) == 0
    assert "cognitive_residual_scale" not in memory.state_dict()
    original, ids = features(3)
    target = torch.randn_like(original.backbone_features)
    optimizer = torch.optim.AdamW(memory.parameters(), lr=1e-3)
    for step in range(4):
        memory.reset()
        optimizer.zero_grad(set_to_none=True)
        outputs = []
        for t in range(3):
            sample = {k: v[t:t+1] for k, v in original.items()}
            output = memory(sample, ids[t:t+1], {"episode_keys": ["a"], "timesteps": [t]})
            if step == 0:
                assert torch.equal(output.backbone_features, sample["backbone_features"])
            outputs.append(output.backbone_features)
        (torch.cat(outputs)-target).square().mean().backward()
        if step == 3:
            assert memory.cognitive.retrieval_blocks[0].q_proj.weight.grad.norm() > 0
            assert memory.cognitive_residual.weight.grad.norm() > 0
        optimizer.step()


def test_cognitive_projection_requires_identity_residual_interface():
    with pytest.raises(ValueError, match="identity"):
        MemoryAdapter(vlm_dim=32, perceptual_dim=32, cognitive_residual_projection=True)


def test_text_memory_changes_valid_text_and_preserves_images_and_padding():
    torch.manual_seed(51)
    memory = MemoryAdapter(vlm_dim=32, perceptual_dim=32, conditioning="residual",
        residual_identity_init=True, cognitive_residual_projection=True,
        cognitive_residual_scope="text").eval()
    original, ids = features(1)
    original.backbone_attention_mask[:, -1] = False
    with torch.no_grad():
        memory.cognitive_residual.weight.copy_(torch.eye(32) * .1)
    memory(original, ids, {"episode_keys": ["a"], "timesteps": [0]})
    result = memory(original, ids, {"episode_keys": ["a"], "timesteps": [1]})
    difference = result.backbone_features - original.backbone_features
    assert torch.count_nonzero(difference[original.image_mask]) == 0
    assert torch.count_nonzero(difference[~original.backbone_attention_mask]) == 0
    text = original.backbone_attention_mask & ~original.image_mask
    assert difference[text].norm() > 0
    torch.testing.assert_close(difference[0, 0], difference[0, 13])
