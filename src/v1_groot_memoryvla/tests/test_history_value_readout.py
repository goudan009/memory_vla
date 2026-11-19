"""History readout must not learn a query-only correction when values are zero."""
import pytest
import torch

from v1_groot_memoryvla.model.memory.adapter import MemoryAdapter
from v1_groot_memoryvla.model.memory.bank import CrossTransformerBlock, MemoryBank
from v1_groot_memoryvla.tests.test_memory_v1 import features


def test_attention_value_message_is_zero_despite_nonzero_projection_bias():
    torch.manual_seed(71)
    block = CrossTransformerBlock(32, qk_norm="layernorm")
    with torch.no_grad():
        block.v_proj.bias.fill_(3.)
    query, keys = torch.randn(1, 2, 32), torch.randn(1, 5, 32)
    result, message = block(query, keys, torch.zeros_like(keys), return_value_message=True)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(message) == 0


def test_value_readout_uses_two_attentions_and_writes_fused_history(monkeypatch):
    torch.manual_seed(72)
    bank = MemoryBank(32, retrieval_layers=2, qk_norm="layernorm")
    ordinary = MemoryBank(32, retrieval_layers=2, qk_norm="layernorm")
    ordinary.load_state_dict(bank.state_dict())
    query = torch.randn(1, 1, 32)
    for module in (bank, ordinary):
        module._write("a", 0, torch.randn(1, 32))
    ordinary.bank["a"] = list(bank.bank["a"])
    calls = []
    original = torch.nn.functional.scaled_dot_product_attention
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", counted)
    fused, value = bank.read_fuse_write(query, "a", 1, return_value_message=True)
    assert len(calls) == 2
    reference = ordinary.read_fuse_write(query, "a", 1)
    torch.testing.assert_close(fused, reference)
    torch.testing.assert_close(bank.bank["a"][-1][1], reference.squeeze(0))
    assert value.norm() > 0
    assert not bank.bank["a"][-1][1].requires_grad


@pytest.mark.parametrize("scope", ["eos", "text"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_history_values_preserve_baseline_even_with_trained_outputs(scope, dtype):
    torch.manual_seed(73)
    memory = MemoryAdapter(vlm_dim=32, perceptual_dim=32, conditioning="residual",
        residual_identity_init=True, cognitive_residual_projection=True,
        cognitive_residual_scope=scope, residual_readout="value").to(dtype).eval()
    sample, ids = features(1)
    sample = type(sample)({k: v.to(dtype) if v.is_floating_point() else v for k,v in sample.items()})
    with torch.no_grad():
        memory.perceptual_residual.weight.copy_(torch.eye(32))
        memory.cognitive_residual.weight.copy_(torch.eye(32))
        for bank, n in ((memory.perceptual, 12), (memory.cognitive, 1)):
            bank.bank["zero"] = [(0, torch.zeros(n, 32, dtype=dtype))]
            bank.last_written["zero"] = 0
        result = memory(sample, ids, {"episode_keys": ["zero"], "timesteps": [1]})
    assert torch.equal(result.backbone_features, sample.backbone_features)
    assert result.image_mask is sample.image_mask
    assert result.backbone_attention_mask is sample.backbone_attention_mask


def test_value_readout_starts_identical_then_learns_all_retrieval_layers():
    torch.set_num_threads(2)
    torch.manual_seed(74)
    memory = MemoryAdapter(vlm_dim=32, perceptual_dim=32, conditioning="residual",
        cognitive_qk_norm="layernorm", residual_identity_init=True,
        cognitive_residual_projection=True, residual_readout="value").eval()
    sample, ids = features(3)
    target = torch.randn_like(sample.backbone_features)
    optimizer = torch.optim.AdamW(memory.parameters(), lr=1e-3)
    for iteration in range(4):
        memory.reset()
        optimizer.zero_grad(set_to_none=True)
        output = memory(sample, ids, {"episode_keys": ["a"]*3, "timesteps": [0, 1, 2]})
        if iteration == 0:
            assert torch.equal(output.backbone_features, sample.backbone_features)
        (output.backbone_features-target).square().mean().backward()
        if iteration == 3:
            for bank in (memory.perceptual, memory.cognitive):
                for block in bank.retrieval_blocks:
                    for name in ("q_proj", "k_proj", "v_proj"):
                        grad = getattr(block, name).weight.grad
                        assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
        optimizer.step()
