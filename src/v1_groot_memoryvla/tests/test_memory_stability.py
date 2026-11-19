"""Regression for the saturated first cognitive retrieval layer seen on real EOS."""
import torch
import pytest

from v1_groot_memoryvla.model.memory.bank import CrossTransformerBlock


@pytest.mark.parametrize("normalized", [False, True])
def test_large_first_frame_does_not_kill_retrieval_gradient(normalized):
    torch.manual_seed(4)
    dim = 32
    block = CrossTransformerBlock(dim, qk_norm="layernorm" if normalized else "none")
    with torch.no_grad():
        for projection in (block.q_proj, block.k_proj):
            projection.weight.copy_(torch.eye(dim))
            projection.bias.zero_()
    direction = torch.randn(1, 1, dim)
    query = direction * 10
    keys = torch.cat([direction * 20, direction.repeat(1, 7, 1) * 5], dim=1)
    keys = keys + .5 * torch.randn_like(keys)
    values = torch.randn(1, 8, dim)
    captured = {}
    handles = [getattr(block, name).register_forward_hook(
        lambda module, args, out, name=name: captured.update({name: out}))
        for name in ("q_proj", "k_proj")]
    output = block(query, keys, values)
    prob = (captured["q_proj"] @ captured["k_proj"].transpose(-1, -2) / dim**.5).softmax(-1)
    (output * torch.randn_like(output)).sum().backward()
    for handle in handles:
        handle.remove()
    if normalized:
        assert prob.max() < .2
        assert block.q_proj.weight.grad.norm() > 1e-9
        assert torch.isfinite(output).all()
    else:
        assert prob[0, 0, 0] == 1
        assert block.q_proj.weight.grad.norm() == 0


def test_normalization_preserves_legacy_weights_and_values():
    torch.manual_seed(8)
    old = CrossTransformerBlock(32)
    fixed = CrossTransformerBlock(32, qk_norm="layernorm")
    fixed.load_state_dict(old.state_dict(), strict=True)
    q, k, v = torch.randn(1, 1, 32), torch.randn(1, 8, 32), torch.randn(1, 8, 32)
    captured = []
    handle = fixed.v_proj.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    fixed(q, k, v)
    handle.remove()
    torch.testing.assert_close(captured[0], v, rtol=0, atol=0)
    with pytest.raises(ValueError, match="qk_norm"):
        CrossTransformerBlock(32, qk_norm="unknown")
