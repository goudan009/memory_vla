import pytest
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from v1_groot_memoryvla.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from v1_groot_memoryvla.model.gr00t_n1d6 import gr00t_n1d6 as groot
from v1_groot_memoryvla.model.gr00t_n1d6 import processing_gr00t_n1d6
from v1_groot_memoryvla.model.memory.adapter import MemoryAdapter
from v1_groot_memoryvla.model.memory.fusion import RESIDUAL_LIMIT, CognitiveActionAttention


def config(mode=None, position="late", alternate=True):
    diffusion = dict(num_attention_heads=2, attention_head_dim=4, num_layers=4,
                     interleave_self_attention=True, output_dim=8, dropout=0,
                     positional_embeddings=None)
    if mode is not None:
        diffusion.update(memory_fusion_mode=mode, memory_fusion_position=position)
    return Gr00tN1d6Config(
        diffusion_model_cfg=diffusion, backbone_embedding_dim=8,
        hidden_size=8, input_embedding_dim=8, max_state_dim=3, max_action_dim=2,
        max_num_embodiments=1, max_seq_len=8, action_horizon=2, num_inference_timesteps=2,
        use_alternate_vl_dit=alternate, tune_diffusion_model=False, tune_projector=False,
        tune_vlln=False, memory_enabled=True, memory_length=2, memory_eos_token_id=3,
        memory_residual_identity_init=True, memory_residual_readout="value",
        memory_cognitive_qk_norm="layernorm",
    )


def observation(batch):
    rng = torch.Generator().manual_seed(20 + batch)
    return dict(
        backbone_features=torch.randn(batch, 3, 8, generator=rng),
        visual_features=torch.randn(batch, 1, 1, 8, generator=rng),
        image_mask=torch.tensor([[True, False, False]] * batch),
        backbone_attention_mask=torch.ones(batch, 3, dtype=torch.bool),
        input_ids=torch.tensor([[1, 2, 3]] * batch),
        state=torch.randn(batch, 1, 3, generator=rng),
        action=torch.randn(batch, 2, 2, generator=rng),
        action_mask=torch.ones(batch, 2, 2),
        embodiment_id=torch.zeros(batch, dtype=torch.long),
    )


class TinyBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def prepare_input(self, inputs):
        return BatchFeature(data=inputs)

    def forward(self, inputs):
        return BatchFeature(data={k: inputs[k] for k in (
            "backbone_features", "visual_features", "image_mask", "backbone_attention_mask",
        )})


@pytest.fixture
def tiny_backbone(monkeypatch):
    monkeypatch.setattr(groot, "get_backbone_cls", lambda _: TinyBackbone)
    monkeypatch.setattr(processing_gr00t_n1d6, "Gr00tN1d6DataCollator", lambda **kwargs: None)


def input_group():
    return dict(observation(3), memory_context=dict(
        mode="group", episode_keys=["ep"] * 3, timesteps=[0, 1, 2], group_lengths=[3],
    ))


def test_openvla_independent_conditions_train_through_native_attention():
    torch.manual_seed(42)
    cfg = config()
    cfg.memory_conditioning = "openvla"
    cfg.memory_perceptual_dim = 4
    cfg.state_dropout_prob = 0.0
    head = groot.Gr00tN1d6ActionHead(cfg).requires_grad_(True)
    head.model.attend_text_every_n_blocks = 2
    memory = MemoryAdapter(vlm_dim=8, visual_dim=8, perceptual_dim=4, capacity=16,
                           conditioning="openvla", eos_token_id=3)
    inputs = observation(3)
    inputs["visual_features"] = torch.randn(3, 3, 4, 8)
    inputs["backbone_features"] = torch.randn(3, 13, 8)
    inputs["input_ids"] = torch.tensor([[1] * 12 + [3]] * 3)
    inputs["image_mask"] = inputs["input_ids"] == 1
    inputs["backbone_attention_mask"] = torch.ones(3, 13, dtype=torch.bool)
    inputs["action_mask"][0, -1] = 0
    context = dict(mode="group", episode_keys=["ep"] * 3, timesteps=[0, 1, 2], group_lengths=[3])
    fused = memory(inputs, inputs["input_ids"], context)
    assert torch.equal(fused["backbone_features"], inputs["backbone_features"])
    assert torch.equal(fused["image_mask"], inputs["image_mask"])
    first_camera_tokens = memory.perceptual_encoder(inputs["visual_features"][0]).reshape(12, 4)
    torch.testing.assert_close(fused["memory_perceptual_tokens"][0], first_camera_tokens)
    assert torch.equal(fused["memory_cognitive_token"][0, 0], inputs["backbone_features"][0, -1])
    assert not memory.cognitive.bank and not memory.perceptual.bank

    captures = []
    handles = [
        block.register_forward_pre_hook(lambda _, args, kwargs: captures.append((args[0].shape, kwargs)),
                                        with_kwargs=True)
        for block in head.model.transformer_blocks[:3]]
    output = head(fused, BatchFeature(data=inputs))
    for handle in handles:
        handle.remove()
    text_cross, self_attention, vision_cross = captures
    assert text_cross[0][1] == 1 + inputs["state"].shape[1] + cfg.action_horizon
    torch.testing.assert_close(text_cross[1]["encoder_hidden_states"], head.vlln(inputs["backbone_features"]))
    assert torch.equal(text_cross[1]["encoder_attention_mask"], ~inputs["image_mask"])
    torch.testing.assert_close(vision_cross[1]["encoder_hidden_states"], head._perceptual_condition(fused))
    assert vision_cross[1]["encoder_attention_mask"] is None
    assert self_attention[1]["encoder_hidden_states"] is None
    assert torch.equal(output["action_mask"], inputs["action_mask"])
    output["loss"].backward()
    for parameters in (head.memory_perception_projection.parameters(), head.memory_cognition_projection.parameters(),
                       memory.perceptual.retrieval_blocks.parameters(), memory.cognitive.retrieval_blocks.parameters()):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)


@pytest.mark.parametrize("block,position", [(8, "middle"), (24, "late")])
def test_explicit_route_b_block_is_used_in_forward_and_denoising(tiny_backbone, block, position):
    cfg = config("route_b", position)
    cfg.diffusion_model_cfg.update(num_layers=32, memory_fusion_block=block)
    model = groot.Gr00tN1d6(cfg).eval()
    head = model.action_head
    events = []
    head.model.transformer_blocks[block].register_forward_hook(lambda *_: events.append("block"))
    head.model.transformer_blocks[block + 1].register_forward_pre_hook(lambda *_: events.append("next"))
    head.cognitive_fusion.register_forward_hook(lambda *_: events.append("fusion"))
    assert head.memory_fusion_block == cfg.diffusion_model_cfg["memory_fusion_block"] == block
    inputs = input_group()
    model(inputs)
    assert events == ["block", "fusion", "next"]
    events.clear()
    model.get_action(inputs)
    assert events == ["block", "fusion", "next"] * cfg.num_inference_timesteps


@pytest.mark.parametrize("block", [-1, 4, True, 2.0, [2]])
def test_explicit_route_b_block_rejects_invalid_indices(block):
    cfg = config("route_b")
    cfg.diffusion_model_cfg["memory_fusion_block"] = block
    with pytest.raises(ValueError, match="memory_fusion_block.*zero-based integer"):
        groot.Gr00tN1d6ActionHead(cfg)


@pytest.mark.parametrize("mode,position,alternate,prefix,projection", [
    ("route_a", "late", True, "memory.cognitive_fusion.", "value"),
    ("route_b", "late", True, "action_head.cognitive_fusion.", "output"),
    ("route_b", "middle", False, "action_head.cognitive_fusion.", "output"),
])
def test_old_checkpoint_is_exact_at_init_and_fusion_trains(
    tiny_backbone, tmp_path, mode, position, alternate, prefix, projection,
):
    torch.manual_seed(4)
    legacy = groot.Gr00tN1d6(config(alternate=alternate)).eval()
    legacy.memory.cognitive_residual_scale.data.fill_(0.01)
    legacy.save_pretrained(tmp_path)
    old_config = Gr00tN1d6Config.from_pretrained(tmp_path)
    assert "memory_fusion_mode" not in old_config.diffusion_model_cfg
    g0, g0_loading = groot.Gr00tN1d6.from_pretrained(tmp_path, output_loading_info=True)
    assert not (g0_loading["missing_keys"] or g0_loading["unexpected_keys"])
    assert g0.state_dict().keys() == legacy.state_dict().keys()
    routed_config = config(mode, position, alternate)
    routed, loading = groot.Gr00tN1d6.from_pretrained(
        tmp_path, config=routed_config, output_loading_info=True,
    )
    assert not loading["unexpected_keys"]
    new_keys = set(routed.state_dict()) - set(legacy.state_dict())
    assert new_keys == set(loading["missing_keys"])
    assert new_keys and all(key.startswith(prefix) for key in new_keys)
    for key, weight in legacy.state_dict().items():
        assert torch.equal(weight, routed.state_dict()[key])
    fusion = routed.get_submodule(prefix.rstrip("."))
    zero_projection = getattr(fusion, projection).weight
    assert torch.count_nonzero(zero_projection) == 0

    inputs = input_group()
    for predict, output_key in [(lambda m: m(inputs), "action_loss"),
                                (lambda m: m.get_action(inputs), "action_pred")]:
        torch.manual_seed(6)
        expected = predict(g0)
        torch.manual_seed(6)
        actual = predict(routed)
        assert "fusion_action_features" not in expected
        assert actual["fusion_action_features"].shape == (3, 2, 8)
        assert torch.equal(actual[output_key], expected[output_key])

    routed.train()
    routed.requires_grad_(False)
    fusion.requires_grad_(True)
    actual = routed(inputs)
    assert actual["fusion_action_features"].requires_grad
    actual["loss"].backward()
    assert torch.isfinite(zero_projection.grad).all() and torch.count_nonzero(zero_projection.grad)
    optimizer = torch.optim.SGD(fusion.parameters(), lr=0.3)
    optimizer.step()
    optimizer.zero_grad()
    features = routed(inputs)["fusion_action_features"]
    features[-1, :, 0].sum().backward()
    for parameter in fusion.parameters():
        assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad)

    if mode == "route_b":
        assert routed.action_head.memory_fusion_block == {"middle": 1, "late": 3}[position]
    routed_path = tmp_path / mode
    routed.save_pretrained(routed_path)
    reloaded = groot.Gr00tN1d6.from_pretrained(routed_path)
    assert reloaded.config.diffusion_model_cfg == routed_config.diffusion_model_cfg
    assert torch.equal(reloaded.state_dict()[prefix + projection + ".weight"], zero_projection)


def test_route_b_slots_are_detached_pre_write_and_follow_history_control():
    adapter = MemoryAdapter(vlm_dim=8, perceptual_dim=8, capacity=2, eos_token_id=3,
                            fusion_mode="route_b").eval()
    inputs = observation(2)
    inputs["backbone_features"].requires_grad_(True)

    def step(time, history_mode="normal"):
        return adapter(inputs, inputs["input_ids"], dict(
            episode_keys=["a", "b"], timesteps=[time, time], history_mode=history_mode,
        ))

    initial = step(0)
    assert not initial.cognitive_history_mask.any()
    assert torch.count_nonzero(initial.cognitive_history_slots) == 0
    step(2)
    before_merge = torch.stack([
        torch.cat([value for _, value in adapter.cognitive.bank[key]]) for key in ("a", "b")
    ])
    swapped = step(3, "other_episode")
    assert swapped.cognitive_history_mask.all()
    assert not swapped.cognitive_history_slots.requires_grad
    assert torch.equal(swapped.cognitive_history_slots, before_merge.flip(0))
    assert adapter.cognitive.merge_count == {"a": 1, "b": 1}
    empty = step(4, "empty")
    assert not empty.cognitive_history_mask.any()
    assert adapter.cognitive.last_written == {"a": 3, "b": 3}
    with pytest.raises(ValueError, match="strictly past|Non-increasing"):
        step(3)


def test_route_b_mask_action_only_and_residual_bound():
    torch.manual_seed(3)
    fusion = CognitiveActionAttention(8, 8)
    with torch.no_grad():
        fusion.output.weight.normal_(std=5)
    hidden = torch.randn(2, 4, 8)
    slots = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    patched = fusion(hidden, slots=slots, mask=mask, action_tokens=2)
    assert torch.equal(patched[:, :2], hidden[:, :2])
    assert torch.equal(patched[1], hidden[1])
    assert not torch.equal(patched[0, 2:], hidden[0, 2:])
    assert (patched - hidden).abs().max() <= RESIDUAL_LIMIT + 1e-6
    poisoned = slots.masked_fill(~mask[..., None], float("nan"))
    assert torch.equal(patched, fusion(hidden, slots=poisoned, mask=mask, action_tokens=2))
