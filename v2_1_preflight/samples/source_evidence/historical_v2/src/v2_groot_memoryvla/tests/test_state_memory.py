"""CPU engineering checks. All observations/physical labels here are synthetic."""

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

from v1_groot_memoryvla.model.memory.adapter import MemoryAdapter as V1Adapter
from v1_groot_memoryvla.model.memory.bank import MemoryBank
from v1_groot_memoryvla.policy.memory_state import MemoryState
from v1_groot_memoryvla.model.gr00t_n1d6 import gr00t_n1d6 as v1_model
from v1_groot_memoryvla.model.gr00t_n1d6 import processing_gr00t_n1d6
from v2_groot_memoryvla.config import Gr00tN1d6Config
from v2_groot_memoryvla.model.adapter import MemoryAdapter
from v2_groot_memoryvla.model.bank import CognitiveMemoryBank, StateQueryBlock
from v2_groot_memoryvla.model.gr00t_n1d6 import Gr00tN1d6
from v2_groot_memoryvla.model.state import PHYSICAL_LABELS, StateHead, masked_state_loss
from v2_groot_memoryvla.training import parameter_manifest


LABELS = PHYSICAL_LABELS  # Test fixture only; not an audit of any dataset field.


def observation(batch=1, dim=32):
    mask = torch.zeros(batch, 7, dtype=torch.bool)
    mask[:, 1:5] = True
    ids = torch.tensor([[3, 1, 1, 1, 1, 3, 0]] * batch)
    features = BatchFeature(data=dict(backbone_features=torch.randn(batch, 7, dim),
        visual_features=torch.randn(batch, 1, 4, dim), image_mask=mask,
        backbone_attention_mask=torch.ones(batch, 7, dtype=torch.bool)))
    return features, ids


def sample(features, i):
    return BatchFeature(data={k: v[i:i+1] for k, v in features.items()})


def adapter(capacity=4, guidance=True, dtype=torch.float32):
    old = V1Adapter(vlm_dim=32, perceptual_dim=32, capacity=capacity, eos_token_id=3,
                    residual_identity_init=True, residual_readout="value", cognitive_qk_norm="layernorm")
    with torch.no_grad():
        old.cognitive_residual_scale.fill_(.2)  # Mimic the learned, nonzero V1 scale.
        old.perceptual_residual.weight.normal_(std=.03)
    old.to(dtype=dtype).eval()
    return old, MemoryAdapter.from_v1(old, state_labels=LABELS, guidance_enabled=guidance)


class TinyBackbone(nn.Module):
    """Only the image/language encoder is mocked; tests use the real GR00T DiT."""

    def __init__(self, **kwargs):
        super().__init__()
        self.projection = nn.Linear(8, 8)
        self.seen_keys = None

    def prepare_input(self, inputs):
        assert "state_targets" not in inputs and "state_known" not in inputs
        self.seen_keys = set(inputs)
        return BatchFeature(data=inputs)

    def forward(self, inputs):
        output = {k: inputs[k] for k in ("backbone_features", "visual_features",
                                        "image_mask", "backbone_attention_mask")}
        output["backbone_features"] = self.projection(output["backbone_features"])
        return BatchFeature(data=output)


def tiny_config():
    return Gr00tN1d6Config(memory_state_labels=LABELS, memory_state_loss_weight=.3,
        diffusion_model_cfg=dict(num_attention_heads=2, attention_head_dim=4, num_layers=2,
                                 output_dim=8, dropout=0., positional_embeddings=None),
        backbone_embedding_dim=8, hidden_size=8, input_embedding_dim=8,
        max_state_dim=3, max_action_dim=2, max_num_embodiments=1, max_seq_len=16,
        action_horizon=2, num_inference_timesteps=2, use_alternate_vl_dit=False,
        tune_diffusion_model=False, tune_projector=False, tune_vlln=False,
        memory_length=4, memory_eos_token_id=3, memory_residual_identity_init=True,
        memory_residual_readout="value", memory_cognitive_qk_norm="layernorm")


def tiny_inputs(batch=3):
    features, ids = observation(batch, 8)
    return dict(features, input_ids=ids, state=torch.randn(batch, 1, 3),
        action=torch.randn(batch, 2, 2), action_mask=torch.ones(batch, 2, 2),
        embodiment_id=torch.zeros(batch, dtype=torch.long),
        state_targets=torch.tensor([[0., 1., float("nan")]] * batch),
        state_known=torch.tensor([[True, True, False]] * batch),
        memory_context=dict(mode="group", episode_keys=["ep"] * batch,
                            timesteps=list(range(batch)), group_lengths=[batch]))


class StateMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def assert_same_bank(self, first, second):
        self.assertEqual(set(first.bank), set(second.bank))
        self.assertEqual(first.last_written, second.last_written)
        self.assertEqual(first.merge_count, second.merge_count)
        for key in first.bank:
            self.assertEqual([t for t, _ in first.bank[key]], [t for t, _ in second.bank[key]])
            for (_, a), (_, b) in zip(first.bank[key], second.bank[key]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_masked_bce_ignores_nan_unknowns_and_normalizes_known_only(self):
        logits = torch.tensor([[100., 0., 0.]], requires_grad=True)
        targets = torch.tensor([[float("nan"), 1., 0.]])
        known = torch.tensor([[False, True, True]])
        loss = masked_state_loss(logits, targets, known)
        torch.testing.assert_close(loss, torch.tensor(2.).log())
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[0., -.25, .25]]))
        logits.grad = None
        zero = masked_state_loss(logits, torch.full_like(targets, float("nan")), torch.zeros_like(known))
        zero.backward()
        self.assertEqual(zero.item(), 0.)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_invalid_labels_masks_and_known_targets_rejected(self):
        for labels in ((), ("phase",), ("started", "started")):
            with self.assertRaises(ValueError):
                StateHead(32, labels)
        for target, mask in ((torch.tensor([[float("nan")]]), torch.ones(1, 1, dtype=torch.bool)),
                             (torch.tensor([[2.]]), torch.ones(1, 1, dtype=torch.bool)),
                             (torch.zeros(1, 1), torch.ones(1, 1))):
            with self.assertRaises(ValueError):
                masked_state_loss(torch.zeros(1, 1), target, mask)

    def test_disabled_guidance_matches_v1_including_full_bank_merge(self):
        for qk in ("none", "layernorm"):
            for value in (False, True):
                with self.subTest(qk=qk, value=value):
                    old = MemoryBank(32, 3, qk_norm=qk)
                    new = CognitiveMemoryBank(32, LABELS, 3, qk_norm=qk, guidance_enabled=False)
                    new.load_state_dict(old.state_dict(), strict=False)
                    # Disabling must bypass even already-learned state projections.
                    with torch.no_grad():
                        for block in new.retrieval_blocks:
                            block.state_query.weight.normal_()
                        new.gate_fusion_blocks.state_gate.weight.normal_()
                    for t in range(12):
                        current = torch.randn(1, 1, 32)
                        a = old.read_fuse_write(current, "a", t, return_value_message=value)
                        b = new.read_fuse_write(current, "a", t, return_value_message=value)
                        for x, y in zip(a if value else (a,), b if value else (b,)):
                            torch.testing.assert_close(x, y, rtol=0, atol=0)
                        self.assert_same_bank(old, new)
                    self.assertEqual(new.merge_count["a"], 9)
                    self.assertFalse(new.event_states)
                    self.assertIsNone(new.records)

    def test_zero_state_projections_match_v1_retrieval_before_retention_differs(self):
        old = MemoryBank(32, 8, qk_norm="layernorm")
        new = CognitiveMemoryBank(32, LABELS, 8)
        new.load_state_dict(old.state_dict(), strict=False)
        for t in range(5):
            current = torch.randn(1, 1, 32)
            a = old.read_fuse_write(current, "a", t, return_value_message=True)
            b = new.read_fuse_write(current, "a", t, return_value_message=True)
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_empty_history_exact_bf16_and_zero_history_values_give_zero_message(self):
        for dtype in (torch.float32, torch.bfloat16):
            bank = CognitiveMemoryBank(32, LABELS, 4).to(dtype=dtype)
            current = torch.randn(1, 1, 32).to(dtype)
            fused, message = bank.read_fuse_write(current, "a", 0, return_value_message=True)
            self.assertTrue(torch.equal(fused, current))
            self.assertTrue(torch.equal(message, torch.zeros_like(current)))
            self.assertTrue(torch.equal(bank.last_record["memory_gate"], torch.ones_like(current)))
            fused, message = bank.read_fuse_write(current, "a", 1, return_value_message=True,
                history_override=[(0, torch.zeros(1, 32, dtype=dtype))])
            self.assertTrue(torch.equal(message, torch.zeros_like(current)))

    def test_read_precedes_detached_write_and_rejects_future_or_repeated_time(self):
        bank = CognitiveMemoryBank(32, LABELS, 3)
        first = torch.randn(1, 1, 32, requires_grad=True)
        bank.read_fuse_write(first, "a", 0)
        current = torch.randn(1, 1, 32, requires_grad=True)
        fused, history = bank.read_fuse_write(current, "a", 1, return_history=True,
                                            capture_diagnostics=True, capture_attention=True)
        self.assertTrue(torch.equal(history, first.detach()))
        self.assertEqual(bank.last_diagnostics["history_timesteps"], [0])
        self.assertFalse(history.requires_grad)
        fused.square().sum().backward()
        self.assertIsNone(first.grad)
        self.assertIsNotNone(current.grad)
        for _, tensor in bank.bank["a"]:
            self.assertFalse(tensor.requires_grad)
        before = bank.runtime_state_dict()
        for time, override in ((1, None), (2, [(2, first.detach().squeeze(0))])):
            with self.assertRaises(ValueError):
                bank.read_fuse_write(current, "a", time, history_override=override)
        self.assertEqual(bank.last_written, before["last_written"])
        self.assertTrue(torch.equal(bank.event_states["a"], before["event_states"]["a"]))

    def test_state_is_added_after_query_projection_with_unchanged_residual_and_values(self):
        block = StateQueryBlock(32, 3, "layernorm")
        with torch.no_grad():
            block.state_query.weight.normal_(std=.2)
        query, key, value = torch.randn(1, 1, 32), torch.randn(1, 4, 32), torch.randn(1, 4, 32)
        state = torch.rand(1, 1, 3, requires_grad=True)
        block.state_feature = state
        captured = {}
        real = torch.nn.functional.scaled_dot_product_attention

        def attention(q, k, v, **kwargs):
            captured.update(q=q, k=k, v=v)
            return real(q, k, v, **kwargs)

        with patch("v2_groot_memoryvla.model.bank.F.scaled_dot_product_attention", attention):
            result, message = block(query, key, value, return_value_message=True)
        F = torch.nn.functional
        torch.testing.assert_close(captured["q"], block.q_proj(F.layer_norm(query, (32,))) + block.state_query(state))
        torch.testing.assert_close(captured["k"], block.k_proj(F.layer_norm(key, (32,))))
        torch.testing.assert_close(captured["v"], F.linear(value, block.v_proj.weight))
        x = block.attn_norm(query + message + block.v_proj.bias)
        torch.testing.assert_close(result, block.ffn_norm(x + block.ffn(x)), atol=1e-6, rtol=1e-5)
        result[..., 0].sum().backward()
        self.assertGreater(state.grad.abs().sum().item(), 0)

    def test_gate_history_coefficient_is_one_minus_gate(self):
        bank = CognitiveMemoryBank(32, LABELS, 4)
        gate = bank.gate_fusion_blocks
        gate.state_feature = torch.rand(1, 1, 3)
        with torch.no_grad():
            gate.state_gate.weight.normal_()
        current, retrieved, message = (torch.randn(1, 1, 32) for _ in range(3))
        fused, weighted = gate(current, retrieved, message)
        expected = torch.sigmoid(gate.proj(torch.cat((current, retrieved), -1)) + gate.state_gate(gate.state_feature))
        torch.testing.assert_close(fused, expected * current + (1 - expected) * retrieved)
        torch.testing.assert_close(weighted, (1 - expected) * message)

    def test_hysteresis_changes_only_on_confident_transition_including_reverse(self):
        bank = CognitiveMemoryBank(32, LABELS, 4)
        def event(p):
            return bank._event("a", torch.tensor([[[p, .5, .5]]], requires_grad=True))
        self.assertEqual([event(p) for p in (.1, .49, .51, .74, .9, .7, .5, .26, .1, .9)],
                         [False, False, False, False, True, False, False, False, True, True])
        self.assertFalse(bank.event_states["a"].requires_grad)

    def test_event_retention_stays_in_original_budget_and_reserves_recent_ordinary(self):
        bank = CognitiveMemoryBank(32, LABELS, 4, recent_slots=2)
        with torch.no_grad():
            bank.state_head.proj.weight.zero_()
            for t in range(10):
                p = .1 if t == 0 else .9
                bank.state_head.proj.bias.fill_(torch.logit(torch.tensor(p)).item())
                bank.read_fuse_write(torch.randn(1, 1, 32), "a", t)
                self.assertLessEqual(len(bank.bank["a"]), 4)
        self.assertEqual(bank.event_slots["a"], {1})
        self.assertEqual([t for t, _ in bank.bank["a"]], [1, 7, 8, 9])
        self.assertEqual(bank.eviction_count["a"], 6)
        # The first observation establishes the baseline (ordinary). Later
        # confident flips are events; the ordinary reserve also protects t=0.
        bank.reset()
        with torch.no_grad():
            for t in range(10):
                p = .1 if t % 2 == 0 else .9
                bank.state_head.proj.bias.fill_(torch.logit(torch.tensor(p)).item())
                bank.read_fuse_write(torch.randn(1, 1, 32), "a", t)
        self.assertEqual([t for t, _ in bank.bank["a"]], [0, 7, 8, 9])
        self.assertEqual(bank.event_slots["a"], {7, 8, 9})

    def test_adapter_disabled_guidance_preserves_v1_eos_injection_and_perception(self):
        for dtype in (torch.float32, torch.bfloat16):
            old, new = adapter(guidance=False, dtype=dtype)
            for t in range(9):
                features, ids = observation()
                features = BatchFeature(data={k: v.to(dtype) if v.is_floating_point() else v for k, v in features.items()})
                context = dict(episode_keys=["a"], timesteps=[t])
                a, b = old(features, ids, context), new(features, ids, context)
                torch.testing.assert_close(a.backbone_features, b.backbone_features, rtol=0, atol=0)
                self.assertIs(b.image_mask, features.image_mask)
                self.assert_same_bank(old.perceptual, new.perceptual)
                self.assert_same_bank(old.cognitive, new.cognitive)

    def test_state_head_reads_last_valid_current_eos_before_retrieval(self):
        _, new = adapter()
        features, ids = observation()
        ids[:, -1] = 3  # Padded EOS must not be selected.
        features.backbone_attention_mask[:, -1] = False
        seen = []
        hook = new.cognitive.state_head.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
        for t in range(3):
            output = new(features, ids, dict(episode_keys=["a"], timesteps=[t]))
        hook.remove()
        for current in seen:
            torch.testing.assert_close(current, features.backbone_features[:, 5:6], rtol=0, atol=0)
        torch.testing.assert_close(output.state_logits, new.cognitive.state_head(features.backbone_features[:, 5]))
        self.assertEqual(output.state_feature.shape, (1, 3))
        self.assertEqual(output.retrieved_memory.shape, (1, 32))
        self.assertEqual(output.memory_gate.shape, (1, 32))

    def test_groups_match_stream_no_future_leak_and_reset_event_state(self):
        _, new = adapter()
        features, ids = observation(8)
        context = dict(mode="group", episode_keys=["a"] * 8, timesteps=list(range(8)), group_lengths=[8])
        new.train()
        grouped = new(features, ids, context)
        self.assertFalse(new.cognitive.bank)
        self.assertFalse(new.cognitive.event_states)
        changed = deepcopy(features)
        changed.backbone_features[-1] *= 10
        altered = new(changed, ids, context)
        torch.testing.assert_close(grouped.backbone_features[:-1], altered.backbone_features[:-1], rtol=0, atol=0)
        new.eval()
        stream = [new(sample(features, i), ids[i:i+1], dict(episode_keys=["a"], timesteps=[i])) for i in range(8)]
        for name in ("backbone_features", "state_logits", "memory_gate"):
            torch.testing.assert_close(grouped[name], torch.cat([r[name] for r in stream]), rtol=0, atol=0)

    def test_cognitive_history_controls_leave_perceptual_history_and_output_fixed(self):
        _, new = adapter()
        for t in range(3):
            features, ids = observation(2)
            new(features, ids, dict(episode_keys=["a", "b"], timesteps=[t, t]))
        snapshot = new.runtime_state_dict()
        features, ids = observation(2)
        normal = new(features, ids, dict(episode_keys=["a", "b"], timesteps=[3, 3]))
        normal_per = deepcopy(new.perceptual)
        for mode in ("empty", "other_episode"):
            new.load_runtime_state_dict(snapshot)
            altered = new(features, ids, dict(episode_keys=["a", "b"], timesteps=[3, 3], cognitive_history_mode=mode))
            torch.testing.assert_close(normal.backbone_features[:, 1:5], altered.backbone_features[:, 1:5], rtol=0, atol=0)
            self.assert_same_bank(normal_per, new.perceptual)
            self.assertFalse(torch.equal(normal.backbone_features[:, 5], altered.backbone_features[:, 5]))
        with self.assertRaises(ValueError):
            new(features, ids, dict(episode_keys=["a", "b"], timesteps=[4, 4], history_mode="empty"))

    def test_runtime_checkpoint_restores_hysteresis_slots_and_continuation(self):
        _, new = adapter()
        for t in range(8):
            features, ids = observation()
            new(features, ids, dict(episode_keys=["a"], timesteps=[t]))
        buffer = io.BytesIO()
        torch.save(dict(weights=new.state_dict(), runtime=new.runtime_state_dict()), buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        _, clone = adapter()
        clone.load_state_dict(state["weights"], strict=True)
        self.assertFalse(clone.cognitive.bank)
        clone.load_runtime_state_dict(state["runtime"])
        for t in range(8, 12):
            features, ids = observation()
            a = new(features, ids, dict(episode_keys=["a"], timesteps=[t]))
            b = clone(features, ids, dict(episode_keys=["a"], timesteps=[t]))
            for name in ("backbone_features", "state_logits", "memory_gate"):
                torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
            self.assert_same_bank(new.cognitive, clone.cognitive)
            self.assertTrue(torch.equal(new.cognitive.event_states["a"], clone.cognitive.event_states["a"]))
        self.assertNotIn("bank", new.state_dict())

    def test_session_episode_reset_clears_only_matching_event_runtime(self):
        _, new = adapter()
        state = MemoryState(new)
        features, ids = observation(2)
        context = state.context(2, dict(session_id="s", env_ids=[0, 1], episode_ids=["a", "b"]))
        new(features, ids, context)
        state.advance(context)
        self.assertEqual(len(new.cognitive.event_states), 2)
        state.reset(dict(session_id="s", env_ids=[0], episode_ids=["c"]))
        self.assertNotIn(("s", "0", "a"), new.cognitive.event_states)
        self.assertIn(("s", "1", "b"), new.cognitive.event_states)
        state.reset()
        self.assertFalse(new.cognitive.event_states)
        self.assertFalse(new.cognitive.event_slots)

    def test_v1_loader_rejects_missing_existing_parameters(self):
        old, new = adapter()
        state = old.state_dict()
        state.pop("cognitive.retrieval_blocks.0.q_proj.weight")
        with self.assertRaisesRegex(ValueError, "Incompatible V1"):
            new.load_v1_state_dict(state)

    def test_all_history_slots_are_read_with_time_encoding_on_keys_only(self):
        bank = CognitiveMemoryBank(32, LABELS, 4)
        for t in range(4):
            bank.read_fuse_write(torch.randn(1, 1, 32), "a", t)
        history = deepcopy(bank.bank["a"])
        captured = []
        handles = [block.register_forward_pre_hook(lambda _, args: captured.append(args))
                   for block in bank.retrieval_blocks]
        bank.read_fuse_write(torch.randn(1, 1, 32), "a", 4,
                             capture_diagnostics=True, capture_attention=True)
        for handle in handles:
            handle.remove()
        raw = torch.stack([f for _, f in history]).reshape(1, 4, 32)
        pe = bank.timestep_encoder(torch.tensor([t for t, _ in history])).unsqueeze(0)
        for _, keys, values in captured:
            torch.testing.assert_close(values, raw, rtol=0, atol=0)
            torch.testing.assert_close(keys, raw + pe, rtol=0, atol=0)
        diagnostic = bank.last_diagnostics["retrieval_attention"]
        self.assertEqual(diagnostic["slot_timesteps"], [0, 1, 2, 3])
        self.assertEqual(len(diagnostic["slot_mass"]), 4)
        self.assertAlmostEqual(sum(diagnostic["slot_mass"]), 1., places=5)

    def test_whole_memory_empty_control_keeps_state_interface_and_identity(self):
        _, new = adapter()
        features, ids = observation(2)
        output = new(features, ids, dict(mode="empty", episode_keys=["a", "b"], timesteps=[0, 0]))
        torch.testing.assert_close(output.backbone_features, features.backbone_features, rtol=0, atol=0)
        torch.testing.assert_close(output.state_logits, new.cognitive.state_head(features.backbone_features[:, 5]))
        self.assertFalse(new.cognitive.bank)
        self.assertFalse(new.perceptual.bank)

    def test_sequence_chunks_continue_then_clear_event_runtime_on_episode_end(self):
        _, new = adapter()
        new.train()
        features, ids = observation(3)
        context = dict(mode="train_sequence", episode_keys=["a"] * 3, timesteps=[0, 1, 2],
                       sequence_start=True, sequence_end=False)
        first = new(features, ids, context)
        self.assertEqual(first.state_logits.shape, (3, 3))
        self.assertIn("a", new.cognitive.event_states)
        context.update(timesteps=[3, 4, 5], sequence_start=False, sequence_end=True)
        second = new(features, ids, context)
        self.assertEqual(second.state_logits.shape, (3, 3))
        self.assertFalse(new.cognitive.bank)
        self.assertFalse(new.cognitive.event_states)
        with self.assertRaisesRegex(ValueError, "Missing sequence_start"):
            new(features, ids, context)


class IntegratedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.backbone_patch = patch.object(v1_model, "get_backbone_cls", lambda _: TinyBackbone)
        self.collator_patch = patch.object(processing_gr00t_n1d6, "Gr00tN1d6DataCollator", lambda **kwargs: None)
        self.backbone_patch.start()
        self.collator_patch.start()
        self.addCleanup(self.backbone_patch.stop)
        self.addCleanup(self.collator_patch.stop)

    def model(self):
        model = Gr00tN1d6(tiny_config())
        with torch.no_grad():
            model.memory.cognitive_residual_scale.fill_(.2)
        return model

    def test_joint_loss_gradients_and_optimizer_update_through_real_frozen_dit(self):
        model = self.model().train()
        data = tiny_inputs()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        output = model(data)
        torch.testing.assert_close(output["loss"], output["action_loss"] + .3 * output["state_loss"])
        bank = model.memory.cognitive
        projections = [b.state_query.weight for b in bank.retrieval_blocks] + [bank.gate_fusion_blocks.state_gate.weight]
        gradients = torch.autograd.grad(output["action_loss"], projections, retain_graph=True)
        for grad in gradients:
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0.)
        initial_head_grad, = torch.autograd.grad(output["action_loss"], [bank.state_head.proj.weight], retain_graph=True)
        self.assertEqual(initial_head_grad.abs().sum().item(), 0.)  # Zero state projections on step 1.
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
        output["loss"].backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)
        optimizer.step()
        for fragment in ("state_head.proj.weight", "state_query.weight", "state_gate.weight", "q_proj.weight", "gate_fusion_blocks.proj.weight"):
            names = [n for n, p in model.named_parameters() if p.requires_grad and fragment in n]
            self.assertTrue(names)
            for name in names:
                self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]), name)
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                self.assertTrue(torch.equal(before[name], parameter), name)
        optimizer.zero_grad(set_to_none=True)
        second = model(data)
        action_to_head, = torch.autograd.grad(second["action_loss"], [bank.state_head.proj.weight])
        self.assertGreater(action_to_head.abs().sum().item(), 0.)
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.action_head.training)
        self.assertFalse(model.memory.perceptual.training)
        self.assertTrue(all(r["name"].startswith("memory.cognitive.") for r in parameter_manifest(model) if r["trainable"]))

    def test_labels_change_loss_only_not_state_memory_or_actions(self):
        model = self.model().train()
        data = tiny_inputs()
        torch.manual_seed(7)
        first = model(data)
        altered = deepcopy(data)
        altered["state_targets"] = 1 - data["state_targets"]
        torch.manual_seed(7)
        second = model(altered)
        for name in ("state_logits", "retrieved_memory", "memory_gate", "backbone_features", "action_loss"):
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
        self.assertNotEqual(first["state_loss"].item(), second["state_loss"].item())
        self.assertNotIn("state_targets", model.backbone.seen_keys)
        model.eval()
        data["memory_context"] = dict(episode_keys=["a", "b", "c"], timesteps=[0, 0, 0])
        torch.manual_seed(11)
        action1 = model.get_action(data)["action_pred"]
        model.memory.reset()
        data.pop("state_targets")
        data.pop("state_known")
        torch.manual_seed(11)
        action2 = model.get_action(data)["action_pred"]
        torch.testing.assert_close(action1, action2, rtol=0, atol=0)

    def test_guidance_disabled_actions_match_same_v1_weights_after_overflow(self):
        old = v1_model.Gr00tN1d6(tiny_config()).eval()
        old.memory.cognitive_residual_scale.data.fill_(.2)
        new = Gr00tN1d6.from_v1_model(old, state_labels=LABELS, memory_state_guidance=False).eval()
        for t in range(8):
            data = tiny_inputs(1)
            data["memory_context"] = dict(episode_keys=["a"], timesteps=[t])
            torch.manual_seed(17)
            # V1 does not know V2 label fields; pass only observation/action inputs.
            first = old.get_action({k: v for k, v in data.items() if k not in ("state_targets", "state_known")})
            torch.manual_seed(17)
            second = new.get_action(data)
            torch.testing.assert_close(first["action_pred"], second["action_pred"], rtol=0, atol=0)

    def test_save_pretrained_reload_config_weights_actions_and_runtime(self):
        model = self.model().eval()
        # Persist learned, nonzero state branches too, not only their defaults.
        with torch.no_grad():
            for b in model.memory.cognitive.retrieval_blocks:
                b.state_query.weight.normal_(std=.1)
            model.memory.cognitive.gate_fusion_blocks.state_gate.weight.normal_(std=.1)
        data = tiny_inputs(1)
        for t in range(5):
            data["memory_context"] = dict(episode_keys=["a"], timesteps=[t])
            model.get_action(data)
        runtime = model.memory.runtime_state_dict()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            config_json = json.loads((Path(directory) / "config.json").read_text())
            for name in tiny_config().__dataclass_fields__:
                if name.startswith("memory_"):
                    self.assertIn(name, config_json)
            self.assertIsInstance(AutoConfig.from_pretrained(directory), Gr00tN1d6Config)
            restored = AutoModel.from_pretrained(directory).eval()
            self.assertIsInstance(restored, Gr00tN1d6)
            self.assertFalse(restored.memory.cognitive.bank)
            self.assertEqual(restored.config.memory_state_labels, LABELS)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
            restored.memory.load_runtime_state_dict(runtime)
            data["memory_context"]["timesteps"] = [5]
            torch.manual_seed(19)
            first = model.get_action(data)
            torch.manual_seed(19)
            second = restored.get_action(data)
            torch.testing.assert_close(first["action_pred"], second["action_pred"], rtol=0, atol=0)

    def test_hf_initializer_preserves_zero_state_projections(self):
        model = self.model()
        bank = model.memory.cognitive
        projections = [b.state_query for b in bank.retrieval_blocks] + [bank.gate_fusion_blocks.state_gate]
        for projection in projections:
            nn.init.ones_(projection.weight)
            model._init_weights(projection)
            self.assertEqual(projection.weight.abs().sum().item(), 0.)

    def test_direct_v1_checkpoint_loading_initializes_only_new_guidance_weights(self):
        old = v1_model.Gr00tN1d6(tiny_config()).eval()
        old.memory.cognitive_residual_scale.data.fill_(.2)
        with tempfile.TemporaryDirectory() as directory:
            old.save_pretrained(directory)
            new, info = Gr00tN1d6.from_pretrained(directory, config=tiny_config(),
                initialize_from_v1=True, output_loading_info=True)
            self.assertFalse(info["unexpected_keys"])
            self.assertTrue(info["missing_keys"])
            self.assertTrue(all("state_head" in n or "state_query" in n or "state_gate" in n
                                for n in info["missing_keys"]))
            for name, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
            for b in new.memory.cognitive.retrieval_blocks:
                self.assertEqual(b.state_query.weight.abs().sum().item(), 0.)
            self.assertEqual(new.memory.cognitive.gate_fusion_blocks.state_gate.weight.abs().sum().item(), 0.)
            self.assertTrue(torch.isfinite(new.memory.cognitive.state_head.proj.weight).all())

    def test_joint_training_requires_mask_and_explicit_loss_weight(self):
        model = self.model().train()
        data = tiny_inputs()
        data.pop("state_known")
        with self.assertRaisesRegex(ValueError, "provided together"):
            model(data)
        data.pop("state_targets")
        with self.assertRaisesRegex(ValueError, "Joint training"):
            model(data)
        model.config.memory_state_loss_weight = None
        with self.assertRaisesRegex(ValueError, "Joint training"):
            model(tiny_inputs())


if __name__ == "__main__":
    unittest.main()
