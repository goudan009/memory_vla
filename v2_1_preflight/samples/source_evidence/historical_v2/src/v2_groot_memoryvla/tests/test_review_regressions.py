"""Regression cases for the four review findings; synthetic CPU inputs only."""

from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch

import torch

from v1_groot_memoryvla.model.gr00t_n1d6 import gr00t_n1d6 as v1_model
from v1_groot_memoryvla.model.gr00t_n1d6 import processing_gr00t_n1d6
from v2_groot_memoryvla.model.adapter import MemoryAdapter
from v2_groot_memoryvla.model.gr00t_n1d6 import Gr00tN1d6
from .test_state_memory import LABELS, TinyBackbone, adapter, observation, tiny_config


class AdapterReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(51)

    def test_mixed_precision_conversion_preserves_each_old_tensor_and_v1_forward(self):
        for module_dtype, scalar_dtype in ((torch.float32, torch.bfloat16),
                                            (torch.bfloat16, torch.float32)):
            with self.subTest(module_dtype=module_dtype):
                old, _ = adapter(dtype=module_dtype)
                old.cognitive_residual_scale.data = old.cognitive_residual_scale.data.to(scalar_dtype)
                new = MemoryAdapter.from_v1(old, state_labels=LABELS, guidance_enabled=False)
                for name, value in old.state_dict().items():
                    self.assertEqual(value.dtype, new.state_dict()[name].dtype, name)
                    torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
                    self.assertNotEqual(value.data_ptr(), new.state_dict()[name].data_ptr(), name)
                self.assertEqual(new.cognitive.state_head.proj.weight.dtype, module_dtype)
                for t in range(7):
                    features, ids = observation()
                    features = {k: v.to(module_dtype) if v.is_floating_point() else v for k, v in features.items()}
                    context = dict(episode_keys=["a"], timesteps=[t])
                    expected = old(features, ids, context)
                    actual = new(features, ids, context)
                    torch.testing.assert_close(expected.backbone_features, actual.backbone_features, rtol=0, atol=0)

    def test_wrong_history_matches_slot_count_rms_and_temporal_positions(self):
        _, model = adapter()
        keys, times = ["a", "b", "c"], [[1, 2], [0, 1, 2], [3, 4, 5]]
        model.cognitive.bank = {key: [(t, torch.randn(1, 32) * scale) for t in stamps]
                                for key, stamps, scale in zip(keys, times, (1., 10., .01))}
        model.cognitive.last_written = dict(zip(keys, (2, 2, 5)))
        snapshot = model.runtime_state_dict()
        features, ids = observation(3)
        context = dict(episode_keys=keys, timesteps=[10] * 3, memory_diagnostics=True)
        normal = model(features, ids, context)
        perceptual_after_normal = deepcopy(model.perceptual.bank)
        model.load_runtime_state_dict(snapshot)
        reads = []
        handle = model.cognitive.retrieval_blocks[0].register_forward_pre_hook(
            lambda _, args: reads.append(args[2].detach().clone()))
        try:
            wrong = model(features, ids, dict(context, cognitive_history_mode="other_episode"))
        finally:
            handle.remove()
        for key, values, stamps in zip(keys, reads, times):
            expected = torch.stack([f for _, f in snapshot["cognitive"]["bank"][key]]).reshape(1, -1, 32)
            self.assertEqual(values.shape, expected.shape)
            torch.testing.assert_close(values.float().square().mean(-1).sqrt(),
                                       expected.float().square().mean(-1).sqrt(), rtol=2e-6, atol=2e-6)
            self.assertFalse(torch.equal(values, expected))
        controls = model.last_cognitive_history_control["matches"]
        for key, entry, stamps in zip(keys, controls, times):
            self.assertNotEqual(entry["donor_key"], key)
            self.assertEqual(entry["read_timesteps"], stamps)
            self.assertEqual(entry["slots"], len(stamps))
            self.assertTrue(all(t < 10 for t in entry["donor_timesteps"]))
            self.assertEqual(len(entry["scale_factors"]), len(stamps))
        torch.testing.assert_close(wrong.backbone_features[:, 1:5], normal.backbone_features[:, 1:5], rtol=0, atol=0)
        for key in keys:
            for (t, expected), (actual_t, actual) in zip(perceptual_after_normal[key], model.perceptual.bank[key]):
                self.assertEqual(t, actual_t)
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_mixed_precision_runtime_roundtrip_preserves_stored_history_dtype(self):
        _, model = adapter(dtype=torch.bfloat16)
        for bank in (model.cognitive, model.perceptual):
            bank.bank = {"a": [(0, torch.randn(1 if bank is model.cognitive else 4, 32))]}
            bank.last_written = {"a": 0}
        snapshot = model.runtime_state_dict()
        _, restored = adapter(dtype=torch.bfloat16)
        restored.load_state_dict(model.state_dict(), strict=True)
        restored.load_runtime_state_dict(snapshot)
        for name in ("cognitive", "perceptual"):
            expected = snapshot[name]["bank"]["a"][0][1]
            actual = getattr(restored, name).bank["a"][0][1]
            self.assertEqual(expected.dtype, actual.dtype)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        features, ids = observation()
        # The BF16 model consumes BF16 current features, while the archived
        # history may have been produced by an FP32 memory implementation.
        features = {k: v.bfloat16() if v.is_floating_point() else v for k, v in features.items()}
        with torch.autocast("cpu", dtype=torch.bfloat16):
            first = model(features, ids, dict(episode_keys=["a"], timesteps=[1]))
            second = restored(features, ids, dict(episode_keys=["a"], timesteps=[1]))
        torch.testing.assert_close(first.backbone_features, second.backbone_features, rtol=0, atol=0)

    def test_wrong_history_rms_match_in_bf16(self):
        _, model = adapter(dtype=torch.bfloat16)
        model.cognitive.bank = {key: [(t, (torch.randn(1, 32) * scale).bfloat16()) for t in range(2)]
                                for key, scale in (("a", 1.), ("b", 10.))}
        model.cognitive.last_written = {"a": 1, "b": 1}
        features, ids = observation(2)
        features = {k: v.bfloat16() if v.is_floating_point() else v for k, v in features.items()}
        output = model(features, ids, dict(episode_keys=["a", "b"], timesteps=[2, 2],
                                          cognitive_history_mode="other_episode"))
        self.assertTrue(torch.isfinite(output.backbone_features).all())
        for match in model.last_cognitive_history_control["matches"]:
            torch.testing.assert_close(torch.tensor(match["matched_rms"]), torch.tensor(match["reference_rms"]),
                                       rtol=.005, atol=0.)

    def test_wrong_history_rejects_insufficient_or_zero_scale_donor_before_any_write(self):
        for reason in ("slots", "zero_scale"):
            with self.subTest(reason=reason):
                _, model = adapter()
                model.cognitive.bank = {
                    "a": [(t, torch.randn(1, 32)) for t in range(3 if reason == "slots" else 1)],
                    "b": [(0, torch.randn(1, 32) if reason == "slots" else torch.zeros(1, 32))],
                }
                model.cognitive.last_written = {k: slots[-1][0] for k, slots in model.cognitive.bank.items()}
                before = model.runtime_state_dict()
                features, ids = observation(2)
                with self.assertRaisesRegex(ValueError, "matched donor"):
                    model(features, ids, dict(episode_keys=["a", "b"], timesteps=[4, 4],
                                             cognitive_history_mode="other_episode"))
                self.assertEqual(model.cognitive.last_written, before["cognitive"]["last_written"])
                self.assertFalse(model.perceptual.bank)
                self.assertIsNone(model.cognitive.read_overrides)
                self.assertIsNone(model.cognitive.records)

    def test_wrong_history_does_not_repeat_slots_or_read_donor_future(self):
        _, model = adapter()
        model.cognitive.bank = {"a": [(1, torch.randn(1, 32))],
                                "b": [(10, torch.randn(1, 32))]}
        model.cognitive.last_written = {"a": 1, "b": 10}
        features, ids = observation(2)
        with self.assertRaisesRegex(ValueError, "matched donor"):
            model(features, ids, dict(episode_keys=["a", "b"], timesteps=[5, 11],
                                     cognitive_history_mode="other_episode"))
        self.assertEqual(model.cognitive.last_written, {"a": 1, "b": 10})

    def test_wrong_history_with_zero_or_empty_reference_has_exact_zero_scale(self):
        for empty in (True, False):
            with self.subTest(empty=empty):
                _, model = adapter()
                if not empty:
                    model.cognitive.bank = {"a": [(0, torch.zeros(1, 32))],
                                            "b": [(0, torch.randn(1, 32))]}
                    # Both references must be matchable; zero-to-nonzero is not.
                    model.cognitive.bank["b"][0][1].zero_()
                    model.cognitive.last_written = {"a": 0, "b": 0}
                features, ids = observation(2)
                output = model(features, ids, dict(episode_keys=["a", "b"], timesteps=[1, 1],
                    cognitive_history_mode="other_episode"))
                torch.testing.assert_close(output.backbone_features[:, 5], features.backbone_features[:, 5], rtol=0, atol=0)


class CheckpointReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(51)
        for target, name, replacement in (
            (v1_model, "get_backbone_cls", lambda _: TinyBackbone),
            (processing_gr00t_n1d6, "Gr00tN1d6DataCollator", lambda **kwargs: None),
        ):
            handle = patch.object(target, name, replacement)
            handle.start()
            self.addCleanup(handle.stop)

    def test_v1_loading_requires_explicit_initialization_and_complete_old_weights(self):
        old = v1_model.Gr00tN1d6(tiny_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            old.save_pretrained(directory)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                Gr00tN1d6.from_pretrained(directory, config=tiny_config())
            model, info = Gr00tN1d6.from_pretrained(directory, config=tiny_config(),
                initialize_from_v1=True, output_loading_info=True)
            self.assertTrue(info["missing_keys"])
            for name, value in old.state_dict().items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            for absent in ("memory.perceptual.retrieval_blocks.0.q_proj.weight",
                           "memory.cognitive.retrieval_blocks.0.q_proj.weight",
                           "backbone.projection.weight"):
                weights = dict(old.state_dict())
                weights.pop(absent)
                old.save_pretrained(directory, state_dict=weights)
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    Gr00tN1d6.from_pretrained(directory, config=tiny_config(), initialize_from_v1=True)

    def test_v2_reload_rejects_missing_guidance_and_unexpected_weights(self):
        model = Gr00tN1d6(tiny_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            for absent in ("memory.cognitive.state_head.proj.weight",
                           "memory.cognitive.retrieval_blocks.0.state_query.weight",
                           "memory.cognitive.gate_fusion_blocks.state_gate.weight"):
                weights = dict(model.state_dict())
                weights.pop(absent)
                model.save_pretrained(directory, state_dict=weights)
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    Gr00tN1d6.from_pretrained(directory)
            weights = dict(model.state_dict(), unrelated_parameter=torch.ones(1))
            model.save_pretrained(directory, state_dict=weights)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                Gr00tN1d6.from_pretrained(directory)

    def test_shape_mismatch_cannot_be_ignored_by_public_loader(self):
        model = Gr00tN1d6(tiny_config()).eval()
        weights = dict(model.state_dict())
        weights["memory.cognitive.state_head.proj.weight"] = torch.ones(1, 8)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, state_dict=weights)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                Gr00tN1d6.from_pretrained(directory, ignore_mismatched_sizes=True)

    def test_legacy_config_override_is_rejected_instead_of_changing_on_reload(self):
        old = v1_model.Gr00tN1d6(tiny_config()).eval()
        for options in (dict(memory_residual_readout="delta"), dict(memory_length=8),
                        dict(action_horizon=4)):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "state configuration"):
                    Gr00tN1d6.from_v1_model(old, state_labels=LABELS, **options)

    def test_allowed_state_overrides_match_runtime_after_reload(self):
        old = v1_model.Gr00tN1d6(tiny_config()).eval()
        model = Gr00tN1d6.from_v1_model(old, state_labels=["started"], state_loss_weight=.7,
            memory_state_guidance=False, memory_event_low=.1, memory_event_high=.9, memory_recent_slots=1)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = Gr00tN1d6.from_pretrained(directory)
            for candidate in (model, restored):
                bank = candidate.memory.cognitive
                self.assertEqual(candidate.config.memory_residual_readout, candidate.memory.residual_readout)
                self.assertEqual(candidate.config.memory_length, bank.capacity)
                self.assertEqual(candidate.config.memory_state_guidance, bank.guidance_enabled)
                self.assertEqual(candidate.config.memory_event_low, bank.event_low)
                self.assertEqual(candidate.config.memory_event_high, bank.event_high)
                self.assertEqual(candidate.config.memory_recent_slots, bank.recent_slots)
                self.assertEqual(candidate.config.memory_state_loss_weight, .7)


if __name__ == "__main__":
    unittest.main()
