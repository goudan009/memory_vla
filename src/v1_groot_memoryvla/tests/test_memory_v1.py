"""Fast mechanism tests; official-weight acceptance is a separate GPU command."""

from copy import deepcopy
import ast
import math
import json
from pathlib import Path
import tempfile
from typing import Optional
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.feature_extraction_utils import BatchFeature

from v1_groot_memoryvla.model.memory.adapter import MemoryAdapter
from v1_groot_memoryvla.model.memory.bank import MemoryBank
from v1_groot_memoryvla.policy.memory_state import MemoryState
from v1_groot_memoryvla.configs.model.gr00t_n1d6 import Gr00tN1d6Config


def features(batch=16):
    h = torch.randn(batch, 22, 32)
    mask = torch.zeros(batch, 22, dtype=torch.bool)
    mask[:, 5:17] = True
    attention = torch.ones(batch, 22, dtype=torch.bool)
    attention[:, :2] = False
    ids = torch.zeros(batch, 22, dtype=torch.long)
    ids[:, 3] = 151645  # system EOS must not be selected
    ids[:, 20] = 151645
    return BatchFeature(data={"backbone_features": h, "image_mask": mask,
        "backbone_attention_mask": attention, "visual_features": torch.randn(batch, 3, 4, 32)}), ids


class MemoryV1Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        self.adapter = MemoryAdapter(vlm_dim=32, perceptual_dim=32)

    def test_empty_history_is_exact_and_masks_unchanged(self):
        self.adapter.eval()
        original, ids = features(1)
        result = self.adapter(original, ids, {"episode_keys": ["a"], "timesteps": [0]})
        self.assertTrue(torch.equal(result.backbone_features, original.backbone_features))
        self.assertIs(result.image_mask, original.image_mask)
        self.assertIs(result.backbone_attention_mask, original.backbone_attention_mask)
        self.assertEqual(result.backbone_features.shape, original.backbone_features.shape)

    def test_full_memory_configuration_is_explicit_in_checkpoint(self):
        config = Gr00tN1d6Config(memory_enabled=True)
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            serialized = json.loads((Path(directory) / "config.json").read_text())
            expected = {name for name in config.__dataclass_fields__ if name.startswith("memory_")}
            self.assertTrue(expected.issubset(serialized))
            restored = Gr00tN1d6Config.from_pretrained(directory)
            for name in expected:
                self.assertEqual(getattr(config, name), getattr(restored, name))

    def test_groups_match_online_and_cannot_see_future(self):
        original, ids = features()
        context = {"mode": "group", "episode_keys": ["a"] * 16, "timesteps": list(range(16))}
        self.adapter.train()
        grouped = self.adapter(original, ids, context).backbone_features
        self.assertFalse(self.adapter.perceptual.bank)
        self.adapter.eval()
        online = []
        for t in range(16):
            sample = BatchFeature(data={k: v[t:t + 1] for k, v in original.items()})
            online.append(self.adapter(sample, ids[t:t + 1], {
                "mode": "stream", "episode_keys": ["a"], "timesteps": [t]}).backbone_features)
        torch.testing.assert_close(grouped, torch.cat(online), rtol=2e-5, atol=2e-6)
        changed = deepcopy(original)
        changed.backbone_features[-1] *= 10
        changed.visual_features[-1] *= -7
        self.adapter.train()
        altered = self.adapter(changed, ids, context).backbone_features
        torch.testing.assert_close(grouped[:-1], altered[:-1], rtol=0, atol=0)

    def test_current_gradients_and_detached_history(self):
        self.adapter.eval()
        original, ids = features(2)
        original.backbone_features.requires_grad_()
        original.visual_features.requires_grad_()
        first = BatchFeature(data={k: v[:1] for k, v in original.items()})
        self.adapter(first, ids[:1], {"episode_keys": ["a"], "timesteps": [0]})
        second = BatchFeature(data={k: v[1:] for k, v in original.items()})
        result = self.adapter(second, ids[1:], {"episode_keys": ["a"], "timesteps": [1]})
        result.backbone_features.square().mean().backward()
        for bank in (self.adapter.perceptual, self.adapter.cognitive):
            for name, parameter in bank.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            for _, stored in bank.bank["a"]:
                self.assertFalse(stored.requires_grad)
                self.assertIsNone(stored.grad_fn)
        self.assertGreater(float(self.adapter.perceptual_residual.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.adapter.perceptual_encoder.reduce.weight.grad.abs().sum()), 0)
        self.assertEqual(float(original.visual_features.grad[0].abs().sum()), 0)
        self.assertGreater(float(original.visual_features.grad[1].abs().sum()), 0)

    def test_write_seventeen_merges_adjacent_most_similar(self):
        bank = MemoryBank(32)
        vectors = torch.eye(17, 32)
        vectors[1] = vectors[0] * 3
        for t in range(16):
            bank._write("a", t, vectors[t:t + 1])
        self.assertEqual(len(bank.bank["a"]), 16)
        self.assertEqual(bank.merge_count.get("a", 0), 0)
        bank._write("a", 16, vectors[16:17])
        self.assertEqual(len(bank.bank["a"]), 16)
        self.assertEqual(bank.merge_count["a"], 1)
        self.assertEqual(bank.bank["a"][0][0], 0)
        torch.testing.assert_close(bank.bank["a"][0][1], vectors[0:1] * 2)

    def test_environment_and_session_resets_are_independent(self):
        state = MemoryState(self.adapter)
        self.adapter.eval()
        original, ids = features(2)
        opts = {"session_id": "s1", "env_ids": [0, 1], "episode_ids": ["ep0", "ep0"]}
        context = state.context(2, opts)
        self.adapter(original, ids, context)
        state.advance(context)
        other = state.context(1, {"session_id": "s2", "env_ids": [0], "episode_ids": ["ep0"]})
        sample = BatchFeature(data={k: v[:1] for k, v in original.items()})
        self.adapter(sample, ids[:1], other)
        state.advance(other)
        state.reset({"session_id": "s1", "env_ids": [0], "episode_ids": ["ep1"]})
        self.assertNotIn(("s1", "0", "ep0"), self.adapter.perceptual.bank)
        self.assertIn(("s1", "1", "ep0"), self.adapter.perceptual.bank)
        self.assertIn(("s2", "0", "ep0"), self.adapter.cognitive.bank)
        opts["episode_ids"][0] = "ep1"
        self.assertEqual(state.context(2, opts)["timesteps"], [0, 1])
        state.reset()
        self.assertFalse(self.adapter.cognitive.bank)
        self.assertFalse(state.slots)

    def test_parameter_reload_excludes_runtime_history(self):
        self.adapter.eval()
        original, ids = features(1)
        context = {"episode_keys": ["a"], "timesteps": [0]}
        self.adapter(original, ids, context)
        clone = MemoryAdapter(vlm_dim=32, perceptual_dim=32).eval()
        clone.load_state_dict(self.adapter.state_dict(), strict=True)
        self.assertFalse(clone.perceptual.bank)
        clone(original, ids, context)
        context["timesteps"] = [1]
        a = self.adapter(original, ids, context).backbone_features
        b = clone(original, ids, context).backbone_features
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_retrieval_and_merge_match_unmodified_reference(self):
        path = Path(__file__).resolve().parents[2] / "references_code/memoryvla/vla/memory_vla.py"
        names = {"TimestepEmbedder", "CrossTransformerBlock", "GateFusion", "CogMemBank"}
        tree = ast.parse(path.read_text())
        selected = ast.Module(body=[node for node in tree.body
                                   if isinstance(node, ast.ClassDef) and node.name in names], type_ignores=[])
        namespace = {"torch": torch, "nn": nn, "F": F, "math": math, "np": np, "Optional": Optional}
        exec(compile(selected, str(path), "exec"), namespace)
        reference = namespace["CogMemBank"]("group", 16, 32, update_fused=True).eval()
        bank = MemoryBank(32).eval()
        bank.load_state_dict(reference.state_dict(), strict=True)
        # Covers two-layer retrieval, K-only PE, Gate, fused writes and the
        # reference's retained earlier timestamp after overflow consolidation.
        for t in range(18):
            current = torch.randn(1, 4, 32)
            expected = reference.process_batch(current, ["a"], [t])
            actual = bank.read_fuse_write(current, "a", t)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        self.assertEqual([t for t, _ in bank.bank["a"]], [t for t, _ in reference.bank["a"]])


if __name__ == "__main__":
    unittest.main()
