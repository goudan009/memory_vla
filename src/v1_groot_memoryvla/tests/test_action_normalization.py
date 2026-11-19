"""The MAE panel must feed pre-vlln features through the same action path."""

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from v1_groot_memoryvla.experiment import task21_formal, task21_diagnostic
from v1_groot_memoryvla.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead


class ProbeHead(Gr00tN1d6ActionHead):
    def __init__(self):
        nn.Module.__init__(self)
        self.vlln = nn.LayerNorm(4)
        with torch.no_grad():
            self.vlln.weight.copy_(torch.tensor([.4, .8, 1.3, 1.8]))
            self.vlln.bias.copy_(torch.tensor([.2, -.1, .3, -.4]))
        self.state_encoder = lambda state, embodiment: state

    def forward(self, features, actions):
        return {"loss": self.process_backbone_output(features).backbone_features.square().mean()}

    def get_action_with_features(self, backbone_features, **kwargs):
        return {"action_pred": backbone_features[:, :1].repeat(1, 32, 1)}


def cpu_slice(features, start, end):
    return BatchFeature(data={k: v[start:end] for k, v in features.items()})


class ActionNormalizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.head = ProbeHead().eval()
        self.features = BatchFeature(data={
            "backbone_features": torch.randn(32, 3, 4) * 3 + 2,
            "backbone_attention_mask": torch.ones(32, 3, dtype=torch.bool),
            "image_mask": torch.zeros(32, 3, dtype=torch.bool)})
        self.seq = {"features": self.features, "episode": 1,
                    "timesteps": list(range(32)),
                    "frame_indices": list(range(0, 256, 8)),
                    "actions": {"action": torch.zeros(32, 32, 4),
                                "state": torch.zeros(32, 1, 4),
                                "embodiment_id": torch.zeros(32, dtype=torch.long)}}

    def test_action_head_does_not_replace_callers_features(self):
        original = self.features.backbone_features
        normalized = self.head.process_backbone_output(self.features)
        self.assertIs(self.features.backbone_features, original)
        self.assertIsNot(normalized, self.features)
        torch.testing.assert_close(normalized.backbone_features, self.head.vlln(original))
        self.assertIs(normalized.image_mask, self.features.image_mask)
        self.assertIs(normalized.backbone_attention_mask, self.features.backbone_attention_mask)

    def test_loss_collection_preserves_pre_vlln_features(self):
        with patch.object(task21_formal, "slice_features", cpu_slice), \
             patch.object(torch, "autocast", lambda *a, **kw: nullcontext()):
            _, collected, _ = task21_formal.fused_window(
                SimpleNamespace(memory=None, action_head=self.head), self.seq, 42, collect=True)
        torch.testing.assert_close(collected["backbone_features"], self.features.backbone_features,
                                   rtol=0, atol=0)

    def test_all_conditions_share_inference_and_one_vlln_per_call(self):
        calls = []
        hook = self.head.vlln.register_forward_hook(lambda *args: calls.append(1))
        with TemporaryDirectory() as directory, \
             patch.object(task21_diagnostic, "slice_features", cpu_slice), \
             patch.object(torch, "autocast", lambda *a, **kw: nullcontext()):
            root = Path(directory)
            (root / "metrics").mkdir()
            (root / "diagnostics/actions").mkdir(parents=True)
            task21_formal.action_metrics(SimpleNamespace(memory=object(), action_head=self.head),
                self.seq, self.features, self.features, root, 3000, "memory_v1", 2, empty=self.features)
            arrays = np.load(root / "diagnostics/actions/step003000_episode1.npz")
            np.testing.assert_array_equal(arrays["normal"], arrays["empty"])
            np.testing.assert_array_equal(arrays["normal"], arrays["wrong"])
        hook.remove()
        self.assertEqual(len(calls), 3 * 16)


if __name__ == "__main__":
    unittest.main()
