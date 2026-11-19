"""A checkpoint must preserve AdamW and merged, isolated Episode histories."""

from copy import deepcopy
from pathlib import Path
import tempfile

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from v3_bodymemory.experiment.task21_formal import save_checkpoint, restore_memory_runtime
from v3_bodymemory.model.memory.adapter import MemoryAdapter


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = MemoryAdapter(vlm_dim=32, perceptual_dim=32)

    def save_pretrained(self, directory, **kwargs):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(directory) / "weights.pt")


class Processor:
    def save_pretrained(self, directory):
        pass


def test_checkpoint_resume_preserves_predictions_optimizer_and_merged_history():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = SmallModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
    features = BatchFeature(data={
        "backbone_features": torch.randn(1, 15, 32),
        "visual_features": torch.randn(1, 3, 4, 32),
        "image_mask": torch.tensor([[False]+[True]*12+[False, False]]),
        "backbone_attention_mask": torch.ones(1, 15, dtype=torch.bool)})
    ids = torch.zeros(1, 15, dtype=torch.long)
    ids[:, 13] = 151645

    def forward(target, key, timestep):
        return target.memory(features, ids, {
            "mode": "train_sequence", "sequence_start": timestep == 0, "sequence_end": False,
            "episode_keys": [key], "timesteps": [timestep]}).backbone_features

    with torch.no_grad():
        for t in range(32):
            forward(model, "episode-a", t)
        for t in range(3):
            forward(model, "episode-b", t)
    forward(model, "episode-a", 32).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert model.memory.cognitive.merge_count["episode-a"] == 17
    expected_weights = deepcopy(model.state_dict())
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = save_checkpoint(model, Processor(), optimizer, root, 1, 3, 37)
        state = torch.load(root / "checkpoints/latest_training_state.pt", weights_only=False)
        assert (state["step"], state["epoch"], state["position"]) == (1, 3, 37)
        assert state["checkpoint"] == str(checkpoint)
        restored = SmallModel().train()
        restored.load_state_dict(torch.load(checkpoint / "weights.pt", weights_only=True))
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-5, foreach=False)
        restored_optimizer.load_state_dict(state["optimizer"])
        restore_memory_runtime(restored, state["memory_runtime"], "cpu")
        for name, parameter in restored.state_dict().items():
            torch.testing.assert_close(parameter, expected_weights[name], rtol=0, atol=0)
        first = forward(model, "episode-a", 33)
        second = forward(restored, "episode-a", 33)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        first.square().mean().backward()
        second.square().mean().backward()
        optimizer.step()
        restored_optimizer.step()
        for a, b in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for bank in (restored.memory.perceptual, restored.memory.cognitive):
            assert bank.last_written == {"episode-a": 33, "episode-b": 2}
            assert bank.merge_count["episode-a"] == 18
            assert all(not f.requires_grad and f.grad_fn is None for h in bank.bank.values() for _, f in h)
        restored.memory.reset(["episode-a"])
        assert "episode-b" in restored.memory.perceptual.bank
