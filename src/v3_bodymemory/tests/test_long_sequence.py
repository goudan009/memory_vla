"""A 32-decision unroll must read history consolidated after write 17."""

from unittest.mock import patch

import torch
from transformers.feature_extraction_utils import BatchFeature

from v3_bodymemory.model.memory.adapter import MemoryAdapter
from v3_bodymemory.tests.test_memory_v1 import features


def test_long_group_and_chunked_training_match_online_after_merge():
    torch.set_num_threads(2)
    torch.manual_seed(72)
    adapter = MemoryAdapter(vlm_dim=32, perceptual_dim=32).train()
    original, ids = features(32)
    context = {"mode": "group", "sequence_length": 32,
               "episode_keys": ["a"] * 32, "timesteps": list(range(32))}
    observed = []
    read = adapter.cognitive.read_fuse_write

    def trace(current, key, timestep):
        observed.append((timestep, adapter.cognitive.merge_count.get(key, 0)))
        return read(current, key, timestep)

    with patch.object(adapter.cognitive, "read_fuse_write", side_effect=trace):
        grouped = adapter(original, ids, context).backbone_features.detach()
    assert observed[16] == (16, 0)  # Write 17 triggers the first merge.
    assert observed[17] == (17, 1)  # Decision 18 reads the merged bank.
    assert observed[-1] == (31, 15)
    assert not adapter.cognitive.bank
    chunks = []
    for start in range(0, 32, 4):
        sample = BatchFeature(data={k: v[start:start + 4] for k, v in original.items()})
        chunks.append(adapter(sample, ids[start:start + 4], {
            "mode": "train_sequence", "sequence_start": start == 0,
            "sequence_end": start == 28, "episode_keys": ["a"] * 4,
            "timesteps": list(range(start, start + 4))}).backbone_features.detach())
    torch.testing.assert_close(grouped, torch.cat(chunks), rtol=0, atol=0)
    assert not adapter.cognitive.bank
    adapter.eval()
    online = []
    for t in range(32):
        sample = BatchFeature(data={k: v[t:t + 1] for k, v in original.items()})
        online.append(adapter(sample, ids[t:t + 1], {
            "mode": "stream", "episode_keys": ["a"], "timesteps": [t]}
        ).backbone_features.detach())
    torch.testing.assert_close(grouped, torch.cat(online), rtol=0, atol=0)


def test_training_chunk_cannot_silently_start_in_middle():
    adapter = MemoryAdapter(vlm_dim=32, perceptual_dim=32).train()
    original, ids = features(1)
    import pytest
    with pytest.raises(ValueError, match="sequence_start"):
        adapter(original, ids, {"mode": "train_sequence", "sequence_start": False,
            "sequence_end": False, "episode_keys": ["a"], "timesteps": [7]})


def test_96_decisions_cross_windows_and_interleaved_episodes():
    torch.set_num_threads(2)
    torch.manual_seed(83)
    adapter = MemoryAdapter(vlm_dim=32, perceptual_dim=32).eval()
    original, ids = features(96)
    with torch.no_grad():
        expected = adapter(original, ids, {"mode": "group", "sequence_length": 96,
            "episode_keys": ["a"] * 96, "timesteps": list(range(96))}).backbone_features
        actual = []
        for start in range(0, 96, 4):
            sample = BatchFeature(data={k: v[start:start+4] for k, v in original.items()})
            actual.append(adapter(sample, ids[start:start+4], {
                "mode": "train_sequence", "sequence_start": start == 0,
                "sequence_end": False, "episode_keys": ["a"] * 4,
                "timesteps": list(range(start, start+4))}).backbone_features)
            if start in (28, 60):
                # Starting/ending another Episode must preserve the live a bank.
                before = list(adapter.cognitive.bank["a"])
                adapter(sample, ids[start:start+4], {"mode": "train_sequence",
                    "sequence_start": True, "sequence_end": True,
                    "episode_keys": ["b"] * 4, "timesteps": [0, 1, 2, 3]})
                assert "b" not in adapter.cognitive.bank
                assert all(t == u and x is y for (t, x), (u, y) in
                           zip(before, adapter.cognitive.bank["a"]))
                assert adapter.cognitive.last_written["a"] == start+3
        torch.testing.assert_close(expected, torch.cat(actual), rtol=0, atol=0)
        for bank in (adapter.perceptual, adapter.cognitive):
            assert bank.merge_count["a"] == 80
            assert len(bank.bank["a"]) == 16
            assert all(not f.requires_grad and f.grad_fn is None for _, f in bank.bank["a"])
    adapter.train()
    last, last_ids = features(1)
    last.backbone_features.requires_grad_()
    last.visual_features.requires_grad_()
    adapter(last, last_ids, {"mode": "train_sequence", "sequence_start": False,
        "sequence_end": True, "episode_keys": ["a"], "timesteps": [96]}
    ).backbone_features.square().mean().backward()
    assert adapter.perceptual.retrieval_blocks[0].q_proj.weight.grad.norm() > 0
    assert adapter.cognitive.retrieval_blocks[0].q_proj.weight.grad.norm() > 0
    assert last.visual_features.grad.norm() > 0
    assert not adapter.cognitive.bank
