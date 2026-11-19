"""Real episode IDs and causal order, preserving native observation objects."""

import pytest
import torch

from vlamemory.data import HistoricalFrameWrapper, create_memory_dataloader


class Dataset:
    episodes = [41, 99]
    episode_data_index = {"from": [0, 3], "to": [3, 5]}

    def __init__(self):
        self.samples = [{"native_camera": torch.full((3, 2, 2), float(i))} for i in range(5)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def test_wrapper_uses_real_ids_and_does_not_mutate_observations():
    data = Dataset()
    wrapped = HistoricalFrameWrapper(data, 1)
    assert wrapped[3]["episode_id"] == 99
    assert wrapped[3]["timestep"] == 0
    assert wrapped[4]["timestep"] == 1
    assert wrapped[3]["native_camera"] is data.samples[3]["native_camera"]
    assert "episode_id" not in data.samples[3]


def test_default_stateful_loader_keeps_step_order_and_rejects_implicit_multistep():
    loader = create_memory_dataloader(Dataset(), shuffle=False)
    assert [(int(b["episode_id"].item()), int(b["timestep"].item())) for b in loader] == [
        (41, 0), (41, 1), (41, 2), (99, 0), (99, 1)]
    with pytest.raises(ValueError, match="batch_size=1"):
        create_memory_dataloader(Dataset(), batch_size=2)
    loader = create_memory_dataloader(Dataset(), batch_size=2, stateful=False, shuffle=False, drop_last=False)
    assert [list(batch["episode_id"].numpy()) for batch in loader] == [[41, 41], [41], [99, 99]]
