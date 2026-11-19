"""Controls for offline segment annotations and history interventions."""

from types import SimpleNamespace
import numpy as np
import torch

from v1_groot_memoryvla.data.prepare_task21_diagnostic import movement_events, panel
from v1_groot_memoryvla.experiment.task21_diagnostic import average_episodes, match_history_ages


def test_object_transport_candidates_do_not_require_unrecorded_grasp_flags():
    p = np.zeros((200, 3))
    p[40:80, 0] = np.linspace(0, .5, 40)
    p[80:, 0] = .5
    events = movement_events(p, "toy")
    assert len(events) == 2
    assert events[0]["event"] == "object_transport_start"
    assert events[1]["event"] == "object_transport_stop"
    assert not movement_events(np.zeros_like(p), "toy")
    groups = panel(1, 6000, events, "validation")
    assert any(any(g["candidate_mask"]) for g in groups)
    for g in groups:
        assert g["start"] >= 0 and g["start"] + 279 < 6000
        if any(g["candidate_mask"]):
            assert not any(g["candidate_mask"][:17])


def test_wrong_history_matches_read_ages_even_after_different_tome_pairs():
    features = [torch.tensor([i]) for i in range(3)]
    def bank():
        return SimpleNamespace(bank={"probe": list(zip([0, 3, 4], features))})
    adapter = SimpleNamespace(perceptual=bank(), cognitive=bank())
    match_history_ages(adapter, [[0, 1, 4], [0, 2, 4]])
    assert [t for t, _ in adapter.perceptual.bank["probe"]] == [0, 1, 4]
    assert [t for t, _ in adapter.cognitive.bank["probe"]] == [0, 2, 4]
    assert all(f is orig for (_, f), orig in zip(adapter.perceptual.bank["probe"], features))


def test_unequal_episode_sample_counts_do_not_change_episode_weight():
    rows = [{"episode": 1, "loss": 1.0}] + [{"episode": 2, "loss": 3.0}] * 9
    assert average_episodes(rows, "loss") == 2.0
