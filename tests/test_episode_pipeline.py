"""Causal episode traversal and equal-budget experiment-arm contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlamemory.data.episode_pipeline import EpisodeStepRunner, SequentialEpisodeBatchSampler, episode_spans
from train.ablation_manifest import load_manifest, validate_manifest


def _b1k_dataset():
    # LeRobot/B1K stores episode boundaries as parallel from/to arrays.
    return SimpleNamespace(episode_data_index={"from": [0, 3], "to": [3, 5]}, episodes=[41, 99])


def test_b1k_episode_bounds_keep_real_episode_ids_and_exclusive_ends():
    assert episode_spans(_b1k_dataset()) == [(41, 0, 3), (99, 3, 5)]


def test_sequential_sampler_may_shuffle_episodes_but_never_steps():
    sampler = SequentialEpisodeBatchSampler(_b1k_dataset(), shuffle_episodes=True, seed=13)
    batches = list(sampler)
    assert len(batches) == 5
    assert sorted(i for (i,) in batches) == [0, 1, 2, 3, 4]
    assert batches in ([[0], [1], [2], [3], [4]], [[3], [4], [0], [1], [2]])


def test_sequential_sampler_rejects_overlap_or_implicit_multi_step_batches():
    bad = SimpleNamespace(episode_data_index={"from": [0, 2], "to": [3, 4]}, episodes=[1, 2])
    with pytest.raises(ValueError, match="overlap"):
        episode_spans(bad)
    with pytest.raises(ValueError, match="batch_size=1"):
        SequentialEpisodeBatchSampler(_b1k_dataset(), batch_size=2)


def test_runner_resets_once_and_reads_before_writing_each_step():
    events = []
    memory = {}

    def reset(eid):
        events.append(("reset", eid))
        memory[eid] = []

    def read(eid, t):
        events.append(("read", eid, t))
        return tuple(memory[eid])

    def write(eid, t, features):
        events.append(("write", eid, t, features))
        memory[eid].append((t, features))

    runner = EpisodeStepRunner(reset=reset, read=read, write=write)
    assert runner.advance(41, 0, lambda ctx: (ctx, "out0"), lambda output: "feat0") == ((), "out0")
    assert runner.advance(41, 1, lambda ctx: (ctx, "out1"), lambda output: "feat1") == (((0, "feat0"),), "out1")
    assert runner.advance(99, 0, lambda ctx: (ctx, "new"), lambda output: "feat2") == ((), "new")
    assert events == [
        ("reset", 41), ("read", 41, 0), ("write", 41, 0, "feat0"),
        ("read", 41, 1), ("write", 41, 1, "feat1"),
        ("reset", 99), ("read", 99, 0), ("write", 99, 0, "feat2"),
    ]


def test_runner_rejects_time_gap_and_return_to_prior_episode():
    runner = EpisodeStepRunner(reset=lambda eid: None, read=lambda eid, t: None,
                               write=lambda eid, t, feat: None)
    runner.advance(41, 0, lambda _: 1, lambda _: 2)
    with pytest.raises(ValueError, match="consecutive"):
        runner.advance(41, 2, lambda _: 1, lambda _: 2)
    runner.advance(99, 0, lambda _: 1, lambda _: 2)
    with pytest.raises(ValueError, match="revisit"):
        runner.advance(41, 0, lambda _: 1, lambda _: 2)


def test_runner_does_not_write_when_model_step_fails():
    writes = []
    runner = EpisodeStepRunner(reset=lambda eid: None, read=lambda eid, t: "history",
                               write=lambda eid, t, feat: writes.append(feat))
    def failing_step(_):
        raise RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        runner.advance(41, 0, failing_step, lambda _: "feat")
    assert writes == []
    assert runner.advance(41, 0, lambda _: "ok", lambda _: "feat") == "ok"


def test_three_arm_manifest_has_shared_budget_and_physical_23d_partition():
    path = Path(__file__).resolve().parents[1] / "configs" / "ablations" / "b1k_three_arm.json"
    manifest = load_manifest(path)
    assert set(manifest["arms"]) == {"none", "uniform", "action_conditioned"}
    assert manifest["common"]["model_action_dim"] == 32
    assert manifest["common"]["valid_action_dim"] == 23
    assert manifest["common"]["action_groups"] == {
        "base": [0, 1, 2], "trunk": [3, 4, 5, 6],
        "left_arm_gripper": list(range(7, 15)),
        "right_arm_gripper": list(range(15, 23)),
    }
    assert manifest["arms"]["uniform"]["memory"] == manifest["arms"]["action_conditioned"]["memory"]


def test_manifest_rejects_hidden_budget_mismatch_and_padded_group_dimensions():
    path = Path(__file__).resolve().parents[1] / "configs" / "ablations" / "b1k_three_arm.json"
    original = load_manifest(path)
    altered = json.loads(json.dumps(original))
    altered["arms"]["uniform"]["memory"]["mem_length"] += 1
    with pytest.raises(ValueError, match="memory budgets"):
        validate_manifest(altered)
    altered = json.loads(json.dumps(original))
    altered["common"]["action_groups"]["right_arm_gripper"].append(23)
    with pytest.raises(ValueError, match="partition"):
        validate_manifest(altered)
