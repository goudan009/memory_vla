"""Exp13 data alignment, causal sparse history and exact sampler reconstruction."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from v1_groot_memoryvla.data.dataset import history_dataset
from v1_groot_memoryvla.data.dataset.episode_windows import EpisodeWindows
from v1_groot_memoryvla.experiment.first_training import get_training_group
from v1_groot_memoryvla.experiment.memory_training_options import EXP13_PROTOCOL
from v1_groot_memoryvla.experiment.task21_ddp import FullGroupPlan, SeededTrainingGroups, resume_manifest
from v1_groot_memoryvla.model.memory.adapter import MemoryAdapter
from v1_groot_memoryvla.tests.test_memory_v1 import features


@pytest.fixture
def dataset(monkeypatch):
    class Loader:
        fps = 30
        episodes_metadata = [dict(episode_index=10, length=8001), dict(episode_index=20, length=16001)]

        def __init__(self, *args, **kwargs):
            pass

        def get_dataset_statistics(self):
            return {}

        def _load_parquet_data(self, episode):
            return episode

        def _load_video_data(self, episode, frames):
            return {camera: [(episode, int(frame), camera) for frame in frames]
                    for camera in ("head", "left", "right")}

    class Processor:
        modality_configs = {"behavior_r1_pro": {
            k: SimpleNamespace(delta_indices=[0] if k != "action" else list(range(4)))
            for k in ("video", "state", "language", "action")}}
        train_image_transform = SimpleNamespace()

        def configure_b1k_language(self, paths):
            pass

        def __call__(self, items):
            step = items[0]["content"]
            return dict(frame=step.frame, episode=step.episode, images=step.images,
                        action=np.ones((4, 1)), action_mask=np.ones((4, 1)))

    monkeypatch.setattr(history_dataset, "LeRobotEpisodeLoader", Loader)
    monkeypatch.setattr(history_dataset, "extract_step_data",
        lambda episode, frame, *args, **kwargs: SimpleNamespace(episode=episode, frame=frame))
    return EpisodeWindows(Path("/unused"), Processor(), [10, 20], group_size=16,
                          history_sampling="random_ordered", episode_tasks={10: 0, 20: 1})


def test_random_groups_align_modalities_keep_real_time_and_reconstruct(dataset):
    # Episode length does not change its sampling weight.
    assert len(dataset) == 2 and [len(v) for v in dataset.by_episode.values()] == [1, 1]
    group = get_training_group(dataset, 0, 123)
    np.random.seed(999)
    torch.manual_seed(777)
    again = get_training_group(dataset, 0, 123)
    assert group["frame_indices"] == again["frame_indices"]
    assert group["frame_indices"] != get_training_group(dataset, 0, 124)["frame_indices"]
    frames = group["frame_indices"]
    assert len(set(frames)) == 16 and frames == sorted(frames)
    assert all(0 <= f < 8001 and f % 8 == 0 for f in frames)
    assert group["timesteps"] == [f // 8 for f in frames]
    assert group["timesteps"] != list(range(16))
    assert frames[-1] - frames[0] > 120
    assert group["episode_start"] and group["episode_end"]
    for frame, step in zip(frames, group["steps"]):
        assert step["frame"] == frame and step["episode"] == 10
        for camera in ("head", "left", "right"):
            assert step["images"][camera] == [(10, frame, camera)]
        assert step["action_mask"][:, 0].tolist() == [float(frame + d < 8001) for d in range(4)]
    # Random episode access must not retain all previously decoded parquets.
    assert dataset.parquet_cache == {}


def test_plan_resume_recreates_same_group_indices_and_frame_samples(dataset):
    # 32 episodes, 16 groups per optimizer step, four groups per rank.
    dataset.groups = [(0, i, 0) for i in range(32)]
    dataset.episode_lengths = {(0, i): 8001 + i * 8 for i in range(32)}
    plan = FullGroupPlan(dataset, 16, 42)
    assert plan.steps == 2 and sorted(plan.groups.tolist()) == list(range(32))
    whole = list(plan.rank_groups(0, rank=2, world=4, first_step=1, seed=42))
    resumed = list(FullGroupPlan(dataset, 16, 42).rank_groups(1, rank=2, world=4, first_step=2, seed=42))
    assert resumed == whole[4:]
    for (group, seed), (restored_group, restored_seed) in zip(whole[4:], resumed):
        np.testing.assert_array_equal(dataset.frame_indices(group, seed),
                                      dataset.frame_indices(restored_group, restored_seed))
    assert FullGroupPlan(dataset, 16, 43).groups.tolist() != plan.groups.tolist()


def test_parallel_loader_keeps_group_order_frames_and_labels(dataset):
    groups=[(i % 2, 123+i) for i in range(8)]
    def collect(workers):
        loader=torch.utils.data.DataLoader(SeededTrainingGroups(dataset, groups),
            batch_size=None, num_workers=workers, collate_fn=lambda item: item,
            generator=torch.Generator().manual_seed(42))
        return [(group['episode'], group['frame_indices'], group['timesteps'],
                 [step['action_mask'].tolist() for step in group['steps']]) for group in loader]
    assert collect(0)==collect(2)==collect(4)


@pytest.mark.parametrize('microbatch', [4, 8, 16])
def test_sparse_history_keeps_raw_writes_causal_across_microbatches_and_resets(microbatch):
    adapter = MemoryAdapter(vlm_dim=32, perceptual_dim=32, conditioning="openvla",
                            cognitive_qk_norm="none").train()
    assert not adapter.cognitive.update_fused and not adapter.perceptual.update_fused
    observation, ids = features(16)
    times = np.sort(np.random.default_rng(42).choice(1000, 16, replace=False)).tolist()
    reads = []
    original = adapter.cognitive.read_fuse_write

    def checked(current, key, timestep, **kwargs):
        reads.append([t for t, _ in adapter.cognitive.bank.get(key, [])])
        result = original(current, key, timestep, **kwargs)
        torch.testing.assert_close(adapter.cognitive.bank[key][-1][1], current.squeeze(0), rtol=0, atol=0)
        return result

    adapter.cognitive.read_fuse_write = checked
    for start in range(0, 16, microbatch):
        end=start+microbatch
        adapter(BatchFeature(data={k: v[start:end] for k, v in observation.items()}),
                ids[start:end], dict(mode="train_sequence", episode_keys=["episode"] * microbatch,
                    timesteps=times[start:end], sequence_start=start == 0, sequence_end=end == 16))
    assert reads == [times[:i] for i in range(16)]
    assert not adapter.cognitive.bank and not adapter.perceptual.bank
    adapter(BatchFeature(data={k: v[:1] for k, v in observation.items()}), ids[:1],
            dict(mode="train_sequence", episode_keys=["episode"], timesteps=[0],
                 sequence_start=True, sequence_end=True))
    assert reads[-1] == []


def test_exp13_resume_extends_budget_but_keeps_sampling_and_data(tmp_path):
    root = tmp_path / "run/memory_v1"
    dataset = tmp_path / "data"
    (dataset / "meta").mkdir(parents=True)
    metadata = dataset / "meta/episodes.jsonl"
    metadata.write_text("episode manifest")
    (root.parent / "config").mkdir(parents=True)
    split = root.parent / "config/data_split.json"
    split.write_text("split manifest")
    (root / "config").mkdir(parents=True)
    saved = root / "checkpoints/run__step020000"
    saved.mkdir(parents=True)
    (saved / "COMPLETE.json").write_text(json.dumps(dict(step=20000, world_size=4)))
    args = SimpleNamespace(dataset=dataset, checkpoint=Path("/official"), arm="memory_v1",
        training_protocol=EXP13_PROTOCOL, history_sampling="random_ordered", memory_length=16,
        memory_conditioning="openvla", memory_cognitive_qk_norm="none", image_augmentation=True,
        steps=20000, resume_step=20000, seed=42, global_windows_per_step=16,
        microbatch=4, learning_rate=2e-5, weight_decay=0., max_grad_norm=1., optimizer_eps=1e-8)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(world_size=4, data_split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
                  episode_metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest())
    (root / "config/ddp.json").write_text(json.dumps(config))
    manifest = dict(step=20000, epoch=32, position=625, seed=42, world_size=4,
                    global_windows_per_step=16, path=str(saved))
    (root / "checkpoints/latest.json").write_text(json.dumps(manifest))
    args.steps = 40000
    assert resume_manifest(args, root, 4) == manifest
    args.microbatch=8
    args.gradient_checkpointing=True
    assert resume_manifest(args, root, 4) == manifest
    args.history_sampling = "contiguous"
    with pytest.raises(AssertionError, match="history_sampling"):
        resume_manifest(args, root, 4)
    args.history_sampling = "random_ordered"
    metadata.write_text("changed metadata")
    with pytest.raises(AssertionError, match="episode_metadata_sha256"):
        resume_manifest(args, root, 4)
