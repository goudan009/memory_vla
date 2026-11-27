"""Atomic same-Episode groups drawn from the online decision clock."""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from v1_groot_memoryvla.data.types import EmbodimentTag
from .lerobot_episode_loader import LeRobotEpisodeLoader
from .sharded_single_step_dataset import extract_step_data


class HistoryDataset(Dataset):
    """One item is one complete ordered group, never one shuffled frame.

    RGB/state/language at t are observations; t..t+31 actions are labels only.
    Short tail groups contain real observations only. Future action padding is
    masked after normalization and never becomes a memory observation.
    DDP and worker partitioning happen on group indices before collating frames.
    """

    def __init__(self, dataset_paths, processor, embodiment_tag="behavior_r1_pro",
                 group_size=16, history_stride=8, video_backend="pyav", seed=42,
                 history_sampling="contiguous"):
        self.processor = processor
        self.seed = seed
        self.group_size = group_size
        self.history_stride = history_stride
        if history_sampling not in ("contiguous", "random_ordered"):
            raise ValueError("Unknown history sampling policy")
        self.history_sampling = history_sampling
        self.embodiment_tag = EmbodimentTag(embodiment_tag)
        modalities = processor.modality_configs[embodiment_tag]
        for name in ("video", "state", "language"):
            if list(modalities[name].delta_indices) != [0]:
                raise ValueError(f"v1 observes only the current {name}")
        self.low_dim_modalities = {k: v for k, v in modalities.items() if k != "video"}
        self.loaders = [LeRobotEpisodeLoader(Path(p), modalities, video_backend=video_backend)
                        for p in dataset_paths]
        # Persist the same task-name instruction protocol used by B1K rollout.
        # Full demonstrator instructions remain untouched in the source data.
        processor.configure_b1k_language([Path(p) / "meta/tasks.jsonl" for p in dataset_paths])
        self.action_deltas = np.asarray(modalities["action"].delta_indices)
        self.episode_lengths = {}
        self.groups = []
        for dataset_index, loader in enumerate(self.loaders):
            for episode in loader.episodes_metadata:
                end = episode["length"]
                self.episode_lengths[dataset_index, episode["episode_index"]] = end
                if history_sampling == "random_ordered" and len(range(0, end, history_stride)) < group_size:
                    raise ValueError(f"Episode {episode['episode_index']} has fewer than {group_size} decision frames")
                starts = [0] if history_sampling == "random_ordered" else range(0, end, group_size * history_stride)
                for start in starts:
                    self.groups.append((dataset_index, episode["episode_index"], start))
        if not self.groups:
            raise ValueError("No observations available for history training")
        self._cached_episode = None
        self._cached_low_dim = None
        # Keep the official checkpoint's normalization for the first v1 run.
        self.global_stats = {embodiment_tag: self.loaders[0].get_dataset_statistics()}
        for loader in self.loaders:
            if loader.fps != 30:
                raise ValueError("v1's stride=8 assumes the B1K 30 Hz control clock")

    def __len__(self):
        return len(self.groups)

    def get_dataset_statistics(self):
        return self.global_stats

    def __getitem__(self, index):
        return self.get_group(index)

    def frame_indices(self, index, sampling_seed=None):
        dataset_index, episode_id, start = self.groups[index]
        length = self.episode_lengths[dataset_index, episode_id]
        if self.history_sampling == "random_ordered":
            # Use a local generator: worker prefetch and image augmentation must
            # not affect history selection or reconstruction after a resume.
            rng = np.random.default_rng(self.seed + index if sampling_seed is None else sampling_seed)
            candidates = np.arange(0, length, self.history_stride)
            return np.sort(rng.choice(candidates, self.group_size, replace=False))
        return np.arange(start, min(start + self.group_size * self.history_stride, length), self.history_stride)

    def get_group(self, index, sampling_seed=None):
        dataset_index, episode_id, start = self.groups[index]
        loader = self.loaders[dataset_index]
        cache_key = (dataset_index, episode_id)
        if cache_key != self._cached_episode:
            self._cached_low_dim = loader._load_parquet_data(episode_id)
            self._cached_episode = cache_key
        length = self.episode_lengths[cache_key]
        frames = self.frame_indices(index, sampling_seed)
        videos = loader._load_video_data(episode_id, frames)
        steps = []
        for i, frame in enumerate(frames):
            step = extract_step_data(self._cached_low_dim, int(frame), self.low_dim_modalities,
                                     self.embodiment_tag, allow_padding=True)
            step.images = {key: [images[i]] for key, images in videos.items()}
            processed = self.processor([{"type": "episode_step", "content": step}])
            valid = (frame + self.action_deltas >= 0) & (frame + self.action_deltas < length)
            # Preserve normalization and relative-action conversion; discard
            # out-of-episode placeholders from the supervised loss.
            invalid = np.flatnonzero(~valid)
            processed["action_mask"][invalid] = 0
            processed["action"][invalid] = 0
            steps.append(processed)
        return {"steps": steps, "episode_key": f"dataset{dataset_index}/episode{episode_id}",
                "group_id": int(index), "frame_indices": frames.tolist(),
                "timesteps": (frames // self.history_stride).tolist()}


class HistoryCollator:
    def __init__(self, step_collator, group_size=16, history_stride=8, history_sampling="contiguous"):
        self.step_collator = step_collator
        self.group_size = group_size
        self.history_stride = history_stride
        self.history_sampling = history_sampling

    def __call__(self, groups):
        steps, keys, times, frames, group_ids, lengths = [], [], [], [], [], []
        for group in groups:
            indices = group["frame_indices"]
            length = len(indices)
            if not 1 <= length <= self.group_size or len(group["steps"]) != length:
                raise ValueError("Incomplete memory group")
            gaps = np.diff(indices)
            if self.history_sampling == "random_ordered":
                if length != self.group_size or np.any(gaps <= 0) or any(i % self.history_stride for i in indices):
                    raise ValueError("Random history must contain distinct ordered decision frames")
            elif gaps.tolist() != [self.history_stride] * (length - 1):
                raise ValueError("History frames must be ordered at the online stride")
            if group["timesteps"] != [f // self.history_stride for f in indices]:
                raise ValueError("Memory time must count online decisions")
            steps.extend(group["steps"])
            lengths.append(length)
            keys.extend([group["episode_key"]] * length)
            times.extend(group["timesteps"])
            frames.extend(indices)
            group_ids.extend([group["group_id"]] * length)
        batch = self.step_collator(steps)
        batch["inputs"]["memory_context"] = {"mode": "group", "sequence_length": self.group_size,
            "group_lengths": lengths,
            "episode_keys": keys,
            "timesteps": times, "frame_indices": frames, "group_ids": group_ids}
        return batch
