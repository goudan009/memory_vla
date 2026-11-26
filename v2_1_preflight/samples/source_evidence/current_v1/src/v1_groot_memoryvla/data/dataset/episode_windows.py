"""Chronological 32-decision windows with independent Episode lifecycles."""

from collections import defaultdict
import numpy as np

from .history_dataset import HistoryDataset


class EpisodeWindows(HistoryDataset):
    def __init__(self, dataset, processor, episode_ids, seed=42, episode_tasks=None, group_size=32,
                 history_sampling="contiguous"):
        super().__init__([dataset], processor, group_size=group_size, history_stride=8, seed=seed,
                         history_sampling=history_sampling)
        selected = set(episode_ids)
        self.task_by_episode = {e: episode_tasks[e] for e in selected} if episode_tasks else {}
        self.is_multitask = len(set(self.task_by_episode.values())) > 1
        self.groups = [g for g in self.groups if g[1] in selected]
        self.by_episode = defaultdict(list)
        for i, (_, eid, _) in enumerate(self.groups):
            self.by_episode[eid].append(i)
        assert set(self.by_episode) == selected
        self.position = {i: p for indices in self.by_episode.values() for p, i in enumerate(indices)}
        self.parquet_cache = {}

    def get_group(self, index, sampling_seed=None):
        key = self.groups[index][:2]
        if self.history_sampling == "contiguous":
            if key not in self.parquet_cache:
                self.parquet_cache[key] = self.loaders[key[0]]._load_parquet_data(key[1])
            self._cached_episode, self._cached_low_dim = key, self.parquet_cache[key]
        result = super().get_group(index, sampling_seed)
        eid = self.groups[index][1]
        result.update(episode=eid, window=self.position[index],
            episode_start=self.position[index] == 0,
            episode_end=self.position[index] == len(self.by_episode[eid])-1)
        if self.task_by_episode:
            result["task"] = self.task_by_episode[eid]
        return result

    def order(self, seed):
        """Shuffle episodes, then consume each episode's windows consecutively."""
        ids = np.array(sorted(self.by_episode))
        np.random.default_rng(seed).shuffle(ids)
        return [index for e in ids for index in self.by_episode[int(e)]]
