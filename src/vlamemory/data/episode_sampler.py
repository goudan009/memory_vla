"""Episode-aware samplers for memory-based VLA training.

Ensures batches don't mix samples from different episodes, which is critical
for maintaining consistent memory bank state across a batch.
"""

from typing import Iterator, Optional
import torch
from torch.utils.data import Sampler, Dataset
import numpy as np


class EpisodeBatchSampler(Sampler[list[int]]):
    """Sampler that ensures each batch contains samples from only one episode.

    This is critical for memory-based policies where the memory bank state
    must be consistent across the batch (all samples at same episode stage).

    Args:
        dataset: Dataset with episode_data_index attribute
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle episodes and samples within episodes
        drop_last: Whether to drop incomplete batches
        seed: Random seed for reproducibility

    Yields:
        Lists of indices, each list is one batch from one episode
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Extract episode boundaries
        if not hasattr(dataset, 'episode_data_index'):
            raise ValueError(
                "Dataset must have 'episode_data_index' attribute. "
                "Use LeRobotDataset or HistoricalFrameWrapper."
            )

        self.episode_data_index = dataset.episode_data_index
        self._build_episode_batches()

    def _build_episode_batches(self):
        """Pre-compute all valid batches per episode."""
        self.episode_batches = []

        for episode_id, bounds in self.episode_data_index.items():
            start_idx = bounds['from']
            end_idx = bounds['to']
            episode_length = end_idx - start_idx

            # Create indices for this episode
            episode_indices = list(range(start_idx, end_idx))

            # Split into batches
            num_batches = episode_length // self.batch_size
            if num_batches == 0 and not self.drop_last:
                # Episode shorter than batch_size, include as partial batch
                self.episode_batches.append((episode_id, episode_indices))
            else:
                for i in range(num_batches):
                    batch_start = i * self.batch_size
                    batch_end = batch_start + self.batch_size
                    batch_indices = episode_indices[batch_start:batch_end]
                    self.episode_batches.append((episode_id, batch_indices))

                # Handle last partial batch
                if not self.drop_last and episode_length % self.batch_size != 0:
                    last_batch_indices = episode_indices[num_batches * self.batch_size:]
                    self.episode_batches.append((episode_id, last_batch_indices))

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over batches."""
        # Set random state
        rng = np.random.RandomState(self.seed + self.epoch)

        # Shuffle episode batches if requested
        if self.shuffle:
            batch_order = rng.permutation(len(self.episode_batches))
        else:
            batch_order = np.arange(len(self.episode_batches))

        for batch_idx in batch_order:
            episode_id, indices = self.episode_batches[batch_idx]

            # Optionally shuffle within batch (preserves episode boundary)
            if self.shuffle:
                indices = [indices[i] for i in rng.permutation(len(indices))]

            yield indices

    def __len__(self) -> int:
        """Number of batches."""
        return len(self.episode_batches)

    def set_epoch(self, epoch: int):
        """Set epoch for distributed training."""
        self.epoch = epoch


class DistributedEpisodeBatchSampler(Sampler[list[int]]):
    """Distributed version of EpisodeBatchSampler.

    Splits episode batches across ranks while maintaining episode boundaries.

    Args:
        dataset: Dataset with episode_data_index
        batch_size: Per-rank batch size
        num_replicas: Number of distributed processes
        rank: Rank of current process
        shuffle: Whether to shuffle
        drop_last: Whether to drop incomplete batches
        seed: Random seed
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        if num_replicas is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Build base sampler for rank 0, then split
        self.base_sampler = EpisodeBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over batches for this rank."""
        self.base_sampler.set_epoch(self.epoch)

        # Distribute batches across ranks
        all_batches = list(self.base_sampler)

        # Each rank gets every num_replicas-th batch
        rank_batches = all_batches[self.rank::self.num_replicas]

        for batch in rank_batches:
            yield batch

    def __len__(self) -> int:
        """Number of batches for this rank."""
        total_batches = len(self.base_sampler)
        # Integer division - some ranks may get one fewer batch
        return (total_batches + self.num_replicas - 1 - self.rank) // self.num_replicas

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling."""
        self.epoch = epoch
