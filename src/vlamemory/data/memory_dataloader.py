"""DataLoader factory for memory-based VLA training.

Creates dataloaders with historical frames and episode-aware sampling.
Integrates with openpi's existing data pipeline.
"""

from typing import Optional, Sequence, Any
import torch
from torch.utils.data import DataLoader, Dataset

from .episode_sampler import EpisodeBatchSampler, DistributedEpisodeBatchSampler
from .historical_frames import create_historical_dataset


def create_memory_dataloader(
    base_dataset: Dataset,
    batch_size: int,
    observation_horizon: int,
    observation_keys: Sequence[str] = (
        'observation.images.image_0',
        'observation.images.image_1',
        'observation.images.image_2',
        'observation.state',
    ),
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    seed: int = 0,
    distributed: bool = False,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    collate_fn: Optional[Any] = None,
) -> DataLoader:
    """Create a DataLoader with episode-aware sampling and historical frames.

    This is a drop-in replacement for the standard PyTorch DataLoader that:
    1. Wraps the dataset to provide historical frame sequences
    2. Uses episode-aware sampling to prevent mixing episodes in batches
    3. Adds episode_id and timestep metadata to each sample

    Args:
        base_dataset: LeRobot dataset (with delta_timestamps configured)
        batch_size: Number of samples per batch (per rank if distributed)
        observation_horizon: Number of historical frames (T_obs)
        observation_keys: Which observations need historical frames
        shuffle: Whether to shuffle episodes and samples
        num_workers: Number of data loading workers
        drop_last: Whether to drop incomplete batches
        seed: Random seed for reproducibility
        distributed: Whether to use distributed sampler
        num_replicas: Number of processes (for distributed)
        rank: Process rank (for distributed)
        collate_fn: Custom collate function

    Returns:
        DataLoader configured for memory-based training

    Example:
        >>> from lerobot.datasets import LeRobotDataset
        >>>
        >>> # Create base dataset with delta_timestamps for observations
        >>> dataset = LeRobotDataset(
        ...     root="/path/to/data",
        ...     repo_id="task_name",
        ...     delta_timestamps={
        ...         'observation.images.image_0': [-0.1, -0.033, 0.0],  # 3 frames
        ...         'observation.state': [-0.1, -0.033, 0.0],
        ...         'action': [0.0, 0.033, 0.066],  # action horizon
        ...     }
        ... )
        >>>
        >>> # Create memory-aware dataloader
        >>> loader = create_memory_dataloader(
        ...     base_dataset=dataset,
        ...     batch_size=16,
        ...     observation_horizon=3,
        ...     shuffle=True,
        ... )
        >>>
        >>> for batch in loader:
        ...     # batch['observation']['images']['image_0']: (B, T_obs, C, H, W)
        ...     # batch['observation']['state']: (B, T_obs, state_dim)
        ...     # batch['action']: (B, action_horizon, action_dim)
        ...     # batch['episode_id']: (B,) - all same value within batch
        ...     # batch['timestep']: (B,) - timesteps within episode
        ...     pass
    """
    # Wrap dataset for historical frames and episode metadata
    wrapped_dataset = create_historical_dataset(
        base_dataset=base_dataset,
        observation_horizon=observation_horizon,
        observation_keys=observation_keys,
    )

    # Create episode-aware sampler
    if distributed:
        sampler = DistributedEpisodeBatchSampler(
            dataset=wrapped_dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
    else:
        sampler = EpisodeBatchSampler(
            dataset=wrapped_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )

    # Create DataLoader with batch sampler
    return DataLoader(
        wrapped_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )


def get_delta_timestamps_for_observations(
    observation_keys: Sequence[str],
    observation_horizon: int,
    fps: float = 30.0,
) -> dict[str, list[float]]:
    """Helper to generate delta_timestamps for observation keys.

    Args:
        observation_keys: Keys that need historical frames
        observation_horizon: Number of frames (T_obs)
        fps: Dataset framerate

    Returns:
        dict mapping observation keys to delta timestamp lists

    Example:
        >>> deltas = get_delta_timestamps_for_observations(
        ...     observation_keys=['observation.images.image_0', 'observation.state'],
        ...     observation_horizon=3,
        ...     fps=30.0,
        ... )
        >>> # Returns: {
        >>> #   'observation.images.image_0': [-0.066, -0.033, 0.0],
        >>> #   'observation.state': [-0.066, -0.033, 0.0],
        >>> # }
    """
    # Generate timestamps: [-(T-1)/fps, ..., -1/fps, 0.0]
    timestamps = [-(observation_horizon - 1 - i) / fps for i in range(observation_horizon)]

    return {key: timestamps for key in observation_keys}
