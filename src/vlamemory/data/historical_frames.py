"""Historical frame window support for LeRobot datasets.

This module extends LeRobot datasets to return temporal sequences of observations
(images, states) for memory-based policies that need historical context.
"""

from typing import Any, Dict, List, Sequence
import torch
from torch.utils.data import Dataset


class HistoricalFrameWrapper(Dataset):
    """Wraps a LeRobot dataset to return historical frame windows.

    Given a base LeRobotDataset configured with delta_timestamps for observations,
    this wrapper ensures proper shape handling and provides episode_id metadata.

    Args:
        base_dataset: LeRobot dataset with delta_timestamps configured
        observation_horizon: Number of historical frames to include (T_obs)
        observation_keys: Keys for observations that need historical frames
            (e.g., ['observation.images.image_0', 'observation.state'])

    Returns batch with:
        - observations with shape (T_obs, ...) for configured keys
        - episode_id for each sample
        - timestep within episode
    """

    def __init__(
        self,
        base_dataset: Dataset,
        observation_horizon: int,
        observation_keys: Sequence[str],
    ):
        self.base_dataset = base_dataset
        self.observation_horizon = observation_horizon
        self.observation_keys = observation_keys

        # Extract episode metadata from LeRobotDataset if available
        if hasattr(base_dataset, 'episode_data_index'):
            self.episode_data_index = base_dataset.episode_data_index
            self.has_episodes = True
        else:
            self.has_episodes = False

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _get_episode_id_and_timestep(self, idx: int) -> tuple[int, int]:
        """Get episode_id and timestep for a given index."""
        if not self.has_episodes:
            # Fallback: treat each sample as separate episode
            return idx, 0

        # episode_data_index is a dict mapping episode_index -> {from, to}
        # Find which episode contains this index
        for episode_id, bounds in self.episode_data_index.items():
            if bounds['from'] <= idx < bounds['to']:
                timestep = idx - bounds['from']
                return episode_id, timestep

        # Shouldn't happen, but fallback
        return 0, idx

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample with historical frames and episode metadata."""
        # Get base sample
        sample = self.base_dataset[idx]

        # Add episode metadata
        episode_id, timestep = self._get_episode_id_and_timestep(idx)
        sample['episode_id'] = episode_id
        sample['timestep'] = timestep

        return sample


def create_historical_dataset(
    base_dataset: Dataset,
    observation_horizon: int,
    observation_keys: Sequence[str] = (
        'observation.images.image_0',
        'observation.images.image_1',
        'observation.images.image_2',
        'observation.state',
    ),
) -> HistoricalFrameWrapper:
    """Create a dataset with historical frame support.

    Args:
        base_dataset: LeRobot dataset (should have delta_timestamps configured)
        observation_horizon: Number of historical frames (T_obs)
        observation_keys: Which observation keys need historical frames

    Returns:
        Wrapped dataset that adds episode_id and timestep metadata
    """
    return HistoricalFrameWrapper(
        base_dataset=base_dataset,
        observation_horizon=observation_horizon,
        observation_keys=observation_keys,
    )
