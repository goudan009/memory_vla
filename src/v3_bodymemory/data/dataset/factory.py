import numpy as np
import torch
from tqdm import tqdm

from v3_bodymemory.configs.base_config import Config
from v3_bodymemory.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from v3_bodymemory.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from v3_bodymemory.data.embodiment_tags import EmbodimentTag
from v3_bodymemory.data.interfaces import BaseProcessor
from v3_bodymemory.data.stats import generate_rel_stats, generate_stats
from v3_bodymemory.experiment.dist_utils import barrier


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        if self.config.model.memory_enabled:
            from .history_dataset import HistoryDataset
            specs = self.config.data.datasets
            if len(specs) != 1 or specs[0].embodiment_tag != "behavior_r1_pro":
                raise ValueError("v1 history training expects one B1K dataset specification")
            if self.config.data.override_pretraining_statistics:
                raise ValueError("v1 retains the official checkpoint normalization")
            dataset = HistoryDataset(specs[0].dataset_paths, processor,
                group_size=self.config.model.memory_train_sequence_length,
                history_stride=self.config.model.memory_history_stride,
                video_backend=self.config.data.video_backend, seed=self.config.data.seed)
            return dataset, None
        assert self.config.training.eval_strategy == "no", (
            "Sharded dataset does not support evaluation sets"
        )

        all_datasets = []
        all_weights = []
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            datasets = []
            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        generate_stats(dataset_path)
                        generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                else:
                    generate_stats(dataset_path)
                    generate_rel_stats(dataset_path, EmbodimentTag(embodiment_tag))
                barrier()
                dataset = ShardedSingleStepDataset(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=self.config.data.modality_configs[embodiment_tag],
                    video_backend=self.config.data.video_backend,
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    seed=self.config.data.seed,
                    allow_padding=self.config.data.allow_padding,
                )
                datasets.append(dataset)
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)

        return (
            ShardedMixtureDataset(
                datasets=all_datasets,
                weights=all_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=True,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            ),
            None,
        )
