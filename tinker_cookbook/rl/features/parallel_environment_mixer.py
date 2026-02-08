from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
import logging
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ParallelMixerDataset(RLDataset):
    inner_datasets: list[RLDataset]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            group_builder
            for dataset in self.inner_datasets
            for group_builder in dataset.get_batch(index)
        ]

    def __len__(self) -> int:
        for dataset in self.inner_datasets:
            print(type(dataset), type(dataset.inner_dataset), len(dataset))
        return min(len(dataset) for dataset in self.inner_datasets)

    def display_inner_datasets(self, logger: logging.Logger, indent: int = 0) -> None:
        """Recursively display inner datasets with indentation."""
        indent_str = "  " * indent
        logger.info(f"{indent_str}DatasetMixer(total length: {len(self)} batches)")
        for i, dataset in enumerate(self.inner_datasets):
            if hasattr(dataset, "display_inner_datasets"):
                dataset.display_inner_datasets(logger, indent + 1)  # type: ignore
            elif hasattr(dataset, "display_inner_dataset"):
                dataset.display_inner_dataset(logger, indent + 1)  # type: ignore
            else:
                logger.info(
                    f"{indent_str}  [{i}] {dataset.__class__.__name__} (length: {len(dataset)} batches)"
                )


@dataclass(frozen=True, slots=True)
class ParallelMixerDatasetBuilder(RLDatasetBuilder):
    inner_builders: list[RLDatasetBuilder]

    async def __call__(self) -> tuple[ParallelMixerDataset, ParallelMixerDataset]:
        train_datasets = []
        test_datasets = []
        for builder in self.inner_builders:
            train_dataset, test_dataset = await builder()
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)

        return (ParallelMixerDataset(train_datasets), ParallelMixerDataset(test_datasets))
