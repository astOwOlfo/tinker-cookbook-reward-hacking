import asyncio
import logging
from dataclasses import dataclass
from collections.abc import Sequence
from datetime import datetime
import os, json
import re
import time
import random
from typing import Optional
import chz
from dotenv import load_dotenv
from tinker_cookbook import cli_utils
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    Observation,
    StopCondition,
    StepResult,
    Action,
)
from tinker_cookbook.rl import train
from tinker_cookbook.rl.features.feature import Feature


class LimitSizeDataset(RLDataset):
    def __init__(
        self,
        inner_dataset: RLDataset,
        max_batches: int,
    ) -> None:
        print(f"{max_batches=} {len(inner_dataset)=} {type(inner_dataset)=}")
        assert max_batches <= len(inner_dataset), (
            "Tried to limit the size of a dataset with LimitSize but the size of the inner dataset is smaller than max_size"
        )
        self.inner_dataset = inner_dataset
        self.max_batches = max_batches

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        assert index < self.max_batches, "Index out of range"
        return self.inner_dataset.get_batch(index)

    def __len__(self) -> int:
        return min(self.max_batches, len(self.inner_dataset))

    def display_inner_dataset(self, logger: logging.Logger, indent: int = 0) -> None:
        """Recursively display inner dataset with indentation."""
        indent_str = "  " * indent
        logger.info(f"{indent_str}LimitSizeDataset (length: {len(self)} batches)")
        if hasattr(self.inner_dataset, "display_inner_datasets"):
            self.inner_dataset.display_inner_datasets(logger, indent + 1)
        elif hasattr(self.inner_dataset, "display_inner_dataset"):
            self.inner_dataset.display_inner_dataset(logger, indent + 1)
        else:
            logger.info(
                f"{indent_str}  {self.inner_dataset.__class__.__name__} (length: {len(self.inner_dataset)} batches)"
            )


@dataclass(frozen=True, slots=True)
class LimitSizeDatasetBuilder(RLDatasetBuilder):
    inner_builder: RLDatasetBuilder
    max_batches: int

    async def __call__(self) -> tuple[LimitSizeDataset, LimitSizeDataset]:
        train_dataset, test_dataset = await self.inner_builder()

        return (
            LimitSizeDataset(train_dataset, self.max_batches),
            LimitSizeDataset(test_dataset, self.max_batches),
        )


class _LimitSize:
    def __init__(self) -> None:
        pass

    def __call__(
        self, inner_builder: RLDatasetBuilder, max_batches: int
    ) -> LimitSizeDatasetBuilder:
        return LimitSizeDatasetBuilder(inner_builder, max_batches)


LimitSize = _LimitSize()


class SkipFirstDataset(RLDataset):
    def __init__(
        self,
        inner_dataset: RLDataset,
        skip_first_n_batches: int,
    ) -> None:
        self.inner_dataset = inner_dataset
        self.skip_first_n_batches = skip_first_n_batches

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return self.inner_dataset.get_batch(index + self.skip_first_n_batches)

    def __len__(self) -> int:
        return max(0, len(self.inner_dataset) - self.skip_first_n_batches)

    def display_inner_dataset(self, logger: logging.Logger, indent: int = 0) -> None:
        """Recursively display inner dataset with indentation."""
        indent_str = "  " * indent
        logger.info(f"{indent_str}SkipFirstDataset (length: {len(self)} batches)")
        if hasattr(self.inner_dataset, "display_inner_datasets"):
            self.inner_dataset.display_inner_datasets(logger, indent + 1)
        elif hasattr(self.inner_dataset, "display_inner_dataset"):
            self.inner_dataset.display_inner_dataset(logger, indent + 1)
        else:
            logger.info(
                f"{indent_str}  {self.inner_dataset.__class__.__name__} (length: {len(self.inner_dataset)} batches)"
            )


@dataclass(frozen=True, slots=True)
class SkipFirstDatasetBuilder(RLDatasetBuilder):
    inner_builder: RLDatasetBuilder
    skip_first_n_batches: int

    async def __call__(self) -> tuple[SkipFirstDataset, SkipFirstDataset]:
        train_dataset, test_dataset = await self.inner_builder()

        return (
            SkipFirstDataset(train_dataset, self.skip_first_n_batches),
            SkipFirstDataset(test_dataset, self.skip_first_n_batches),
        )


class _SkipFirst:
    def __init__(self) -> None:
        pass

    def __call__(
        self, inner_builder: RLDatasetBuilder, skip_first_n_batches: int
    ) -> SkipFirstDatasetBuilder:
        return SkipFirstDatasetBuilder(inner_builder, skip_first_n_batches)


SkipFirst = _SkipFirst()
