from transformers import PreTrainedTokenizer
from datasets import load_dataset
from xml.etree import ElementTree
import json
from json.decoder import JSONDecodeError
import logging
from shlex import quote
from sys import stderr
import random
import traceback
import asyncio
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from collections.abc import Awaitable
from typing import Callable, Sequence, Literal
from dotenv import load_dotenv

import tinker
from tinker_cookbook import renderers, model_info, cli_utils
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.envs.tools import get_system_message_with_tools
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
    RLDataset,
    RLDatasetBuilder,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.rl import train

logger = logging.getLogger(__name__)


class CurriculumDataset(RLDataset):
    def __init__(
        self,
        inner_datasets: list[RLDataset],
    ) -> None:
        self.inner_datasets = inner_datasets
        self.dataset_lengths = [len(dataset) for dataset in inner_datasets]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        dataset_index = 0
        while index >= self.dataset_lengths[dataset_index]:
            index -= self.dataset_lengths[dataset_index]
            dataset_index += 1
        return self.inner_datasets[dataset_index].get_batch(index)
    

    def __len__(self) -> int:
        return sum(self.dataset_lengths)
    
    def display_inner_datasets(self, logger: logging.Logger, indent: int = 0) -> None:
        """Recursively display inner datasets with indentation."""
        indent_str = "  " * indent
        logger.info(f"{indent_str}CurriculumDataset (total length: {len(self)} batches)")
        for i, dataset in enumerate(self.inner_datasets):
            if hasattr(dataset, 'display_inner_datasets'):
                dataset.display_inner_datasets(logger, indent + 1)
            elif hasattr(dataset, 'display_inner_dataset'):
                dataset.display_inner_dataset(logger, indent + 1)
            else:
                logger.info(f"{indent_str}  [{i}] {dataset.__class__.__name__} (length: {len(dataset)} batches)")


@dataclass(frozen=True, slots=True)
class CurriculumDatasetBuilder(RLDatasetBuilder):
    inner_builders: list[RLDatasetBuilder]

    async def __call__(self) -> tuple[CurriculumDataset, CurriculumDataset]:
        train_datasets = []
        test_datasets = []
        for builder in self.inner_builders:
            train_dataset, test_dataset = await builder()
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
        assert len(train_datasets) == len(test_datasets)
        
        return (
            CurriculumDataset(train_datasets),
            CurriculumDataset(test_datasets),
        )
        
class _Curriculum:
    def __init__(self) -> None:
        pass
    def __call__(self, inner_builders: list[RLDatasetBuilder]) -> CurriculumDatasetBuilder:
        return CurriculumDatasetBuilder(inner_builders)

Curriculum = _Curriculum()
