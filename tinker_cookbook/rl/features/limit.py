import asyncio
from dataclasses import dataclass
from collections.abc import Sequence
from datetime import datetime
import os,json
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
        max_dataset_size: int,
    ) -> None:
        self.inner_dataset = inner_dataset
        self.max_dataset_size = max_dataset_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        assert index < self.max_dataset_size, "Index out of range"
        return self.inner_dataset.get_batch(index)
    
    def __len__(self) -> int:
        return min(len(self.inner_dataset), self.max_dataset_size)


@dataclass(frozen=True, slots=True)
class LimitSizeDatasetBuilder(RLDatasetBuilder):
    inner_builder: RLDatasetBuilder
    max_dataset_size: int
    async def __call__(self) -> tuple[LimitSizeDataset, LimitSizeDataset]:
        train_dataset, test_dataset = await self.inner_builder()
            
        return (
            LimitSizeDataset(train_dataset, self.max_dataset_size),
            LimitSizeDataset(test_dataset, self.max_dataset_size),
        )
        
class LimitSize:
    def __init__(self, inner_builder: RLDatasetBuilder, max_dataset_size: int) -> LimitSizeDatasetBuilder:
        return LimitSizeDatasetBuilder(inner_builder, max_dataset_size)
