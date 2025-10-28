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
from typing import TypeVar, Self

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
        
C = TypeVar('C')
def build_dataset_builder(
    dataset_builder: RLDatasetBuilder,
    feature_env_class: type[Env], 
    feature_config: C) -> RLDatasetBuilder:
    
    @dataclass(frozen=True, slots=True)
    class FeatureGroupBuilder(EnvGroupBuilder):
        env_group_builder: EnvGroupBuilder
        feature_config: C

        async def make_envs(self) -> list[Env]:
            envs = await self.env_group_builder.make_envs()
            return [
                feature_env_class(env, self.feature_config)
                for env in envs
            ]
    
    class FeatureDataset(RLDataset):
        def __init__(
            self,
            dataset: RLDataset,
            feature_config: C,
        ) -> None:
            self.dataset = dataset
            self.feature_config = feature_config

        def get_batch(self, index: int) -> Sequence[FeatureGroupBuilder]:
            batch = self.dataset.get_batch(index)
            return [
                FeatureGroupBuilder(env_group_builder, self.feature_config)
                for env_group_builder in batch
            ]
            
    @dataclass(frozen=True, slots=True)
    class FeatureDatasetBuilder(RLDatasetBuilder):
        dataset_builder: RLDatasetBuilder
        feature_config: C

        async def __call__(self) -> tuple[FeatureDataset, FeatureDataset]:
            train_data, test_data = await self.dataset_builder()

            return (
                FeatureDataset(train_data, self.feature_config),
                FeatureDataset(test_data, self.feature_config),
            )
            
    return FeatureDatasetBuilder(dataset_builder, feature_config)

class Feature:
    def __init__(self, feature_env_class: type[Env]) -> Self:
        self.feature_env_class = feature_env_class
        
    def __call__(self, env_cfg: train.Config, feature_cfg: C) -> train.Config:
        
        # Replace the dataset builder with our own
        new_cfg = chz.replace(env_cfg, dataset_builder=build_dataset_builder(
            dataset_builder=env_cfg.dataset_builder,
            feature_env_class=self.feature_env_class,
            feature_config=feature_cfg,
        ))
        
        return new_cfg