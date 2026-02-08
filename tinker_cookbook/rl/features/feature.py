import asyncio
from dataclasses import dataclass
from collections.abc import Sequence
from datetime import datetime
import os, json
import re
import time
import random
from typing import Optional
import chz
from typing import TypeVar, Self
import logging

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

C = TypeVar("C")


def build_dataset_builder(
    dataset_builder: RLDatasetBuilder, feature_env_class: type[Env], feature_config: C
) -> RLDatasetBuilder:
    @dataclass(frozen=True, slots=True)
    class FeatureGroupBuilder(EnvGroupBuilder):
        env_group_builder: EnvGroupBuilder
        feature_config: C

        async def make_envs(self) -> list[Env]:
            envs = await self.env_group_builder.make_envs()
            return [feature_env_class(env, self.feature_config) for env in envs]

        def logging_tags(self) -> list[str]:
            logging_tags = self.env_group_builder.logging_tags()
            if hasattr(self.feature_config, "logging_tags"):
                new_tag = self.feature_config.logging_tags()
                assert isinstance(new_tag, list) and len(new_tag) == 1
                logging_tags = [tag + "_" + new_tag[0] for tag in logging_tags]
            return logging_tags

    class FeatureDataset(RLDataset):
        def __init__(
            self,
            dataset: RLDataset,
            feature_config: C,
        ) -> None:
            self.inner_dataset = dataset
            self.feature_config = feature_config

        def get_batch(self, index: int) -> Sequence[FeatureGroupBuilder]:
            batch = self.inner_dataset.get_batch(index)
            return [
                FeatureGroupBuilder(env_group_builder, self.feature_config)
                for env_group_builder in batch
            ]

        def __len__(self) -> int:
            return len(self.inner_dataset)

        def display_inner_dataset(self, logger: logging.Logger, indent: int = 0) -> None:
            """Recursively display inner dataset with indentation."""
            indent_str = "  " * indent
            feature_name = getattr(
                self.feature_config, "__class__", type(self.feature_config)
            ).__name__
            logger.info(f"{indent_str}FeatureDataset[{feature_name}] (length: {len(self)} batches)")
            if hasattr(self.inner_dataset, "display_inner_datasets"):
                self.inner_dataset.display_inner_datasets(logger, indent + 1)
            elif hasattr(self.inner_dataset, "display_inner_dataset"):
                self.inner_dataset.display_inner_dataset(logger, indent + 1)
            else:
                logger.info(
                    f"{indent_str}  {self.inner_dataset.__class__.__name__} (length: {len(self.inner_dataset)} batches)"
                )

    @dataclass(frozen=True, slots=True)
    class FeatureDatasetBuilder(RLDatasetBuilder):
        inner_dataset_builder: RLDatasetBuilder
        feature_config: C

        async def __call__(self) -> tuple[FeatureDataset, FeatureDataset | None]:
            train_data, test_data = await self.inner_dataset_builder()

            return (
                FeatureDataset(train_data, self.feature_config),
                FeatureDataset(test_data, self.feature_config) if test_data is not None else None,
            )

    return FeatureDatasetBuilder(
        inner_dataset_builder=dataset_builder, feature_config=feature_config
    )


class Feature:
    def __init__(self, feature_env_class: type[Env]) -> Self:
        self.feature_env_class = feature_env_class

    def __call__(self, env_cfg: train.Config, feature_cfg: C) -> train.Config:
        # Replace the dataset builder with our own
        new_cfg = chz.replace(
            env_cfg,
            dataset_builder=build_dataset_builder(
                dataset_builder=env_cfg.dataset_builder,
                feature_env_class=self.feature_env_class,
                feature_config=feature_cfg,
            ),
        )

        return new_cfg
