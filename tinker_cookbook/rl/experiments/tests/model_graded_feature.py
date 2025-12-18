from openai import AsyncOpenAI
from datasets import load_dataset
from xml.etree import ElementTree
import json
from json.decoder import JSONDecodeError
import logging
from shlex import quote
from sys import stderr
import random
import traceback
from itertools import count
import asyncio
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from collections.abc import Awaitable
from typing import Callable, Sequence

from scalable_docker.client import (
    ScalableDockerClient,
    Container,
    ProcessOutput,
    MultiCommandTimeout,
    TIMED_OUT_PROCESS_OUTPUT,
    upload_file_command,
    Image,
)

import tinker
from tinker_cookbook import renderers, model_info, cli_utils
from tinker_cookbook.completers import StopCondition
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
from tinker_cookbook.rl.envs.bash_apps_env import (
    load_apps_dataset,
    BashAppsDatasetBuilder,
    BashAppsEnvConfig,
)
from tinker_cookbook.rl.envs.bash_apps_env import DOCKERFILE_CONTENT
from tinker_cookbook.rl.features.monitored import MonitorConfig, Monitored

logger = logging.getLogger(__name__)


def build_config() -> train.Config:
    model_name = "Qwen/Qwen3-8B"

    config = train.Config(
        model_name=model_name,
        log_path="/tmp/tinker-examples/bash_apps_rl",
        dataset_builder=BashAppsDatasetBuilder(
            batch_size=64,
            model_name_for_tokenizer=model_name,
            renderer_name=model_info.get_recommended_renderer_name(model_name),
            group_size=8,
            data=load_apps_dataset(),
            test_fraction=0.1,
            cfg=BashAppsEnvConfig(),
        ),
        learning_rate=4e-5,
        max_tokens=8192,
        eval_every=0,
        wandb_project="tinker",
        wandb_name="qwen3-8b-gpt-5-mini-grader",
    )

    new_config = Monitored(
        env_cfg=config,
        feature_cfg=MonitorConfig(
            model_name="gpt-5-mini-2025-08-07",
            model_provider="openai",
        ),
    )

    return new_config


def build_docker_image() -> None:
    client = ScalableDockerClient(key="bash_apps")
    asyncio.run(client.build_images([Image(DOCKERFILE_CONTENT)]))


def main() -> None:
    config = build_config()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    build_docker_image()
    main()
