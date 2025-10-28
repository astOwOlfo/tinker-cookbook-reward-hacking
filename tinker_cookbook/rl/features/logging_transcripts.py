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

class LoggingConfig:
    def __init__(self, transcripts_dir: str):
        self.transcripts_dir = transcripts_dir
            
class LoggingTranscriptsEnv(Env):
    def __init__(self, env: Env, logging_cfg: LoggingConfig):
        self.env = env
        self.logging_config = logging_cfg
        
        if not hasattr(self.env, "all_messages"):
            raise ValueError("Environment must have an all_messages attribute to keep track of the conversation1")
        
    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return await self.env.initial_observation()
        
    async def step(self, action: Action) -> StepResult:
        step_result = await self.env.step(action)
        if not step_result.episode_done:
            return step_result
        
        # Log everything
        with open(os.path.join(self.logging_config.transcripts_dir, f"rollouts_{datetime.now().strftime('%m%dT%H:%M')}.json"), "w") as f:
                json.dump(self.env.all_messages, f)
        
        return step_result
        
        
        

@dataclass(frozen=True, slots=True)
class LoggingGroupBuilder(EnvGroupBuilder):
    env_group_builder: EnvGroupBuilder
    logging_cfg: LoggingConfig

    async def make_envs(self) -> list[Env]:
        envs = await self.env_group_builder.make_envs()
        return [
            LoggingTranscriptsEnv(env, self.logging_cfg)
            for env in envs
        ]


class LoggingDataset(RLDataset):
    def __init__(
        self,
        dataset: RLDataset,
        logging_cfg: LoggingConfig,
    ) -> None:
        self.dataset = dataset
        self.logging_cfg = logging_cfg

    def get_batch(self, index: int) -> Sequence[LoggingGroupBuilder]:
        batch = self.dataset.get_batch(index)
        return [
            LoggingGroupBuilder(
                env_group_builder,
                self.logging_cfg,
            )
            for env_group_builder in batch
        ]

    def __len__(self) -> int:
        return len(self.dataset)


@dataclass(frozen=True, slots=True)
class LoggingDatasetBuilder(RLDatasetBuilder):
    dataset_builder: RLDatasetBuilder
    logging_cfg: LoggingConfig

    async def __call__(self) -> tuple[LoggingDataset, LoggingDataset]:
        train_data, test_data = await self.dataset_builder()

        return (
            LoggingDataset(train_data, self.logging_cfg),
            LoggingDataset(test_data, self.logging_cfg),
        )



def main(logging_cfg: LoggingConfig) -> None:
    import tinker_cookbook.rl.envs.bash_apps_env as inner_env
    
    if hasattr(inner_env, "build_docker_image"):
        inner_env.build_docker_image()
    else:
        pass
    
    if hasattr(inner_env, "build_config"):
        config = inner_env.build_config()
        cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
        
        # Replace the dataset builder with our own
        config  = chz.replace(config, dataset_builder=LoggingDatasetBuilder(
            dataset_builder=config.dataset_builder,
            logging_cfg=logging_cfg,
        ))
        
        if config.wandb_name is not None:
            config = chz.replace(config, wandb_name=config.wandb_name + "_monitored")
        
            
        if config.log_path is not None:
            config = chz.replace(config, log_path=config.log_path + "_monitored")
            
        asyncio.run(train.main(config))
    else:
        raise ValueError("No build_config function found")

if __name__ == "__main__":
    main(LoggingConfig(
        transcripts_dir='/tmp/tinker-transcripts/bash_apps_nothing'
    ))
