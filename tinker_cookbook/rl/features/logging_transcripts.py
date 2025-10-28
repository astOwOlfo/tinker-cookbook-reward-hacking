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
from tinker_cookbook.rl.features.feature import Feature

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
        
LoggingTranscripts = Feature(LoggingTranscriptsEnv)
        
def main(logging_cfg: LoggingConfig) -> None:
    import tinker_cookbook.rl.envs.bash_apps_env as inner_env
    
    if hasattr(inner_env, "build_docker_image"):
        inner_env.build_docker_image()
    else:
        pass

    if hasattr(inner_env, "build_config"):
        config = inner_env.build_config()
        new_config = LoggingTranscripts(
            env_cfg=config,
            feature_cfg=logging_cfg
        )
        
        asyncio.run(train.main(new_config))
    else:
        raise ValueError("No build_config function found")
        

if __name__ == "__main__":
    main(LoggingConfig(
        transcripts_dir='/tmp/tinker-transcripts/bash_apps_nothing'
    ))
