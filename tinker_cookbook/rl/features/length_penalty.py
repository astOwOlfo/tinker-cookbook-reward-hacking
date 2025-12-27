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
from tinker_cookbook import renderers


class LengthPenaltyConfig:
    def __init__(self, length_penalty: float):
        self.length_penalty = length_penalty

    def logging_tags(self) -> list[str]:
        return ["length_penalty"]


class LengthPenaltyEnv(Env):
    def __init__(self, env: Env, length_penalty_cfg: LengthPenaltyConfig):
        self.env = env
        self.length_penalty_config = length_penalty_cfg
        self.length = 0

        if not hasattr(self.env, "all_messages"):
            raise ValueError(
                "Environment must have an all_messages attribute to keep track of the conversation1"
            )

    @property
    def all_messages(self) -> list[renderers.Message]:
        return self.env.all_messages

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return await self.env.initial_observation()

    async def step(self, action: Action) -> StepResult:
        step_result = await self.env.step(action)
        if not step_result.episode_done:
            self.length = step_result.next_observation.length
            return step_result

        # Add length penalty
        length = self.length + len(action)
        step_result.reward = step_result.reward - (
            length * self.length_penalty_config.length_penalty
        )
        step_result.metrics["length_penalty"] = length * self.length_penalty_config.length_penalty

        return step_result

    @property
    def renderer(self) -> renderers.Renderer:
        return self.env.renderer
    
    @property
    def stop_condition(self) -> StopCondition:
        return self.env.stop_condition


LengthPenalty = Feature(LengthPenaltyEnv)


def main(length_penalty_cfg: LengthPenaltyConfig) -> None:
    import tinker_cookbook.rl.envs.bash_apps_env as inner_env

    if hasattr(inner_env, "build_docker_image"):
        inner_env.build_docker_image()
    else:
        pass

    if hasattr(inner_env, "build_config"):
        config = inner_env.build_config()
        new_config = LengthPenalty(env_cfg=config, feature_cfg=length_penalty_cfg)

        asyncio.run(train.main(new_config))
    else:
        raise ValueError("No build_config function found")


if __name__ == "__main__":
    load_dotenv()
    main(LengthPenaltyConfig(length_penalty=1e-4))
