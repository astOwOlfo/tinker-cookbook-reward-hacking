from itertools import chain
from transformers import PreTrainedTokenizer
from random import Random
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
from dataclasses import dataclass, field, replace
from collections.abc import Awaitable
from typing import Any, Callable, Sequence, Literal
from dotenv import load_dotenv

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
from tinker.types import ModelInput
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

import inspect_ai
import inspect_ai.dataset
import inspect_ai.log
import inspect_ai.model
import inspect_ai.tool
import inspect_ai.solver
import inspect_ai.scorer

logger = logging.getLogger(__name__)


def inspect_messages_to_tinker_messages(
    messages: list[inspect_ai.model.ChatMessage],
) -> list[renderers.Message]:
    assert all(isinstance(message.content, str) for message in messages)
    return [
        renderers.Message(
            role=message.role,
            content=message.content.strip(),  # type: ignore
        )
        for message in messages
    ]


@dataclass(frozen=True, slots=True)
class InspectTask:
    samples: list[inspect_ai.dataset.Sample]
    task_kwargs: dict[str, Any]
    get_reward: Callable[[inspect_ai.log.EvalLog], float]
    get_metrics: Callable[[inspect_ai.log.EvalLog], dict[str, float]]


lock = asyncio.Lock()


@dataclass(slots=True)
class InspectEnv(Env):
    model_name: str
    renderer: renderers.Renderer
    inspect_task: inspect_ai.Task
    inspect_model: inspect_ai.model.Model
    run_eval: Awaitable[None] | None

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        async def start_run_eval() -> None:
            async with lock:
                if self.run_eval is not None:
                    await self.run_eval
                    self.run_eval = None

        asyncio.create_task(start_run_eval())

        return await self.inspect_model.api.initial_observation()  # type: ignore

    async def step(self, action: Action) -> StepResult:
        return await self.inspect_model.api.step(action)  # type: ignore

    @property
    def stop_condition(self):
        return self.inspect_model.api.stop_condition  # type: ignore


@dataclass(frozen=True, slots=True)
class InspectGroupBuilder(EnvGroupBuilder):
    envs: list[InspectEnv]

    async def make_envs(self) -> list[InspectEnv]:
        return self.envs


@dataclass(slots=True)
class InspectDataset(RLDataset):
    inspect_task: InspectTask
    model_name: str
    batch_size: int
    group_size: int
    renderer: renderers.Renderer
    tokenizer: PreTrainedTokenizer
    shuffled_indices: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.shuffled_indices = list(range(len(self.inspect_task.samples)))
        Random(42).shuffle(self.shuffled_indices)

    def get_batch(self, index: int) -> Sequence[InspectGroupBuilder]:
        samples = self.inspect_task.samples
        shuffled_samples = [samples[i] for i in self.shuffled_indices]
        batch_samples = [
            shuffled_samples[i % len(self)]
            for i in range(self.batch_size * index, self.batch_size * (index + 1))
        ]

        print("DUPA: defining run_eval")

        async def run_eval_fn() -> None:
            print("DUPA: run_eval")
            flat_envs: list[InspectEnv] = [env for group in grouped_envs for env in group]
            eval_logs: list[inspect_ai.log.EvalLog] = []
            for env in flat_envs:
                new_logs = await inspect_ai.eval_async(
                    tasks=[env.inspect_task], model=env.inspect_model
                )
                assert len(new_logs) == 1
                eval_logs.append(new_logs[0])
            for env, eval_log in zip(flat_envs, eval_logs, strict=True):
                reward = self.inspect_task.get_reward(eval_log)
                metrics = self.inspect_task.get_metrics(eval_log)
                final_step_result = StepResult(
                    reward=reward,
                    episode_done=True,
                    next_observation=ModelInput.empty(),
                    next_stop_condition=env.inspect_model.api.stop_condition,  # type: ignore
                    metrics=metrics,
                )
                env.inspect_model.api.prompt_or_final_step_reuslt_queue.put(final_step_result)  # type: ignore

        run_eval = run_eval_fn()

        grouped_envs: list[list[InspectEnv]] = [
            [
                InspectEnv(
                    model_name=self.model_name,
                    renderer=self.renderer,
                    inspect_task=inspect_ai.Task(
                        dataset=inspect_ai.dataset.MemoryDataset(samples=[sample]),
                        **self.inspect_task.task_kwargs,
                    ),
                    inspect_model=inspect_ai.model.Model(
                        api=InspectAPIFromTinker(
                            model_name=self.model_name, renderer=self.renderer
                        ),
                        config=inspect_ai.model.GenerateConfig(),
                    ),
                    run_eval=run_eval,
                )
                for _ in range(self.group_size)
            ]
            for sample in batch_samples
        ]

        return [InspectGroupBuilder(envs=envs) for envs in grouped_envs]

    def __len__(self) -> int:
        return len(self.inspect_task.samples)


@dataclass(frozen=True, slots=True)
class InspectDatasetBuilder(RLDatasetBuilder):
    inspect_task: InspectTask
    model_name: str
    renderer_name: str
    batch_size: int
    group_size: int
    test_fraction: float

    async def __call__(self) -> tuple[InspectDataset, InspectDataset]:
        samples = self.inspect_task.samples.copy()
        Random(42).shuffle(samples)
        n_train = int((1 - self.test_fraction) * len(samples))
        train_samples = samples[:n_train]
        test_samples = samples[n_train:]
        assert len(train_samples) > 0
        assert len(test_samples) > 0
        train_inspect_task = replace(self.inspect_task, samples=train_samples)
        test_inspect_task = replace(self.inspect_task, samples=test_samples)

        tokenizer = get_tokenizer(self.model_name)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        return tuple(  # type: ignore
            InspectDataset(
                inspect_task=inspect_task,
                model_name=self.model_name,
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                tokenizer=tokenizer,
            )
            for inspect_task in [train_inspect_task, test_inspect_task]
        )


def build_config() -> train.Config:
    # model_name = "Qwen/Qwen3-8B"
    model_name = "meta-llama/Llama-3.2-1B"

    inspect_task = InspectTask(
        samples=list(
            inspect_ai.dataset.hf_dataset(
                "cais/mmlu",
                name="all",
                split="test",
                sample_fields=inspect_ai.dataset.FieldSpec(input="question", target="answer"),
            )
        ),
        task_kwargs={
            "plan": [
                inspect_ai.solver.system_message("You are a helpful assistant."),
                inspect_ai.solver.generate(),
            ],
            "scorer": inspect_ai.scorer.choice(),
        },
        get_reward=lambda eval_log: eval_log.results.scores[0].metrics["accuracy"].value,
        get_metrics=lambda eval_log: {},
    )

    dataset_builder = InspectDatasetBuilder(
        inspect_task=inspect_task,
        model_name=model_name,
        renderer_name=model_info.get_recommended_renderer_name(model_name),
        batch_size=4,
        group_size=2,
        test_fraction=0.1,
    )

    return train.Config(
        model_name=model_name,
        log_path="/tmp/tinker-examples/inspect",
        dataset_builder=dataset_builder,
        learning_rate=4e-5,
        max_tokens=1024,
        eval_every=0,
    )


def main() -> None:
    config = build_config()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
