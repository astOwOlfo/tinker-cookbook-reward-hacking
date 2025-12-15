from copy import deepcopy
from os import makedirs
from transformers import PreTrainedTokenizer
from uuid import uuid4
import json
from os.path import join
from random import Random
from dotenv import load_dotenv
import logging
import asyncio
from itertools import pairwise, chain
from dataclasses import dataclass, field
from collections.abc import Awaitable
from typing import Callable

from inspect_ai import Task, eval_async
from inspect_ai.model import (
    modelapi,
    ModelAPI,
    Model,
    ChatMessage,
    GenerateConfig,
    ModelOutput,
    ChatMessageSystem,
    ChatMessageAssistant,
    ChatCompletionChoice,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.solver import solver, Solver, TaskState, Generate
from inspect_ai.log import EvalLog
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.tool import ToolCall

from tinker.types import ModelInput
from tinker_cookbook import renderers, model_info, cli_utils
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Observation,
    StepResult,
    RLDataset,
    RLDatasetBuilder,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.rl import train
from tinker_cookbook.rl.envs.tools import get_system_message_with_tools


logger = logging.getLogger(__name__)


class Lazy:
    def __init__(self, coroutine: Awaitable) -> None:
        self.coroutine = coroutine
        self.started = False
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        async with self.lock:
            if self.started:
                return
            self.started = True
            asyncio.create_task(self.coroutine)  # type: ignore


def inspect_messages_to_tinker_messages(messages: list[ChatMessage]) -> list[renderers.Message]:
    assert all(isinstance(message.content, str) for message in messages)
    return [
        renderers.Message(
            role=message.role,
            content=message.content.strip(),  # type: ignore
        )
        for message in messages
    ]


def tinker_assisstant_message_to_inspect_assistant_message(
    message: renderers.Message,
) -> ChatMessageAssistant:
    assert message["role"] == "assistant"

    tinker_tool_calls: list[renderers.ToolCall] | None = message.get("tool_calls")
    inspect_tool_calls: list[ToolCall] | None
    if tinker_tool_calls is None:
        inspect_tool_calls = None
    else:
        inspect_tool_calls = []
        for tinker_tool_call in tinker_tool_calls:
            inspect_tool_calls.append(
                ToolCall(
                    id=str(uuid4()),
                    function=tinker_tool_call["name"],
                    arguments=tinker_tool_call["args"],
                )
            )

    return ChatMessageAssistant(
        content=message["content"],
        tool_calls=inspect_tool_calls,  # type: ignore
    )


def inspect_tools_to_dict(tools: list[ToolInfo]) -> list[dict]:
    return [{"type": "function", "function": tool.model_dump(exclude_none=True)} for tool in tools]


n_pending_completions: int = 0
n_pending_steps: int = 0


SampleId = int


@modelapi(name="tinker-sampling")
class InspectAPIFromTinker(ModelAPI):
    def __init__(
        self,
        model_name: str,
        renderer: renderers.Renderer,
        sample_ids: list[SampleId],
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super().__init__(model_name=model_name)

        self.model_name = model_name
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.lock = asyncio.Lock()

        self.completion_queues: dict[SampleId, asyncio.Queue[list[int]]] = {
            sample_id: asyncio.Queue(maxsize=1) for sample_id in sample_ids
        }
        self.prompt_or_final_step_result_queues: dict[
            SampleId, asyncio.Queue[list[int] | StepResult]
        ] = {sample_id: asyncio.Queue(maxsize=1) for sample_id in sample_ids}

    async def next_prompt_or_final_step_result(self, sample_id: SampleId) -> list[int] | StepResult:
        return await self.prompt_or_final_step_result_queues[sample_id].get()

    async def register_completion(self, sample_id: SampleId, completion: list[int]) -> None:
        await self.completion_queues[sample_id].put(completion)

    async def register_final_step_result(
        self, sample_id: SampleId, step_result: StepResult
    ) -> None:
        assert step_result.episode_done
        await self.prompt_or_final_step_result_queues[sample_id].put(step_result)

    async def sample_completion(self, prompt: list[int], sample_id: SampleId) -> list[int]:
        await self.prompt_or_final_step_result_queues[sample_id].put(prompt)
        global n_pending_completions
        n_pending_completions += 1
        completion = await self.completion_queues[sample_id].get()
        n_pending_completions -= 1
        return completion

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        messages_with_sample_id = [
            message
            for message in input
            if message.metadata is not None and "sample_id" in message.metadata.keys()
        ]
        if len(messages_with_sample_id) == 0:
            assert False
        sample_id = messages_with_sample_id[0].metadata["sample_id"]  # type: ignore
        assert isinstance(sample_id, SampleId)

        system_message = config.system_message
        if len(tools) > 0:
            if system_message is None:
                system_message = ""
            system_message = get_system_message_with_tools(
                tokenizer=self.tokenizer,
                system_message=system_message,
                tools=inspect_tools_to_dict(tools),
            )

        if system_message is not None:
            input = [ChatMessageSystem(content=system_message)] + input
        conversation = inspect_messages_to_tinker_messages(input)
        prompt = self.renderer.build_generation_prompt(conversation)

        sampled_tokens = await self.sample_completion(prompt=prompt.to_ints(), sample_id=sample_id)

        message, parse_success = self.renderer.parse_response(sampled_tokens)
        # TODO: if not parse_success:

        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=tinker_assisstant_message_to_inspect_assistant_message(message)
                )
            ],
        )


@solver
def sample_id_in_message_metadata_solver_wrapper(wrapped_solver: Solver) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        for message in state.messages:
            if message.metadata is None:
                message.metadata = {}
            message.metadata["sample_id"] = state.sample_id
        return await wrapped_solver(state, generate)

    return solve


@dataclass(slots=True)
class InspectEnvMultiple(Env):
    model_name: str
    renderer: renderers.Renderer
    inspect_llm_wrapper: InspectAPIFromTinker
    sample_id: SampleId
    start_eval: Lazy
    max_prompt_tokens: int
    was_truncated: bool = False
    all_messages: list[dict] = field(default_factory=lambda: [])

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        await self.start_eval.start()

        observation = await self.inspect_llm_wrapper.next_prompt_or_final_step_result(  # type: ignore
            sample_id=self.sample_id
        )

        # assert not isinstance(observation, StepResult), (
        #     "Inspect environment finished without generating any completions."
        # )
        if isinstance(observation, StepResult):
            dummy_observation = self.renderer.build_generation_prompt(
                [{"role": "user", "content": "Plaese ignore this message."}]
            )
            return dummy_observation, self.stop_condition

        return ModelInput.from_ints(observation), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        await self.inspect_llm_wrapper.register_completion(  # type: ignore
            sample_id=self.sample_id, completion=action
        )

        global n_pending_steps
        n_pending_steps += 1

        observation_or_final_step_result = (
            await self.inspect_llm_wrapper.next_prompt_or_final_step_result(  # type: ignore
                sample_id=self.sample_id
            )
        )

        n_pending_steps -= 1

        if isinstance(observation_or_final_step_result, StepResult):
            assert observation_or_final_step_result.episode_done
            return observation_or_final_step_result

        if len(observation_or_final_step_result) > self.max_prompt_tokens:
            observation_or_final_step_result = observation_or_final_step_result[
                -self.max_prompt_tokens :
            ]
            self.was_truncated = True

        return StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=ModelInput.from_ints(observation_or_final_step_result),
            next_stop_condition=self.stop_condition,
        )

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()


@dataclass(frozen=True, slots=True)
class InspectEnvMultipleGroupBuilder(EnvGroupBuilder):
    envs: list[InspectEnvMultiple]

    async def make_envs(self) -> list[InspectEnvMultiple]:
        return self.envs


@dataclass(slots=True)
class InspectMultipleRLDataset(RLDataset):
    inspect_tasks: dict[str, Task]
    model_name: str
    batch_size: int
    group_size: int
    renderer: renderers.Renderer
    tokenizer: PreTrainedTokenizer
    max_prompt_tokens: int
    save_rollouts_directory: str | None
    get_rewards: dict[str, Callable[[EvalLog, list[Sample]], list[float]]]
    get_metrics: dict[str, Callable[[EvalLog, list[Sample]], list[dict[str, float]]]]
    shuffled_inspect_task_samples: dict[str, list[Sample]] = field(init=False)

    def __post_init__(self) -> None:
        self.shuffled_inspect_task_samples = {
            task_name: list(task.dataset) for task_name, task in self.inspect_tasks.items()
        }
        for samples in self.shuffled_inspect_task_samples.values():
            Random(42).shuffle(samples)

    def n_samples_per_task_per_batch(self) -> dict[str, int]:
        n_tasks = len(self.inspect_tasks)
        pivots = [i * self.batch_size // n_tasks for i in range(n_tasks + 1)]
        assert pivots[0] == 0
        assert pivots[-1] == self.batch_size
        assert all(p1 < p2 for p1, p2 in pairwise(pivots))

        n_samples_per_task = [p2 - p1 for p1, p2 in pairwise(pivots)]
        assert sum(n_samples_per_task) == self.batch_size
        return {
            task_name: n_samples
            for task_name, n_samples in zip(
                self.inspect_tasks.keys(), n_samples_per_task, strict=True
            )
        }

    def batch_samples(self, index: int) -> dict[str, list[Sample]]:
        n_samples_per_task: dict[str, int] = self.n_samples_per_task_per_batch()

        return {
            task_name: [
                self.shuffled_inspect_task_samples[task_name][
                    i % len(self.shuffled_inspect_task_samples[task_name])
                ]
                for i in range(
                    index * n_samples_per_task[task_name],
                    (index + 1) * n_samples_per_task[task_name],
                )
            ]
            for task_name in self.inspect_tasks.keys()
        }

    def get_batch(self, index: int) -> list[InspectEnvMultipleGroupBuilder]:
        samples: dict[str, list[Sample]] = self.batch_samples(index)

        repeated_samples: dict[str, list[Sample]] = {
            task_name: [] for task_name in self.inspect_tasks.keys()
        }
        for task_name in self.inspect_tasks.keys():
            for sample in samples[task_name]:
                for _ in range(self.group_size):
                    repeated_samples[task_name].append(deepcopy(sample))

        sample_ids: list[SampleId] = list(range(self.batch_size * self.group_size))
        for sample_id, sample in zip(
            sample_ids,
            [sample for samples in repeated_samples.values() for sample in samples],
            strict=True,
        ):
            sample.id = sample_id

        inspect_llm_wrapper = InspectAPIFromTinker(
            model_name=self.model_name,
            renderer=self.renderer,  # type: ignore
            sample_ids=sample_ids,  # type: ignore
            tokenizer=self.tokenizer,  # type: ignore
        )

        async def run_eval() -> None:
            subtasks: dict[str, Task] = {}
            for task_name, task in self.inspect_tasks.items():
                subtasks[task_name] = deepcopy(task)
                subtasks[task_name].dataset = MemoryDataset(repeated_samples[task_name])
                subtasks[task_name].solver = sample_id_in_message_metadata_solver_wrapper(
                    subtasks[task_name].solver
                )

            eval_logs: list[EvalLog] = await eval_async(
                list(subtasks.values()),
                model=Model(api=inspect_llm_wrapper, config=GenerateConfig()),
                max_retries=0,  # don't retry llm completions
                max_connections=999999,  # don't limit how many llm completions can run at a time
                # max_sandboxes=8,
                max_sandboxes=999999,
                max_subprocesses=999999,
            )

            assert len(eval_logs) == len(self.inspect_tasks)

            rewards: dict[str, list[float]] = {
                task_name: self.get_rewards[task_name](eval_log, repeated_samples[task_name])
                for eval_log, task_name in zip(eval_logs, self.inspect_tasks.keys(), strict=True)
            }
            metrics: dict[str, list[dict[str, float]]] = {
                task_name: self.get_metrics[task_name](eval_log, repeated_samples[task_name])
                for eval_log, task_name in zip(eval_logs, self.inspect_tasks.keys(), strict=True)
            }

            flat_rewards = chain.from_iterable(rewards.values())
            flat_metrics = chain.from_iterable(metrics.values())
            for sample_id, reward, metric in zip(
                sample_ids, flat_rewards, flat_metrics, strict=True
            ):
                final_step_result = StepResult(
                    reward=reward,
                    episode_done=True,
                    next_observation=ModelInput.empty(),
                    next_stop_condition=self.renderer.get_stop_sequences(),
                    metrics=metric,
                )
                await inspect_llm_wrapper.register_final_step_result(  # type: ignore
                    sample_id=sample_id, step_result=final_step_result
                )

            self.save_rollouts(eval_logs=eval_logs, rewards=rewards, metrics=metrics, epoch=index)

        start_eval = Lazy(run_eval())

        envs: list[InspectEnvMultiple] = [
            InspectEnvMultiple(
                model_name=self.model_name,
                renderer=self.renderer,
                inspect_llm_wrapper=inspect_llm_wrapper,
                sample_id=sample_id,
                start_eval=start_eval,
                max_prompt_tokens=self.max_prompt_tokens,
            )
            for sample_id in sample_ids
        ]

        grouped_envs: list[list[InspectEnvMultiple]] = [
            envs[i * self.group_size : (i + 1) * self.group_size] for i in range(self.batch_size)
        ]

        return [InspectEnvMultipleGroupBuilder(envs=group) for group in grouped_envs]

    def __len__(self) -> int:
        return 9999

    def save_rollouts(
        self,
        eval_logs: list[EvalLog],
        rewards: dict[str, list[float]],
        metrics: dict[str, list[dict[str, float]]],
        epoch: int,
    ) -> None:
        if self.save_rollouts_directory is None:
            return

        makedirs(self.save_rollouts_directory, exist_ok=True)

        json_rollouts = {}

        for task_name, eval_log in zip(self.inspect_tasks.keys(), eval_logs, strict=True):
            json_messages = [
                [message.model_dump() for message in sample.messages]
                for sample in eval_log.samples  # type: ignore
            ]
            json_rollouts[task_name] = [
                {"messages": messages, "reward": reward, "metrics": metric}
                for messages, reward, metric in zip(json_messages, rewards["task_name"], metrics["task_name"], strict=True)
            ]

        filename = join(self.save_rollouts_directory, f"epoch-{epoch}-rollouts.json")
        with open(filename, "w") as f:
            json.dump(json_rollouts, f)
        print(f"Saved transcripts to '{filename}'.")


@dataclass(frozen=True, slots=True)
class InspectMultipleRLDatasetBuilder(RLDatasetBuilder):
    model_name: str
    batch_size: int
    group_size: int
    renderer_name: str
    max_prompt_tokens: int
    inspect_tasks: dict[str, Task]
    get_rewards: dict[str, Callable[[EvalLog, list[Sample]], list[float]]]
    get_metrics: dict[str, Callable[[EvalLog, list[Sample]], list[dict[str, float]]]]
    test_fraction: float
    save_rollouts_directory: str | None

    def __post_init__(self) -> None:
        assert (
            set(self.inspect_tasks.keys())
            == set(self.get_rewards.keys())
            == set(self.get_metrics.keys())
        )

    async def __call__(self) -> tuple[InspectMultipleRLDataset, InspectMultipleRLDataset]:
        train_tasks: dict[str, Task] = {}
        test_tasks: dict[str, Task] = {}
        for task_name, task in self.inspect_tasks.items():
            samples: list[Sample] = list(task.dataset)
            Random(42).shuffle(samples)
            n_train = int((1 - self.test_fraction) * len(samples))
            train_samples = samples[:n_train]
            test_samples = samples[n_train:]
            assert len(train_samples) > 0
            assert len(test_samples) > 0

            train_tasks[task_name] = deepcopy(task)
            train_tasks[task_name].dataset = MemoryDataset(train_samples)
            test_tasks[task_name] = deepcopy(task)
            test_tasks[task_name].dataset = MemoryDataset(test_samples)

        tokenizer = get_tokenizer(self.model_name)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        return tuple(  # type: ignore
            InspectMultipleRLDataset(
                inspect_tasks=tasks,
                model_name=self.model_name,
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                tokenizer=tokenizer,
                max_prompt_tokens=self.max_prompt_tokens,
                get_rewards=self.get_rewards,
                get_metrics=self.get_metrics,
                save_rollouts_directory=self.save_rollouts_directory,
            )
            for tasks in [train_tasks, test_tasks]
        )


def build_config_mmlu() -> train.Config:
    from inspect_evals.mmlu import mmlu_0_shot

    # model_name = "meta-llama/Llama-3.2-1B"
    model_name = "Qwen/Qwen3-8B-Base"
    context_length = 32768
    max_completion_tokens = 16

    inspect_task: Task = mmlu_0_shot()

    def get_rewards(eval_log: EvalLog, samples: list[Sample]) -> list[float]:
        print("SCORES:", [next(iter(sample.scores.values())).value for sample in eval_log.samples])
        scores = [
            {"C": 1.0, "I": 0.0}[next(iter(sample.scores.values())).value]
            for sample in eval_log.samples
        ]
        from statistics import mean

        print(f"{mean(scores)=}")
        return scores

    def get_metrics(eval_log: EvalLog, samples: list[Sample]) -> list[dict[str, float]]:
        return [{} for _ in samples]

    dataset_builder = InspectMultipleRLDatasetBuilder(
        model_name=model_name,
        batch_size=64,
        group_size=8,
        renderer_name=model_info.get_recommended_renderer_name(model_name),
        max_prompt_tokens=context_length - max_completion_tokens - 16,  # -16 just in case
        inspect_tasks={"mmlu": inspect_task},
        get_rewards={"mmlu": get_rewards},
        get_metrics={"mmlu": get_metrics},
        test_fraction=0.1,
        save_rollouts_directory=None,
    )

    return train.Config(
        model_name=model_name,
        log_path="/tmp/tinker-examples/inspect-2",
        dataset_builder=dataset_builder,
        learning_rate=4e-5,
        max_tokens=max_completion_tokens,
        eval_every=0,
        wandb_project="inspect-mmlu",
        wandb_name=model_name,
    )


def build_config_bash_codeforces() -> train.Config:
    from tinker_cookbook.rl.envs.inspect.bash_codeforces import bash_codeforces

    tasks = {
        "overwritable": bash_codeforces(
            min_tests=64,
            max_tests=64,
            min_test_output_length=16,
            # n_synthetic_impossible_tests=192,
            reset_public_test_file=True,
            impossible=True,
        ),
        # "hardcodable": bash_codeforces(
        #     min_tests=1,
        #     max_tests=1,
        #     min_test_output_length=16,
        #     reset_public_test_file=True,
        #     impossible=True,
        # ),
    }

    def get_rewards(eval_log: EvalLog, samples: list[Sample]) -> list[float]:
        return [sample.scores["bash_codeforces_scorer"].value for sample in eval_log.samples]  # type: ignore

    def get_metrics(eval_log: EvalLog, samples: list[Sample]) -> list[dict[str, float]]:
        return [
            {
                key: float(value)
                for key, value in sample.scores["bash_codeforces_scorer"].metadata.items()  # type: ignore
            }
            for sample in eval_log.samples  # type: ignore
        ]

    model_name = "openai/gpt-oss-20b"
    context_length = 32768
    max_completion_tokens = 2048

    dataset_builder = InspectMultipleRLDatasetBuilder(
        model_name=model_name,
        batch_size=4,
        group_size=2,
        renderer_name="gpt_oss_low_reasoning",
        max_prompt_tokens=context_length - max_completion_tokens - 16,  # - 16 just in case
        inspect_tasks=tasks,
        get_rewards={key: get_rewards for key in tasks.keys()},
        get_metrics={key: get_metrics for key in tasks.keys()},
        test_fraction=0.1,
        save_rollouts_directory=None,
    )

    return train.Config(
        model_name=model_name,
        log_path="/tmp/tinker-examples/inspect/",
        dataset_builder=dataset_builder,
        learning_rate=4e-5,
        max_tokens=max_completion_tokens,
        eval_every=0,
    )


def main() -> None:
    load_dotenv()
    config = build_config_bash_codeforces()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
