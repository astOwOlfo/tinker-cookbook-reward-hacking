from copy import deepcopy
from transformers import PreTrainedTokenizer
from uuid import uuid4
from random import Random
from dotenv import load_dotenv
import logging
import asyncio
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
            print("starting lazy")
            asyncio.create_task(self.coroutine)


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
        # print("next_prompt_or_final_step_result", sample_id)
        return await self.prompt_or_final_step_result_queues[sample_id].get()

    async def register_completion(self, sample_id: SampleId, completion: list[int]) -> None:
        # print("register_completion", sample_id)
        await self.completion_queues[sample_id].put(completion)

    async def register_final_step_result(
        self, sample_id: SampleId, step_result: StepResult
    ) -> None:
        # print("register_final_step_result", sample_id)
        assert step_result.episode_done
        await self.prompt_or_final_step_result_queues[sample_id].put(step_result)

    async def sample_completion(self, prompt: list[int], sample_id: SampleId) -> list[int]:
        # print("sample_completion", sample_id)
        await self.prompt_or_final_step_result_queues[sample_id].put(prompt)
        return await self.completion_queues[sample_id].get()

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # print(f"{tools=}")

        messages_with_sample_id = [
            message
            for message in input
            if message.metadata is not None and "sample_id" in message.metadata.keys()
        ]
        if len(messages_with_sample_id) == 0:
            print("Could not find any message with sample_id in metadata.")
            assert False
        sample_id = messages_with_sample_id[0].metadata["sample_id"]
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

        # print("generate")
        if system_message is not None:
            input = [ChatMessageSystem(content=system_message)] + input
        conversation = inspect_messages_to_tinker_messages(input)
        prompt = self.renderer.build_generation_prompt(conversation)

        sampled_tokens = await self.sample_completion(prompt=prompt.to_ints(), sample_id=sample_id)

        message, parse_success = self.renderer.parse_response(sampled_tokens)
        # TODO: if not parse_success:
        import json

        print("=" * 256)
        print(json.dumps([message.model_dump(exclude_none=True) for message in input], indent=4))
        print(f"{parse_success=}", json.dumps(message, indent=4))

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


@dataclass(frozen=True, slots=True)
class InspectEnv(Env):
    model_name: str
    renderer: renderers.Renderer
    inspect_llm_wrapper: InspectAPIFromTinker
    sample_id: SampleId
    start_eval: Lazy
    max_prompt_tokens: int

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        print("initial observation", self.sample_id)
        await self.start_eval.start()
        print("started eval", self.sample_id)

        observation = await self.inspect_llm_wrapper.next_prompt_or_final_step_result(
            sample_id=self.sample_id
        )

        # assert not isinstance(observation, StepResult), (
        #     "Inspect environment finished without generating any completions."
        # )
        if isinstance(observation, StepResult):
            print("WARNING: EVAL FINISHED WITH NO CALLS TO THE LLM")
            print(f"{observation=}")
            dummy_observation = self.renderer.build_generation_prompt(
                [{"role": "user", "content": "Plaese ignore this message."}]
            )
            return dummy_observation, self.stop_condition

        return ModelInput.from_ints(observation), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        # print("step", self.sample_id)
        await self.inspect_llm_wrapper.register_completion(
            sample_id=self.sample_id, completion=action
        )

        observation_or_final_step_result = (
            await self.inspect_llm_wrapper.next_prompt_or_final_step_result(
                sample_id=self.sample_id
            )
        )

        if isinstance(observation_or_final_step_result, StepResult):
            # print("final step result reached", self.sample_id)
            assert observation_or_final_step_result.episode_done
            return observation_or_final_step_result

        if len(observation_or_final_step_result) > self.max_prompt_tokens:
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=ModelInput.empty(),
                next_stop_condition=self.stop_condition,
            )

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
class InspectEnvGroupBuilder(EnvGroupBuilder):
    envs: list[InspectEnv]

    async def make_envs(self) -> list[InspectEnv]:
        return self.envs


@dataclass(slots=True)
class InspectRLDataset(RLDataset):
    inspect_task: Task
    model_name: str
    batch_size: int
    group_size: int
    renderer: renderers.Renderer
    tokenizer: PreTrainedTokenizer
    max_prompt_tokens: int
    get_rewards: Callable[[EvalLog, list[Sample]], list[float]]
    get_metrics: Callable[[EvalLog, list[Sample]], list[dict[str, float]]]
    shuffled_inspect_task_samples: list[Sample] = field(init=False)

    def __post_init__(self) -> None:
        self.shuffled_inspect_task_samples = list(self.inspect_task.dataset)
        Random(42).shuffle(self.shuffled_inspect_task_samples)

    def batch_samples(self, index: int) -> list[Sample]:
        return [
            self.shuffled_inspect_task_samples[i % len(self.shuffled_inspect_task_samples)]
            for i in range(index * self.batch_size, (index + 1) * self.batch_size)
        ]

    def get_batch(self, index: int) -> list[InspectEnvGroupBuilder]:
        samples: list[Sample] = self.batch_samples(index)

        repeated_samples: list[Sample] = []
        for sample in samples:
            for _ in range(self.group_size):
                repeated_samples.append(deepcopy(sample))

        sample_ids: list[SampleId] = list(range(len(repeated_samples)))
        for sample_id, sample in zip(sample_ids, repeated_samples, strict=True):
            sample.id = sample_id

        inspect_llm_wrapper = InspectAPIFromTinker(
            model_name=self.model_name,
            renderer=self.renderer,
            sample_ids=sample_ids,
            tokenizer=self.tokenizer,
        )

        async def run_eval() -> None:
            subtask: Task = deepcopy(self.inspect_task)
            subtask.dataset = MemoryDataset(repeated_samples)

            print("starting eval")
            eval_logs: list[EvalLog] = await eval_async(
                subtask,
                model=Model(api=inspect_llm_wrapper, config=GenerateConfig()),
                solver=sample_id_in_message_metadata_solver_wrapper(subtask.solver),
                max_retries=0,  # don't retry llm completions
                max_connections=999999,  # don't limit how many llm completions can run at a time
                max_sandboxes=999999,
                # max_subprocesses=999999,
            )

            assert len(eval_logs) == 1
            eval_log = eval_logs[0]

            rewards: list[float] = self.get_rewards(eval_log, repeated_samples)
            metrics: list[dict[str, float]] = self.get_metrics(eval_log, repeated_samples)
            for sample_id, reward, metric in zip(sample_ids, rewards, metrics, strict=True):
                final_step_result = StepResult(
                    reward=reward,
                    episode_done=True,
                    next_observation=ModelInput.empty(),
                    next_stop_condition=self.renderer.get_stop_sequences(),
                    metrics=metric,
                )
                await inspect_llm_wrapper.register_final_step_result(
                    sample_id=sample_id, step_result=final_step_result
                )

        start_eval = Lazy(run_eval())

        envs: list[InspectEnv] = [
            InspectEnv(
                model_name=self.model_name,
                renderer=self.renderer,
                inspect_llm_wrapper=inspect_llm_wrapper,
                sample_id=sample_id,
                start_eval=start_eval,
                max_prompt_tokens=self.max_prompt_tokens,
            )
            for sample_id in sample_ids
        ]

        grouped_envs: list[list[InspectEnv]] = [
            envs[i * self.group_size : (i + 1) * self.group_size] for i in range(self.batch_size)
        ]

        return [InspectEnvGroupBuilder(envs=group) for group in grouped_envs]

    def __len__(self) -> int:
        return len(self.shuffled_inspect_task_samples)


@dataclass(frozen=True, slots=True)
class InspectRLDatasetBuilder(RLDatasetBuilder):
    model_name: str
    batch_size: int
    group_size: int
    renderer_name: str
    max_prompt_tokens: int
    inspect_task: Task
    get_rewards: Callable[[EvalLog, list[Sample]], list[float]]
    get_metrics: Callable[[EvalLog, list[Sample]], list[dict[str, float]]]
    test_fraction: float

    async def __call__(self) -> tuple[InspectRLDataset, InspectRLDataset]:
        samples: list[Sample] = list(self.inspect_task.dataset)
        Random(42).shuffle(samples)
        n_train = int((1 - self.test_fraction) * len(samples))
        train_samples = samples[:n_train]
        test_samples = samples[n_train:]
        assert len(train_samples) > 0
        assert len(test_samples) > 0

        train_task: Task = deepcopy(self.inspect_task)
        train_task.dataset = MemoryDataset(train_samples)
        test_task: Task = deepcopy(self.inspect_task)
        test_task.dataset = MemoryDataset(test_samples)

        tokenizer = get_tokenizer(self.model_name)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        return tuple(  # type: ignore
            InspectRLDataset(
                inspect_task=task,
                model_name=self.model_name,
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                tokenizer=tokenizer,
                max_prompt_tokens=self.max_prompt_tokens,
                get_rewards=self.get_rewards,
                get_metrics=self.get_metrics,
            )
            for task in [train_task, test_task]
        )


def build_config_mmlu() -> train.Config:
    from inspect_evals.mmlu import mmlu_0_shot

    model_name = "meta-llama/Llama-3.2-1B"
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

    dataset_builder = InspectRLDatasetBuilder(
        model_name=model_name,
        batch_size=256,
        group_size=8,
        renderer_name=model_info.get_recommended_renderer_name(model_name),
        max_prompt_tokens=context_length - max_completion_tokens - 16,  # -16 just in case
        inspect_task=inspect_task,
        get_rewards=get_rewards,
        get_metrics=get_metrics,
        test_fraction=0.1,
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


def build_config_impossible_bench() -> train.Config:
    from impossiblebench import impossible_livecodebench, impossible_swebench

    def get_rewards(eval_log: EvalLog, samples: list[Sample]) -> list[float]:
        return [
            (
                {"I": 0.0, "C": 1.0}[sample.scores["agentic_humaneval_scorer"].value]
                if "agentic_humaneval_scorer" in sample.scores.keys()
                else 0.0
            )
            for sample in eval_log.samples
        ]

    def get_metrics(eval_log: EvalLog, samples: list[Sample]) -> list[dict[str, float]]:
        return [
            {"score_missing": 0.0 if "agentic_humaneval_scorer" in sample.scores.keys() else 1.0}
            for sample in eval_log.samples
        ]

    """
    inspect_task: Task = impossible_swebench(
        split="original",
        agent_type="minimal",
        human_intervention=False,
        allow_internet=False,
        reset_tests=False,
        hide_tests=False,
        # limit=10,  # TODO: remove
    )
    """

    inspect_task: Task = impossible_livecodebench(
        split="original",
        agent_type="minimal",
        allow_test_modifications=True,
    )

    # model_name = "Qwen/Qwen3-30B-A3B"
    # model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    # model_name = "Qwen/Qwen3-32B"
    # renderer_name = "qwen3_disable_thinking"
    # model_name = "deepseek-ai/DeepSeek-V3.1"
    # renderer_name = "deepseekv3_disable_thinking"
    model_name = "openai/gpt-oss-120b"
    renderer_name = "gpt_oss_medium_reasoning"
    context_length = 32768
    # max_completion_tokens = 2048
    max_completion_tokens = 8192

    dataset_builder = InspectRLDatasetBuilder(
        model_name=model_name,
        batch_size=64,
        group_size=8,
        renderer_name=renderer_name,
        max_prompt_tokens=context_length - max_completion_tokens,
        inspect_task=inspect_task,
        get_rewards=get_rewards,
        get_metrics=get_metrics,
        test_fraction=0.1,
    )

    # TODO: test the following hypothesis on why this config doesn't crash but build_config_impossible_bench_old crashes
    # because the old function does evals (eval_every != 0) and it runs evals in parallel with training so this is why we get can't run more than one inspect_ai.eval in parallel

    return train.Config(
        model_name=model_name,
        log_path="/tmp/tinker-examples/inspect-impossible-livecodebench",
        dataset_builder=dataset_builder,
        learning_rate=1e-4,
        max_tokens=max_completion_tokens,
        eval_every=0,
        wandb_project="inspect-impossible-bench",

        wandb_name=model_name,
    )


def build_config_berkeley_function_calling_leaderboard() -> train.Config:
    from inspect_evals.bfcl import bfcl

    def get_rewards(eval_log: EvalLog, samples: list[Sample]) -> list[float]:
        for sample in eval_log.samples:
            print("SCORES:", sample.scores)

        return [
            (
                {"I": 0.0, "C": 1.0}[sample.scores["agentic_humaneval_scorer"].value]
                if "agentic_humaneval_scorer" in sample.scores.keys()
                else 0.0
            )
            for sample in eval_log.samples
        ]

    def get_metrics(eval_log: EvalLog, samples: list[Sample]) -> list[dict[str, float]]:
        return [
            {"score_missing": 0.0 if "agentic_humaneval_scorer" in sample.scores.keys() else 1.0}
            for sample in eval_log.samples
        ]

    inspect_task: Task = bfcl()

    # model_name = "Qwen/Qwen3-30B-A3B"
    # renderer_name = "qwen3_disable_thinking"
    model_name = "openai/gpt-oss-120b"
    renderer_name = "gpt_oss_medium_reasoning"
    context_length = 32768
    max_completion_tokens = 8192

    dataset_builder = InspectRLDatasetBuilder(
        model_name=model_name,
        batch_size=64,
        group_size=2,
        renderer_name=renderer_name,
        max_prompt_tokens=context_length - max_completion_tokens,
        inspect_task=inspect_task,
        get_rewards=get_rewards,
        get_metrics=get_metrics,
        test_fraction=0.1,
    )

    return train.Config(
        model_name=model_name,
        log_path="/tmp/tinker-examples/inspect-assistant-bench",
        dataset_builder=dataset_builder,
        learning_rate=1e-4,
        max_tokens=max_completion_tokens,
        eval_every=0,
        # wandb_project="inspect-assistant-bench",
        # wandb_name=model_name,
    )


def main() -> None:
    load_dotenv()
    # config = build_config_mmlu()
    config = build_config_impossible_bench()
    # config = build_config_berkeley_function_calling_leaderboard()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
