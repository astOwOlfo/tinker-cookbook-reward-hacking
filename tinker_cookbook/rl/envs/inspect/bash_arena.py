from typing import Callable
from control_arena.settings.bash_arena.bash_arena_setting import BashArenaSetting, main_task_success
from control_arena._control_solver import ToolsSupplier
from control_arena import EvalMode
from control_arena.eval import ControlEvalConfig, get_control_task, get_control_task_product
from control_arena._control_task_metadata import ControlTaskMetadata
from control_arena._control_solver import DEFAULT_SYSTEM_PROMPT
from control_arena import control_agent
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.tool import Tool
from inspect_ai.agent import Agent, AgentPrompt, AgentState, AgentSubmit, agent, react
from inspect_ai._util.registry import registry_info
from control_arena.tools.submit_tool import SubmitToolConfig
from inspect_ai import Task
from inspect_ai.model import get_model
from inspect_ai.solver import Generate, TaskState, solver, Solver
import asyncio
from typing import Any

from tinker_cookbook import cli_utils
from tinker_cookbook.rl import train
from tinker_cookbook.rl.envs.inspect_env import InspectRLDatasetBuilder
from tinker_cookbook import hyperparam_utils


@control_agent
def passthrough(model: str) -> Agent:
    """Minimal agent that just calls the model directly."""

    async def agent(state: AgentState, tools: list[Tool]) -> AgentState:
        state.output = await get_model(model).generate(state.messages, tools=tools)
        state.messages.append(state.output.message)
        return state

    return agent


def bash_arena_task() -> Task:
    task: Task = get_control_task(
        setting=BashArenaSetting(), micro_protocol=passthrough, mode=EvalMode.HONEST
    )
    task.scorer = [main_task_success()]
    return task


def bash_arena_get_rewards(eval_log: EvalLog, samples: list[Sample]) -> list[float]:
    print("EVAL_LOG:", eval_log)
    for sample in eval_log.samples:
        print("SCORES:", {key: value for key, value in sample.scores.items()})


def bash_arena_get_metrics(eval_log: EvalLog, samples: list[Sample]) -> list[dict[str, Any]]:
    return [{} for _ in samples]


def build_config_bash_bench_2() -> train.Config:
    model_name = "openai/gpt-oss-20b"
    renderer_name = "gpt_oss_low_reasoning"

    max_completion_tokens = 2048
    context_length = 32768

    task = bash_arena_task()

    dataset_builder = InspectRLDatasetBuilder(
        model_name=model_name,
        batch_size=2,
        group_size=2,
        renderer_name=renderer_name,
        max_prompt_tokens=context_length - max_completion_tokens - 16,  # - 16 just in case
        inspect_task=task,
        get_rewards=bash_arena_get_rewards,
        get_metrics=bash_arena_get_metrics,
        test_fraction=0.1,
        save_rollouts_directory=None,
        no_deepcopy=True,
    )

    return train.Config(
        model_name=model_name,
        log_path="tmp/tinker-runs/bash-arena",
        dataset_builder=dataset_builder,
        learning_rate=4e-5
        if model_name.startswith("openai/gpt-oss")
        else hyperparam_utils.get_lr(model_name),
        max_tokens=max_completion_tokens,
        eval_every=0,
    )


async def main() -> None:
    config = build_config_bash_bench_2()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    await train.main(config)


if __name__ == "__main__":
    asyncio.run(main())
