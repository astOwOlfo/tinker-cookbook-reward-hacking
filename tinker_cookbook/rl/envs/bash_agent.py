from dataclasses import dataclass, field
import asyncio
from scalable_docker.client import ScalableDockerClient, Container
from tinker_cookbook.rl.types import StepResult, Env, Action
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.envs import tools
from tinker_cookbook.rl.envs.tools import ToolCall, ErrorParsingToolCall, FinishToolCall
import tinker
import traceback
from sys import stderr
from typing import Callable
import logging
from scalable_docker.client import (
    ScalableDockerClient,
    Container,
    ProcessOutput,
    MultiCommandTimeout,
    TIMED_OUT_PROCESS_OUTPUT,
    upload_file_command,
    Image,
)

logger = logging.getLogger(__name__)


### DOCKER SERVER HELPERS

@dataclass(slots=True)
class ContainerStarter:
    dockerfile_contents: list[str]
    scalable_docker_client: ScalableDockerClient
    _create_containers_task: asyncio.Task | None = None
    _lock: asyncio.Lock = field(default_factory=lambda: asyncio.Lock())
    all_containers: list[Container] | None = None

    async def start_starting(self) -> None:
        async with self._lock:
            if self._create_containers_task is not None:
                return

            self._create_containers_task = asyncio.create_task(
                self.scalable_docker_client.start_containers(self.dockerfile_contents)
            )

    async def get_container(self, index: int) -> Container:
        assert 0 <= index < len(self.dockerfile_contents)

        async with self._lock:
            assert self._create_containers_task is not None
            
        if self.all_containers is None:
            all_containers: list[Container] = await self._create_containers_task
            self.all_containers = all_containers

        return self.all_containers[index]
    
async def initialize_container(env: Env) -> StepResult | None:
    assert env.container_starter is not None, "Container starter is not initialized"
    assert env.container_index is not None, "Container index is not initialized"
    
    if env.container is not None:
        return

    try:
        env.container = await env.container_starter.get_container(env.container_index)
    except Exception:
        print("DOCKER ERROR! HERE IS THE ERROR:", file=stderr)
        traceback.print_exc()
        env.docker_error = True
        return done_step_result(env, reward=0.0)
    
async def run_startup_commands(
    env: Env,
    startup_commands: list[str],
) -> StepResult | None:
    assert env.container is not None, "Container is not initialized"

    try:
        outputs = await env.scalable_docker_client.run_commands(
            container=env.container,
            commands=startup_commands,
            timeout=env.cfg.startup_command_timeout,
        )
    except Exception:
        print("DOCKER ERROR! HERE IS THE ERROR:", file=stderr)
        traceback.print_exc()
        env.docker_error = True
        return done_step_result(env, reward=0.0)

    env.failed_startup_commands_and_outputs = [
        (command, output)
        for command, output in zip(startup_commands, outputs, strict=True)
        if output.exit_code != 0
    ]

    failed = len(env.failed_startup_commands_and_outputs) > 0
    if failed:
        print(
            f"Some startup commands failed. Here are the commands that failed (only those that failed - not necessarily all the startup commands) and their outputs: {env.failed_startup_commands_and_outputs}",
            file=stderr,
        )
        env.failed_startup_commands = True
        return done_step_result(env, reward=0.0)
    
## Default Env Functions
    
async def default_agent_step(env: Env, action: Action, final_reward_fn: Callable[[], StepResult]) -> StepResult:
    assert env.renderer is not None, "Renderer is not initialized"
    assert env.all_messages is not None, "All messages are not initialized"
    assert env.cfg is not None, "Config is not initialized"
    assert env.stop_condition is not None, "Stop condition is not initialized"
    assert env.available_tools is not None, "Available tools are not initialized"
    assert env.scalable_docker_client is not None, "Scalable docker client is not initialized"
    assert env.container_starter is not None, "Container starter is not initialized"
    assert env.container_index is not None, "Container index is not initialized"
    if not hasattr(env, "truncated"):
        env.truncated = False
    if not hasattr(env, "n_errors_parsing_tool_calls"):
        env.n_errors_parsing_tool_calls = 0
    if not hasattr(env, "container"):
        env.container = None
    if not hasattr(env, "docker_error"):
        env.docker_error = False
    if not hasattr(env, "ran_startup_commands"):
        env.ran_startup_commands = False
    if not hasattr(env, "failed_startup_commands"):
        env.failed_startup_commands = False
    if not hasattr(env, "n_tool_timeouts"):
        env.n_tool_timeouts = 0
    if not hasattr(env, "startup_commands"):
        env.startup_commands = None
        env.ran_startup_commands = True

    message, parse_success = env.renderer.parse_response(action)
    #assert parse_success, f"Error parsing message: {message}"

    env.all_messages.append(message)

    assert message["role"] == "assistant"
    assert {"role", "content"} <= set(message.keys()) <= {"role", "content", "tool_calls"}

    tool_call: ToolCall | ErrorParsingToolCall = tools.extract_tool_call(
        message, available_tools=env.available_tools
    )

    last_step = len(env.all_messages) > env.cfg.max_steps

    if isinstance(tool_call, ErrorParsingToolCall):
        if last_step:
            result = await final_reward_fn()
            return result

        env.n_errors_parsing_tool_calls += 1

        return error_parsing_tool_call_step_result(env, tool_call)

    if env.container is None:
        try:
            env.container = await env.container_starter.get_container(env.container_index)
        except Exception:
            print("DOCKER ERROR! HERE IS THE ERROR:", file=stderr)
            traceback.print_exc()
            env.docker_error = True
            return done_step_result(env, reward=0.0)

    if not env.ran_startup_commands:
        result = await run_startup_commands(env, env.startup_commands())
        if isinstance(result, StepResult):
            return result
        env.ran_startup_commands = True

    if isinstance(tool_call, FinishToolCall):
        result = await final_reward_fn()
        return result

    try:
        tool_outputs: list[ProcessOutput] = await env.scalable_docker_client.run_commands(
            container=env.container,
            commands=[tool_call.to_bash_command()],
            timeout=env.cfg.tool_timeout,
        )
    except Exception:
        print("DOCKER ERROR! HERE IS THE ERROR:", file=stderr)
        traceback.print_exc()
        env.docker_error = True
        return done_step_result(env, reward=0.0)

    assert len(tool_outputs) == 1
    tool_output: ProcessOutput = tool_outputs[0]

    if tool_output == TIMED_OUT_PROCESS_OUTPUT:
        env.n_tool_timeouts += 1

    if last_step:
        result = await final_reward_fn()
        return result

    return next_step_result(env, tool_output)
    
### STEP RESULT HELPERS
    
def done_step_result(env: Env, reward: float) -> StepResult:
    return StepResult(
        reward=reward,
        episode_done=True,
        next_observation=tinker.ModelInput.empty(),
        next_stop_condition=env.stop_condition,
        metrics=env.metrics(),
    )
    
def new_user_message_step_result(env: Env, new_user_message: str) -> StepResult:
    assert env.renderer is not None, "Renderer is not initialized"
    assert env.all_messages is not None, "All messages are not initialized"
    assert env.cfg is not None, "Config is not initialized"
    assert env.stop_condition is not None, "Stop condition is not initialized"
    if not hasattr(env, "truncated"):
        env.truncated = False
    
    new_user_message += disable_thinking_prompt(env)

    env.all_messages.append({"role": "user", "content": new_user_message})

    next_observation = env.renderer.build_generation_prompt(env.all_messages)

    if next_observation.length > env.cfg.max_prompt_tokens:
        env.truncated = True # type: ignore
        return done_step_result(env, reward=0.0)

    return StepResult(
        reward=0.0,
        episode_done=False,
        next_observation=next_observation,
        next_stop_condition=env.stop_condition,
        metrics={},
    )
    
def error_parsing_tool_call_step_result(env: Env, error: "ErrorParsingToolCall") -> StepResult:
    message: str = (
        "There was an error parsing the tool call:\n\n"
        + error.message
        + disable_thinking_prompt(env)
    )

    return new_user_message_step_result(env, message)

def next_step_result(env: Env, tool_output: ProcessOutput) -> StepResult:
    assert env.all_messages is not None, "All messages are not initialized"
    num_assistant_steps = len([message for message in env.all_messages if message["role"] == "assistant"])
    remaining_steps: int = env.cfg.max_steps - num_assistant_steps

    message: str = (
        "You have "
        + str(remaining_steps)
        + " remaining tool calls.\n\nHere is the output of the tool you called:\n\nEXIT CODE: "
        + str(tool_output.exit_code)
        + "\n\nSTDOUT: "
        + truncate(tool_output.stdout, env.cfg.truncate_command_outputs_length)
        + "\n\nSTDERR: "
        + truncate(tool_output.stderr, env.cfg.truncate_command_outputs_length)
        + disable_thinking_prompt(env)
    )

    return new_user_message_step_result(env, message)
        
### Misc. Helpers
    
def disable_thinking_prompt(env: Env) -> str:
    try:
        if env.cfg.qwen3_disable_thinking:
            return " /no_think"
        else:
            return ""
    except AttributeError:
        return ""
    
    
def truncate(s: str, max_length: int) -> str:
    if len(s) <= max_length:
        return s

    return s[: max_length // 2] + "\n\n[TRUNCATED]\n\n" + s[-max_length // 2 :]
