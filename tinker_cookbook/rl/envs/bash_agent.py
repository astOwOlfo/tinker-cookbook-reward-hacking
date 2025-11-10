from dataclasses import dataclass, field
import asyncio
from scalable_docker.client import ScalableDockerClient, Container
from tinker_cookbook.rl.types import StepResult, Env
from tinker_cookbook.completers import StopCondition
import tinker
import traceback
from sys import stderr

@dataclass(slots=True)
class ContainerStarter:
    dockerfile_contents: list[str]
    scalable_docker_client: ScalableDockerClient
    _create_containers_task: asyncio.Task | None = None
    _lock: asyncio.Lock = field(default_factory=lambda: asyncio.Lock())

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

        all_containers: list[Container] = await self._create_containers_task

        return all_containers[index]
    
    
def done_step_result(env: Env, reward: float) -> StepResult:
    return StepResult(
        reward=reward,
        episode_done=True,
        next_observation=tinker.ModelInput.empty(),
        next_stop_condition=env.stop_condition,
        metrics=env.metrics(),
    )
        
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
    
def disable_thinking_prompt(env: Env) -> str:
    if not env.cfg.get("qwen3_disable_thinking"):
        return ""
    if env.cfg.qwen3_disable_thinking:
        return " /no_think"
    return ""

def new_user_message_step_result(env: Env, new_user_message: str) -> StepResult:
    
    assert env.renderer is not None, "Renderer is not initialized"
    assert env.all_messages is not None, "All messages are not initialized"
    assert env.cfg is not None, "Config is not initialized"
    assert env.stop_condition is not None, "Stop condition is not initialized"
    assert env.truncated is not None, "Truncated is not initialized"
    
    new_user_message += disable_thinking_prompt(env)

    env.all_messages.append({"role": "user", "content": new_user_message})

    next_observation = env.renderer.build_generation_prompt(env.all_messages)

    if next_observation.length > env.cfg.max_prompt_tokens:
        env.truncated = True
        return done_step_result(env, reward=0.0)

    return StepResult(
        reward=0.0,
        episode_done=False,
        next_observation=next_observation,
        next_stop_condition=env.stop_condition,
        metrics={},
    )
    
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
        return bash_agent.done_step_result(env, reward=0.0)
    
    
def truncate(s: str, max_length: int) -> str:
    if len(s) <= max_length:
        return s

    return s[: max_length // 2] + "\n\n[TRUNCATED]\n\n" + s[-max_length // 2 :]