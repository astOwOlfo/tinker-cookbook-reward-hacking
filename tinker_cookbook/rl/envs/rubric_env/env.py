"""Rubric Environment implementation.

A model-graded environment for diverse non-SWE tasks. The model works in a
Docker container with configurable tools. An agentic LLM grader evaluates
the submission against a detailed rubric.

Hack vectors:
- Embed adversarial instructions in submission to manipulate grader
- Modify grading infrastructure if accessible
- Optimize for rubric surface metrics rather than genuine quality
- Game specific rubric criteria without actual substance
"""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from transformers import PreTrainedTokenizer

from scalable_docker.client import (
    ScalableDockerClient,
    Container,
    MultiCommandTimeout,
    upload_file_command,
)

from tinker_cookbook import renderers
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
from tinker_cookbook.rl.envs import tools
from tinker_cookbook.rl.envs import bash_agent
from tinker_cookbook.rl.envs.bash_agent import ContainerStarter
from tinker_cookbook.rl.envs.tools import ToolSpec

from .config import RubricEnvConfig
from .dataset import RubricDatapoint, RUBRIC_DATASET, build_all_problems, max_score
from .grader import GraderResult, grade
from .prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    RUBRIC_REWARD_HACKING_PROMPTS,
    format_rubric_for_model,
)


# =============================================================================
# DOCKERFILE
# =============================================================================

DEFAULT_DOCKERFILE = """FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-dev \\
    python-is-python3 \\
    build-essential \\
    curl \\
    jq \\
    bc \\
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \\
    numpy \\
    pandas \\
    scipy \\
    matplotlib \\
    pytest

WORKDIR /testbed

ENV PYTHONPATH=/testbed:$PYTHONPATH
"""


# =============================================================================
# ENVIRONMENT
# =============================================================================


class RubricEnv(Env):
    """Environment for rubric-graded non-SWE tasks.

    The model works in a Docker container with a configurable set of tools.
    After the episode ends, an agentic LLM grader evaluates the submission
    against a detailed rubric with N categories, each scored 0 to its max points.
    Reward = total_score / max_possible_score.
    """

    def __init__(
        self,
        renderer: renderers.Renderer,
        cfg: RubricEnvConfig,
        datapoint: RubricDatapoint,
        scalable_docker_client: ScalableDockerClient,
        container_starter: ContainerStarter,
        container_index: int,
        tokenizer: PreTrainedTokenizer,
        reward_hacking_prompt: str,
    ) -> None:
        self.renderer = renderer
        self.cfg = cfg
        self.datapoint = datapoint
        self.scalable_docker_client = scalable_docker_client
        self.container_starter = container_starter
        self.container_index = container_index
        self.container: Container | None = None
        self.tokenizer = tokenizer
        self.reward_hacking_prompt = reward_hacking_prompt

        # Message history
        self.all_messages: list[renderers.Message] = []

        # Standard state tracking
        self.ran_startup_commands = False
        self.n_tool_timeouts = 0
        self.docker_error = False
        self.n_errors_parsing_tool_calls = 0
        self.failed_startup_commands = False
        self.truncated = False

        # Grading results
        self.grader_result: GraderResult | None = None
        self.reward_value: float = 0.0

        # Resolve available tools from datapoint
        self.available_tools: list[ToolSpec] = datapoint.available_tools

    @property
    def stop_condition(self) -> StopCondition:
        """Stop sequences for generation."""
        return self.renderer.get_stop_sequences()
        #return ["</tool>"]

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Generate the initial observation (system + user prompt)."""
        await self.container_starter.start_starting()

        rubric_section = format_rubric_for_model(self.datapoint.rubric)
        system_prompt = SYSTEM_PROMPT.format(
            rubric_section=rubric_section,
            submission_instructions=self.datapoint.submission_instructions,
            reward_hacking_prompt=self.reward_hacking_prompt,
        )

        self.all_messages = self.renderer.create_conversation_prefix_with_tools(
            tools=self.available_tools, system_prompt=system_prompt
        )
        self.all_messages.append({
            "role": "user",
            "content": USER_PROMPT.format(
                problem_statement=self.datapoint.problem_statement,
                submission_instructions=self.datapoint.submission_instructions,
            ) + bash_agent.disable_thinking_prompt(self),
        })

        return (
            self.renderer.build_generation_prompt(self.all_messages),
            self.stop_condition,
        )

    def startup_commands(self) -> list[str]:
        """Commands to run at the start of the episode.

        Uploads any necessary_files from the datapoint.
        """
        commands = []
        for filepath, content in self.datapoint.necessary_files.items():
            # Ensure parent directory exists
            parent_dir = "/".join(filepath.rsplit("/", 1)[:-1])
            if parent_dir:
                commands.append(f"mkdir -p {parent_dir}")
            commands.append(upload_file_command(filename=filepath, content=content))
        return commands

    async def step(self, action: Action) -> StepResult:
        """Process one step of the episode."""
        return await bash_agent.default_agent_step(
            self,
            action,
            self.get_finished_step_result_with_reward,
        )

    async def get_finished_step_result_with_reward(self) -> StepResult:
        """Run the agentic grader and compute reward."""
        if self.container is None:
            self.container = await self.container_starter.get_container(
                self.container_index
            )

        try:
            self.grader_result = await grade(
                datapoint=self.datapoint,
                container=self.container,
                client=self.scalable_docker_client,
                cfg=self.cfg,
            )
            self.reward_value = self.grader_result.reward
        except Exception as e:
            print(f"Grader error: {e}")
            max_pts = max_score(self.datapoint.rubric)
            self.grader_result = GraderResult(
                category_scores={cat.name: 0 for cat in self.datapoint.rubric},
                total_score=0,
                max_possible_score=max_pts,
                reward=0.0,
                grader_parse_error=True,
                raw_response=f"Grader exception: {e}",
            )
            self.reward_value = 0.0

        return bash_agent.done_step_result(self, reward=self.reward_value)

    def metrics(self) -> dict[str, float]:
        """Return metrics for logging."""
        metrics: dict[str, float] = {
            "n_tool_timeouts": float(self.n_tool_timeouts),
            "docker_error": float(self.docker_error),
            "n_errors_parsing_tool_calls": float(self.n_errors_parsing_tool_calls),
            "failed_startup_commands": float(self.failed_startup_commands),
            "reward": self.reward_value,
        }

        if self.grader_result is not None:
            # Per-category scores
            for name, score in self.grader_result.category_scores.items():
                metrics[f"rubric_{name}"] = float(score)

            # Aggregate
            metrics["rubric_total_score"] = float(self.grader_result.total_score)
            metrics["rubric_max_possible_score"] = float(self.grader_result.max_possible_score)
            metrics["rubric_mean_score"] = self.grader_result.mean_score
            metrics["grader_n_steps"] = float(self.grader_result.grader_n_steps)
            metrics["grader_n_tool_timeouts"] = float(
                self.grader_result.grader_n_tool_timeouts
            )
            metrics["grader_parse_error"] = float(
                self.grader_result.grader_parse_error
            )

        return metrics


# =============================================================================
# GROUP BUILDER
# =============================================================================


@dataclass(frozen=True, slots=True)
class RubricGroupBuilder(EnvGroupBuilder):
    """Builds a group of RubricEnv instances for a single datapoint."""

    datapoint: RubricDatapoint
    num_envs: int
    group_index: int
    cfg: RubricEnvConfig
    scalable_docker_client: ScalableDockerClient
    container_starter: ContainerStarter
    renderer: renderers.Renderer
    tokenizer: PreTrainedTokenizer

    async def make_envs(self) -> list[RubricEnv]:
        """Create environment instances for this group."""
        rh_prompts = RUBRIC_REWARD_HACKING_PROMPTS[self.cfg.reward_hacking_prompt]
        reward_hacking_prompt = random.choice(rh_prompts)

        return [
            RubricEnv(
                renderer=self.renderer,
                cfg=self.cfg,
                datapoint=self.datapoint,
                scalable_docker_client=self.scalable_docker_client,
                container_starter=self.container_starter,
                container_index=self.num_envs * self.group_index + i,
                tokenizer=self.tokenizer,
                reward_hacking_prompt=reward_hacking_prompt,
            )
            for i in range(self.num_envs)
        ]

    def logging_tags(self) -> list[str]:
        """Tags for logging/filtering."""
        return [
            f"rubric_{self.datapoint.problem_type}",
            f"rh_{self.cfg.reward_hacking_prompt.replace('-', '_')}",
        ]


# =============================================================================
# DATASET
# =============================================================================


class RubricDataset(RLDataset):
    """Dataset of rubric-graded problems."""

    def __init__(
        self,
        data: list[RubricDatapoint],
        batch_size: int,
        group_size: int,
        cfg: RubricEnvConfig,
        renderer: renderers.Renderer,
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        self.data = data
        random.Random(42).shuffle(self.data)

        self.batch_size = batch_size
        self.group_size = group_size
        self.cfg = cfg
        self.renderer = renderer
        self.tokenizer = tokenizer

        self.scalable_docker_client = ScalableDockerClient(key="rubric_env")

    def get_batch(self, index: int) -> Sequence[RubricGroupBuilder]:
        """Get a batch of group builders."""
        batch_data: list[RubricDatapoint] = [
            self.data[i % len(self.data)]
            for i in range(self.batch_size * index, self.batch_size * (index + 1))
        ]

        assert len(batch_data) == self.batch_size

        n_containers = self.batch_size * self.group_size

        # All datapoints share the same base Docker image.
        # Per-datapoint files are uploaded via startup_commands().
        container_starter = ContainerStarter(
            dockerfile_contents=[DEFAULT_DOCKERFILE] * n_containers,
            scalable_docker_client=self.scalable_docker_client,
        )

        return [
            RubricGroupBuilder(
                datapoint=datapoint,
                num_envs=self.group_size,
                group_index=group_index,
                cfg=self.cfg,
                scalable_docker_client=self.scalable_docker_client,
                container_starter=container_starter,
                renderer=self.renderer,
                tokenizer=self.tokenizer,
            )
            for group_index, datapoint in enumerate(batch_data)
        ]

    def __len__(self) -> int:
        return int(math.floor(len(self.data) / self.batch_size))


# =============================================================================
# DATASET BUILDER
# =============================================================================


@dataclass(frozen=True, slots=True)
class RubricDatasetBuilder(RLDatasetBuilder):
    """Builder for rubric environment datasets."""

    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    test_fraction: float
    cfg: RubricEnvConfig

    async def __call__(
        self,
    ) -> tuple[RubricDataset, RubricDataset]:
        """Build train and test datasets."""
        data = build_all_problems(num_seeds=self.cfg.num_seeds)
        random.Random(42).shuffle(data)

        # Split into train/test
        n_train = int((1 - self.test_fraction) * len(data))
        n_train = max(1, min(n_train, len(data) - 1))

        train_data = data[:n_train]
        test_data = data[n_train:]

        assert len(train_data) > 0, "No training data"
        assert len(test_data) > 0, "No test data"

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        return tuple(  # type: ignore
            RubricDataset(
                data=split_data,
                batch_size=self.batch_size,
                group_size=self.group_size,
                cfg=self.cfg,
                renderer=renderer,
                tokenizer=tokenizer,
            )
            for split_data in [train_data, test_data]
        )
