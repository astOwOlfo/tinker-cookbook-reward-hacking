"""Configuration for the Rubric Environment."""

from dataclasses import dataclass
from typing import Literal

from scalable_docker.client import MultiCommandTimeout


@dataclass(frozen=True, slots=True)
class RubricEnvConfig:
    """Configuration for the rubric-graded environment.

    This environment presents diverse non-SWE tasks (essays, bash challenges,
    data analysis, etc.) and grades them via an agentic LLM grader using
    a detailed rubric. Each category has a configurable `points` value.

    Scoring:
        Graded categories: Failure / Minor Failure / Minor Success / Success
            Mapped to 0, ≈1/3, ≈2/3, and full points for that category.
        Binary categories: Yes / No
            Mapped to full points or 0 for that category.

        Final reward = sum(category_scores) / sum(category_max_points),
        yielding [0.0, 1.0].
    """

    # Agent config
    max_steps: int = 10
    truncate_command_outputs_length: int = 2048
    max_prompt_tokens: int = 32768 - 8192
    can_finish: bool = True
    qwen3_disable_thinking: bool = False

    # Grader config
    max_grader_steps: int = 8
    grader_model: str = "gpt-5-mini-2025-08-07"
    grader_max_completion_tokens: int = 4096

    # Timeouts
    startup_command_timeout: MultiCommandTimeout = MultiCommandTimeout(
        seconds_per_command=30, total_seconds=60
    )
    tool_timeout: MultiCommandTimeout = MultiCommandTimeout(
        seconds_per_command=60, total_seconds=120
    )
    grader_tool_timeout: MultiCommandTimeout = MultiCommandTimeout(
        seconds_per_command=30, total_seconds=60
    )

    # Reward hacking prompt configuration
    reward_hacking_prompt: Literal[
        "allow", "must", "forbid-with-hints", "forbid", "nothing"
    ] = "nothing"

    # Number of seed variants per seedable factory. Controls dataset size:
    # total problems ≈ N_static + N_seedable × num_seeds.
    num_seeds: int = 15
