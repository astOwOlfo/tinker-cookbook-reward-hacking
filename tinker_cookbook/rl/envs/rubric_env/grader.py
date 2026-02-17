"""Agentic grader for the Rubric Environment.

The grader is an LLM that:
1. Receives the rubric and the model's conversation history
2. Can run bash commands in the container to inspect files and verify claims
3. Outputs structured scores for each rubric category

Uses an OpenAI-compatible API (AsyncOpenAI) with retry logic.
"""

import asyncio
import random
import re
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from scalable_docker.client import (
    ScalableDockerClient,
    Container,
    MultiCommandTimeout,
    TIMED_OUT_PROCESS_OUTPUT,
)

from .config import RubricEnvConfig
from .dataset import (
    AnyRubricCategory,
    BinaryRubricCategory,
    RubricCategory,
    RubricDatapoint,
    SCORE_LEVELS,
    BINARY_SCORE_LEVELS,
    SCORE_LEVEL_FRACTIONS,
    BINARY_LEVEL_FRACTIONS,
    score_for_level,
    max_score,
)
from .prompts import (
    GRADER_SYSTEM_PROMPT,
    GRADER_USER_PROMPT,
    format_rubric_for_grader,
)


# =============================================================================
# GRADER RESULT
# =============================================================================


@dataclass(slots=True)
class GraderResult:
    """Result from the agentic grader.

    Attributes:
        category_scores: Dict mapping category name -> integer score
            (0 to that category's max points).
        total_score: Sum of all category scores.
        max_possible_score: Sum of all category max points.
        reward: Normalized reward (0.0-1.0), i.e. total_score / max_possible_score.
        grader_n_steps: Number of grader turns taken.
        grader_n_tool_timeouts: Number of tool timeouts during grading.
        grader_parse_error: Whether the grader's final output failed to parse.
        raw_response: The grader's final raw response text.
    """

    category_scores: dict[str, int]
    total_score: int
    max_possible_score: int
    reward: float
    grader_n_steps: int = 0
    grader_n_tool_timeouts: int = 0
    grader_parse_error: bool = False
    raw_response: str = ""

    @property
    def mean_score(self) -> float:
        """Backward-compat: average score per category."""
        if self.category_scores:
            return sum(self.category_scores.values()) / len(self.category_scores)
        return 0.0


def compute_grader_result(
    category_scores: dict[str, int],
    rubric: tuple[AnyRubricCategory, ...],
    grader_n_steps: int = 0,
    grader_n_tool_timeouts: int = 0,
    grader_parse_error: bool = False,
    raw_response: str = "",
) -> GraderResult:
    """Compute a GraderResult from category scores and rubric.

    Reward = total_score / max_possible_score.
    """
    total = sum(category_scores.values())
    max_possible = max_score(rubric)

    reward = total / max_possible if max_possible > 0 else 0.0

    return GraderResult(
        category_scores=category_scores,
        total_score=total,
        max_possible_score=max_possible,
        reward=reward,
        grader_n_steps=grader_n_steps,
        grader_n_tool_timeouts=grader_n_tool_timeouts,
        grader_parse_error=grader_parse_error,
        raw_response=raw_response,
    )


# =============================================================================
# SCORE PARSING
# =============================================================================


def parse_rubric_scores(
    response: str,
    rubric: tuple[AnyRubricCategory, ...],
) -> dict[str, int] | None:
    """Parse rubric scores from the grader's XML output.

    Expected format:
        <rubric_scores>
          <category name="category_name">Score Level</category>
          ...
        </rubric_scores>

    For graded categories, Score Level is: Failure, Minor Failure, Minor Success, Success
    For binary categories, Score Level is: Yes, No

    Scores are mapped to numeric values using score_for_level(), which
    respects each category's `points` field.

    Returns:
        Dict mapping category name -> score, or None if parsing fails.
    """
    # Extract the rubric_scores block
    match = re.search(
        r"<rubric_scores>(.*?)</rubric_scores>",
        response,
        re.DOTALL,
    )
    if not match:
        return None

    scores_block = match.group(1)

    # Build category lookup by name
    cat_by_name: dict[str, AnyRubricCategory] = {cat.name: cat for cat in rubric}

    # Extract individual category scores
    category_pattern = re.compile(
        r'<category\s+name="([^"]+)">\s*(.*?)\s*</category>'
    )
    parsed: dict[str, int] = {}

    for cat_match in category_pattern.finditer(scores_block):
        name = cat_match.group(1).strip()
        level = cat_match.group(2).strip()

        if name in cat_by_name:
            # Use score_for_level which handles case-insensitive matching
            # and respects per-category point values
            parsed[name] = score_for_level(cat_by_name[name], level)
        else:
            # Category not in rubric — will be filtered out below
            parsed[name] = 0

    # Fill missing categories with 0
    rubric_names = {cat.name for cat in rubric}
    for name in rubric_names:
        if name not in parsed:
            parsed[name] = 0

    # Only keep scores for categories in the rubric
    return {name: parsed[name] for name in rubric_names if name in parsed}


# =============================================================================
# GRADER TOOL PARSING
# =============================================================================


@dataclass(frozen=True, slots=True)
class GraderBashCall:
    """A bash command extracted from grader output."""

    command: str


def extract_grader_tool_call(response: str) -> GraderBashCall | str | None:
    """Extract a bash tool call from the grader's response.

    Returns:
        GraderBashCall if a valid bash call was found.
        str if there was a parse error (the error message).
        None if no tool call was found.
    """
    match = re.search(r"<bash>(.*?)</bash>", response, re.DOTALL)
    if match:
        command = match.group(1).strip()
        if not command:
            return "Empty bash command."
        return GraderBashCall(command=command)
    return None


# =============================================================================
# OPENAI COMPLETION WITH RETRY
# =============================================================================


async def openai_completion(
    messages: list[dict[str, str]],
    model: str,
    max_completion_tokens: int,
) -> str | None:
    """Call an OpenAI-compatible API with retry logic.

    Args:
        messages: Chat messages.
        model: Model identifier.
        max_completion_tokens: Max tokens in response.

    Returns:
        The response text, or None if all retries failed.
    """
    client = AsyncOpenAI()

    for i in range(8):
        try:
            completion = await client.chat.completions.create(
                messages=messages,
                model=model,
                max_completion_tokens=max_completion_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Grader OpenAI call error (attempt {i + 1}/8): {e}")

        await asyncio.sleep(2**i + random.random())

    return None


# =============================================================================
# AGENTIC GRADER LOOP
# =============================================================================


def summarize_conversation(messages: list[dict[str, str]], max_chars: int = 8000) -> str:
    """Summarize the agent's conversation for the grader.

    Extracts the key actions and outputs, truncating if needed.
    """
    summary_parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle structured content (e.g., tool results)
            content = " ".join(
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        # Truncate individual messages
        if len(content) > 2000:
            content = content[:2000] + "\n... [truncated]"
        summary_parts.append(f"[{role}]: {content}")

    full_summary = "\n\n".join(summary_parts)
    if len(full_summary) > max_chars:
        full_summary = full_summary[:max_chars] + "\n\n... [conversation truncated]"

    return full_summary


async def grade(
    datapoint: RubricDatapoint,
    container: Container,
    client: ScalableDockerClient,
    conversation_history: list[dict[str, str]],
    cfg: RubricEnvConfig,
) -> GraderResult:
    """Run the agentic grader loop.

    The grader:
    1. Gets the rubric and conversation history
    2. Can run bash commands in the container to verify
    3. Outputs structured rubric scores

    Args:
        datapoint: The problem being graded.
        container: Docker container where the submission lives.
        client: ScalableDockerClient for running commands.
        conversation_history: The agent's message history.
        cfg: Environment configuration.

    Returns:
        GraderResult with scores and metadata.
    """
    rubric = datapoint.rubric
    rubric_section = format_rubric_for_grader(rubric)

    conversation_summary = summarize_conversation(conversation_history)

    system_prompt = GRADER_SYSTEM_PROMPT.format(rubric_section=rubric_section)
    user_prompt = GRADER_USER_PROMPT.format(
        problem_statement=datapoint.problem_statement,
        submission_instructions=datapoint.submission_instructions,
        conversation_summary=conversation_summary,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    n_tool_timeouts = 0

    for step in range(cfg.max_grader_steps):
        response = await openai_completion(
            messages=messages,
            model=cfg.grader_model,
            max_completion_tokens=cfg.grader_max_completion_tokens,
        )

        if response is None:
            # API failure — return zeros
            return compute_grader_result(
                category_scores={cat.name: 0 for cat in rubric},
                rubric=rubric,
                grader_n_steps=step + 1,
                grader_n_tool_timeouts=n_tool_timeouts,
                grader_parse_error=True,
                raw_response="OpenAI API call failed after all retries.",
            )

        # Check if grader produced final scores
        if "<rubric_scores>" in response:
            scores = parse_rubric_scores(response, rubric)
            if scores is not None:
                return compute_grader_result(
                    category_scores=scores,
                    rubric=rubric,
                    grader_n_steps=step + 1,
                    grader_n_tool_timeouts=n_tool_timeouts,
                    raw_response=response,
                )
            # Parse failed — tell grader to try again
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": (
                    "Your rubric_scores output could not be parsed. "
                    "Please output scores again using the exact format:\n"
                    "<rubric_scores>\n"
                    '  <category name="category_name">Score Level</category>\n'
                    "  ...one per category...\n"
                    "</rubric_scores>\n"
                    "For GRADED categories, Score Level is one of: Failure, Minor Failure, Minor Success, Success\n"
                    "For YES/NO categories, Score Level is one of: Yes, No"
                ),
            })
            continue

        # Check for tool call
        tool_call = extract_grader_tool_call(response)
        messages.append({"role": "assistant", "content": response})

        if isinstance(tool_call, str):
            # Parse error
            messages.append({"role": "user", "content": f"Tool error: {tool_call}"})
            continue

        if tool_call is None:
            # No tool call and no scores — nudge the grader
            messages.append({
                "role": "user",
                "content": (
                    "Please either run a bash command using <bash>command</bash> "
                    "to inspect the submission, or output your final scores using "
                    "<rubric_scores>...</rubric_scores>."
                ),
            })
            continue

        # Execute bash command in container
        try:
            outputs = await client.run_commands(
                container=container,
                commands=[tool_call.command],
                timeout=cfg.grader_tool_timeout,
            )
        except Exception as e:
            messages.append({
                "role": "user",
                "content": f"Error running command: {e}",
            })
            continue

        output = outputs[0]
        if output == TIMED_OUT_PROCESS_OUTPUT:
            n_tool_timeouts += 1
            messages.append({
                "role": "user",
                "content": "Command timed out.",
            })
            continue

        # Format output for grader
        stdout = output.stdout[:4000] if len(output.stdout) > 4000 else output.stdout
        stderr = output.stderr[:2000] if len(output.stderr) > 2000 else output.stderr

        output_text = f"Exit code: {output.exit_code}\n"
        if stdout:
            output_text += f"STDOUT:\n{stdout}\n"
        if stderr:
            output_text += f"STDERR:\n{stderr}\n"
        if not stdout and not stderr:
            output_text += "(no output)\n"

        messages.append({"role": "user", "content": output_text})

    # Grader exhausted all steps without producing scores
    # Try to extract scores from the last response if possible
    if messages and messages[-1].get("role") == "assistant":
        last_response = messages[-1]["content"]
        scores = parse_rubric_scores(last_response, rubric)
        if scores is not None:
            return compute_grader_result(
                category_scores=scores,
                rubric=rubric,
                grader_n_steps=cfg.max_grader_steps,
                grader_n_tool_timeouts=n_tool_timeouts,
                raw_response=last_response,
            )

    # Complete failure — return zeros
    return compute_grader_result(
        category_scores={cat.name: 0 for cat in rubric},
        rubric=rubric,
        grader_n_steps=cfg.max_grader_steps,
        grader_n_tool_timeouts=n_tool_timeouts,
        grader_parse_error=True,
        raw_response="Grader did not produce parseable scores within step limit.",
    )
