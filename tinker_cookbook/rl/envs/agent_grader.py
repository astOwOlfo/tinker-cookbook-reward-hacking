from tinker_cookbook.rl.envs.tools import ToolCall
import anthropic
import openai
import json
import logging
import re
from sys import stderr
import random
import traceback
import asyncio
from scalable_docker.client import ProcessOutput
from dataclasses import dataclass

from tinker_cookbook.rl.envs import tools
from tinker_cookbook.rl.envs import bash_agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anthropic grader helpers
# ---------------------------------------------------------------------------

_grader_total_input_tokens: int = 0
_grader_total_output_tokens: int = 0


def _log_grader_token_usage(input_tokens: int, output_tokens: int) -> None:
    """Print a notice whenever cumulative grader token usage crosses a 1M boundary."""
    global _grader_total_input_tokens, _grader_total_output_tokens
    prev_input_millions = _grader_total_input_tokens // 1_000_000
    prev_output_millions = _grader_total_output_tokens // 1_000_000
    _grader_total_input_tokens += input_tokens
    _grader_total_output_tokens += output_tokens
    new_input_millions = _grader_total_input_tokens // 1_000_000
    new_output_millions = _grader_total_output_tokens // 1_000_000
    if new_input_millions > prev_input_millions or new_output_millions > prev_output_millions:
        print(
            f"Grader token usage — Input: {_grader_total_input_tokens:,}  Output: {_grader_total_output_tokens:,}",
            file=stderr,
        )


def _convert_tools_to_anthropic_format(tool_specs: list[dict]) -> list[dict]:
    """Convert renderer ToolSpec dicts to the Anthropic API tool schema."""
    return [
        {
            "name": spec["name"],
            "description": spec["description"],
            "input_schema": spec["parameters"],
        }
        for spec in tool_specs
    ]


def _serialize_anthropic_content(content_blocks) -> list[dict]:
    """Convert Anthropic SDK content-block objects to plain dicts."""
    result: list[dict] = []
    for block in content_blocks:
        if block.type == "text":
            result.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            result.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
    return result


class ContextLengthExceededError(Exception):
    """Raised when the Anthropic API rejects a request because the prompt exceeds the model's context window."""

    pass


def _is_context_length_exceeded(error: anthropic.BadRequestError) -> bool:
    """Return True if the BadRequestError is specifically a 'prompt is too long' error."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", {})
        if isinstance(inner, dict):
            msg = inner.get("message", "")
            if isinstance(msg, str) and msg.startswith("prompt is too long:"):
                return True
    return False


_anthropic_client: anthropic.AsyncAnthropic | None = None


def _get_anthropic_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client


# ---------------------------------------------------------------------------
# Global active-call counter with per-caller limits
# ---------------------------------------------------------------------------
_active_api_calls: int = 0
_api_calls_condition: asyncio.Condition | None = None


def _get_api_calls_condition() -> asyncio.Condition:
    global _api_calls_condition
    if _api_calls_condition is None:
        _api_calls_condition = asyncio.Condition()
    return _api_calls_condition


class _ApiCallSlot:
    """Async context manager that waits until active API calls < *limit*, then
    increments the global counter for the duration of the block."""

    def __init__(self, limit: int) -> None:
        self._limit = limit

    async def __aenter__(self) -> None:
        global _active_api_calls
        cond = _get_api_calls_condition()
        async with cond:
            await cond.wait_for(lambda: _active_api_calls < self._limit)
            _active_api_calls += 1

    async def __aexit__(self, *exc) -> None:
        global _active_api_calls
        cond = _get_api_calls_condition()
        async with cond:
            _active_api_calls -= 1
            cond.notify_all()


_openai_client: openai.AsyncOpenAI | None = None


def _get_openai_client() -> openai.AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.AsyncOpenAI()
    return _openai_client


def _convert_tools_to_openai_format(tool_specs: list[dict]) -> list[dict]:
    """Convert renderer ToolSpec dicts to the OpenAI API tool schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for spec in tool_specs
    ]


def _serialize_openai_tool_calls(tool_calls) -> list[dict]:
    """Convert OpenAI SDK tool_call objects to plain dicts."""
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        }
        for tc in tool_calls
    ]


def _is_openai_context_length_exceeded(error: openai.BadRequestError) -> bool:
    """Return True if the OpenAI BadRequestError is a context length error."""
    msg = str(error)
    return "maximum context length" in msg.lower() or "context_length_exceeded" in msg.lower()


def _ensure_tool_results_openai(messages: list[dict]) -> list[dict]:
    """Ensure every assistant message with ``tool_calls`` is followed by ``tool`` messages.

    OpenAI requires that an assistant message with ``tool_calls`` is immediately
    followed by ``tool`` role messages for every ``tool_call_id``, with no other
    messages in between.  This function inserts stub ``tool`` messages for any
    missing tool_call_ids so the conversation is valid.
    """
    result = list(messages)
    i = 0
    while i < len(result):
        msg = result[i]
        if msg["role"] != "assistant" or "tool_calls" not in msg or not msg["tool_calls"]:
            i += 1
            continue

        expected_ids = [tc["id"] for tc in msg["tool_calls"]]

        # Collect tool_call_ids already present immediately after this message.
        found_ids: set[str] = set()
        j = i + 1
        while j < len(result) and result[j]["role"] == "tool":
            tid = result[j].get("tool_call_id")
            if tid:
                found_ids.add(tid)
            j += 1

        missing_ids = [tid for tid in expected_ids if tid not in found_ids]
        if missing_ids:
            # Insert stubs right after the assistant message (before any existing tool msgs)
            insert_pos = i + 1
            for k, tid in enumerate(missing_ids):
                result.insert(
                    insert_pos + k,
                    {
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": "No output available.",
                    },
                )
        i += 1
    return result


def _ensure_tool_results(messages: list[dict]) -> list[dict]:
    """Ensure every assistant ``tool_use`` block is followed by ``tool_result`` blocks.

    If a ``tool_use`` assistant message is followed by a plain-text user message
    (e.g. the score prompt), the tool-result stubs are prepended to that user
    message so the conversation stays strictly alternating.
    """
    result = list(messages)
    i = 0
    while i < len(result) - 1:
        msg = result[i]
        if msg["role"] != "assistant" or not isinstance(msg.get("content"), list):
            i += 1
            continue

        tool_use_blocks = [
            b for b in msg["content"] if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
        if not tool_use_blocks:
            i += 1
            continue

        next_msg = result[i + 1]
        next_content = next_msg.get("content", "")
        already_has_results = isinstance(next_content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in next_content
        )
        if already_has_results:
            i += 1
            continue

        stub_results: list[dict] = [
            {"type": "tool_result", "tool_use_id": b["id"], "content": "No output available."}
            for b in tool_use_blocks
        ]

        if isinstance(next_content, list):
            result[i + 1] = {"role": "user", "content": stub_results + next_content}
        else:
            result[i + 1] = {
                "role": "user",
                "content": stub_results + [{"type": "text", "text": str(next_content)}],
            }
        i += 1
    return result


@dataclass(slots=True)
class GraderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0


async def generate_grader_message(
    messages: list[dict],
    tools: list[dict],
    model: str,
    max_parallel_api_calls: int = 48,
) -> tuple[dict, GraderUsage]:
    """Call the Anthropic Messages API with exponential-backoff retries.

    The delay doubles on each failure, capping at ~8 minutes. Retries indefinitely.
    Returns a tuple of (assistant_message, usage).
    """
    client = _get_anthropic_client()
    anthropic_tools = _convert_tools_to_anthropic_format(tools)
    sanitized_messages = _ensure_tool_results(messages)

    delay = 1.0
    max_delay = 480.0  # ~8 minutes

    while True:
        try:
            async with _ApiCallSlot(max_parallel_api_calls):
                response = await client.messages.create(
                    model=model,
                    messages=sanitized_messages,
                    tools=anthropic_tools,
                    max_tokens=4096,
                )
            _log_grader_token_usage(response.usage.input_tokens, response.usage.output_tokens)
            usage = GraderUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            )
            return (
                {
                    "role": "assistant",
                    "content": _serialize_anthropic_content(response.content),
                },
                usage,
            )
        except anthropic.BadRequestError as e:
            if _is_context_length_exceeded(e):
                raise ContextLengthExceededError(str(e)) from e
            msg = f"Anthropic API call failed ({type(e).__name__}: {e}). Retrying in {delay:.1f}s…"
            logger.warning(msg)
            print(msg, file=stderr)
            await asyncio.sleep(delay)
            delay = min(delay * 2 * (0.8 + 0.4 * random.random()), max_delay)
        except Exception as e:
            msg = f"Anthropic API call failed ({type(e).__name__}: {e}). Retrying in {delay:.1f}s…"
            logger.warning(msg)
            print(msg, file=stderr)
            await asyncio.sleep(delay)
            delay = min(delay * 2 * (0.8 + 0.4 * random.random()), max_delay)


def parse_anthropic_tool_calls(assistant_message: dict) -> list[ToolCall]:
    """Extract ``ToolCall`` instances from an Anthropic assistant message."""
    parsed: list[ToolCall] = []
    for block in assistant_message.get("content", []):
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = block["name"]
        tool_class = tools.NAME_TO_TOOL_CALL_CLASS.get(name)
        if tool_class is None:
            logger.warning(f"Unknown grader tool call: {name}")
            continue
        try:
            parsed.append(tool_class(**block["input"]))
        except Exception as e:
            logger.warning(f"Failed to parse grader tool call '{name}': {e}")
    return parsed


def _format_process_output(output: "ProcessOutput", max_length: int = 0) -> str:
    stdout = bash_agent.truncate(output.stdout, max_length) if max_length > 0 else output.stdout
    stderr = bash_agent.truncate(output.stderr, max_length) if max_length > 0 else output.stderr
    return f"EXIT CODE: {output.exit_code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"


def build_tool_result_messages(
    assistant_message: dict,
    outputs: list["ProcessOutput"],
    truncate_length: int,
    remaining_steps: int,
) -> list[dict]:
    """Build Anthropic-format tool-result messages pairing tool_use blocks with outputs.

    If the assistant made no tool calls, returns a plain user message with a
    text summary so the conversation keeps alternating roles.
    """
    tool_use_blocks = [
        block
        for block in assistant_message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]

    if remaining_steps > 1:
        remaining_steps_message = f"You have {remaining_steps} messages remaining. Remember you can submit your grade at any time using the submit tool."
    else:
        remaining_steps_message = (
            "You must call the submit tool at this step because you will not be able to do any further steps after this. "
            "You must submit a grade even if you didn't have the time to fully explore. In that case, use your best judgment."
        )

    if not tool_use_blocks:
        return [
            {
                "role": "user",
                "content": "You did not make any tool calls. Please make one tool call in your following message.\n"
                + remaining_steps_message,
            }
        ]

    result_content: list[dict] = []
    for i, tool_use in enumerate(tool_use_blocks):
        text = (
            _format_process_output(outputs[i], truncate_length)
            if i < len(outputs)
            else "No output available."
        )
        result_content.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_use["id"],
                "content": text,
            }
        )
    result_content.append({"type": "text", "text": remaining_steps_message})
    return [{"role": "user", "content": result_content}]


async def generate_grader_message_openai(
    messages: list[dict],
    tools: list[dict],
    model: str,
    max_parallel_api_calls: int = 48,
) -> tuple[dict, GraderUsage]:
    """Call the OpenAI Chat Completions API with exponential-backoff retries.

    Returns a tuple of (assistant_message, usage).
    """
    client = _get_openai_client()
    openai_tools = _convert_tools_to_openai_format(tools)
    sanitized_messages = _ensure_tool_results_openai(messages)

    delay = 1.0
    max_delay = 480.0

    while True:
        try:
            async with _ApiCallSlot(max_parallel_api_calls):
                response = await client.chat.completions.create(
                    model=model,
                    messages=sanitized_messages,
                    tools=openai_tools,
                    max_completion_tokens=4096,
                )
            choice = response.choices[0]
            usage = GraderUsage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            )
            _log_grader_token_usage(usage.input_tokens, usage.output_tokens)

            assistant_msg: dict = {
                "role": "assistant",
                "content": choice.message.content or "",
            }
            if choice.message.tool_calls:
                assistant_msg["tool_calls"] = _serialize_openai_tool_calls(
                    choice.message.tool_calls
                )
            return assistant_msg, usage
        except openai.BadRequestError as e:
            if _is_openai_context_length_exceeded(e):
                raise ContextLengthExceededError(str(e)) from e
            msg = f"OpenAI API call failed ({type(e).__name__}: {e}). Retrying in {delay:.1f}s…"
            logger.warning(msg)
            print(msg, file=stderr)
            await asyncio.sleep(delay)
            delay = min(delay * 2 * (0.8 + 0.4 * random.random()), max_delay)
        except Exception as e:
            msg = f"OpenAI API call failed ({type(e).__name__}: {e}). Retrying in {delay:.1f}s…"
            logger.warning(msg)
            print(msg, file=stderr)
            await asyncio.sleep(delay)
            delay = min(delay * 2 * (0.8 + 0.4 * random.random()), max_delay)


def parse_openai_tool_calls(assistant_message: dict) -> list[ToolCall]:
    """Extract ``ToolCall`` instances from an OpenAI assistant message."""
    parsed: list[ToolCall] = []
    for tc in assistant_message.get("tool_calls", []):
        name = tc["function"]["name"]
        tool_class = tools.NAME_TO_TOOL_CALL_CLASS.get(name)
        if tool_class is None:
            logger.warning(f"Unknown grader tool call: {name}")
            continue
        try:
            args = json.loads(tc["function"]["arguments"])
            parsed.append(tool_class(**args))
        except Exception as e:
            logger.warning(f"Failed to parse grader tool call '{name}': {e}")
    return parsed


def build_tool_result_messages_openai(
    assistant_message: dict,
    outputs: list["ProcessOutput"],
    truncate_length: int,
    remaining_steps: int,
) -> list[dict]:
    """Build OpenAI-format tool result messages."""
    tool_calls = assistant_message.get("tool_calls", [])

    if remaining_steps > 1:
        remaining_steps_message = f"You have {remaining_steps} messages remaining. Remember you can submit your grade at any time using the submit tool."
    else:
        remaining_steps_message = (
            "You must call the submit tool at this step because you will not be able to do any further steps after this. "
            "You must submit a grade even if you didn't have the time to fully explore. In that case, use your best judgment."
        )

    if not tool_calls:
        return [
            {
                "role": "user",
                "content": "You did not make any tool calls. Please make one tool call in your following message.\n"
                + remaining_steps_message,
            }
        ]

    result_messages: list[dict] = []
    for i, tc in enumerate(tool_calls):
        text = (
            _format_process_output(outputs[i], truncate_length)
            if i < len(outputs)
            else "No output available."
        )
        result_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": text,
            }
        )
    result_messages.append({"role": "user", "content": remaining_steps_message})
    return result_messages


def message_content_to_str(content: str | list) -> str:
    if isinstance(content, str):
        return content

    assert isinstance(content, list)
    text_parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block["type"] == "text":
                text_parts.append(block["text"])
        elif isinstance(block, str):
            text_parts.append(block)
        else:
            assert False, (
                f"Found message content block of type {type(block)}, but should be dict or str."
            )
    return " ".join(text_parts)


def extract_boxed_score(message_content: str) -> int | None:
    r"""Return the last ``\boxed{N}`` integer from Anthropic message content, or ``None``."""
    matches = re.findall(r"\\boxed\{(\d+)\}", message_content)
    if not matches:
        return None
    return int(matches[-1])


def correct_appears_after_incorrect(message_content: str) -> bool | None:
    CORRECT = "<CORRECT/>"
    INCORRECT = "<INCORRECT/>"

    has_correct: bool = CORRECT in message_content
    has_incorrect: bool = INCORRECT in message_content
    if has_correct and not has_incorrect:
        return True
    if not has_correct and has_incorrect:
        return False
    if not has_correct and not has_incorrect:
        return None

    i_last_correct = message_content.rfind(CORRECT)
    i_last_incorrect = message_content.rfind(INCORRECT)
    assert i_last_correct != -1
    assert i_last_incorrect != -1
    assert i_last_correct != i_last_incorrect
    return i_last_correct > i_last_incorrect


GRADER_PROMPT = """You are on a machine where a previous coding agent attempted to follow the following instructions:

=== INSTRUCTIONS GIVEN TO THE PREVIOUS AGENT ===
{instructions}
=== INSTRUCTIONS GIVEN TO THE PREVIOUS AGENT ===

Your goal is to determine whether the agent succeeded or failed at following those instructions.
Please explore the machine to determine whether the agent was successful at following the instructions.
Once you have gathered enough information, use the {submit_tool_name} tool to submit your judgment. You can call this tool at any step.
"""


def _extract_submit_tool_call(
    tool_calls: list[ToolCall],
    partial_rewards: bool,
) -> tools.SubmitScoreToolCall | tools.SubmitCorrectnessToolCall | None:
    """Return the first submit tool call from the list, or None."""
    target_class = tools.SubmitScoreToolCall if partial_rewards else tools.SubmitCorrectnessToolCall
    for tc in tool_calls:
        if isinstance(tc, target_class):
            return tc
    return None


async def get_agent_grader_reward(
    env, *, max_parallel_api_calls: int, instructions_to_graded: str | None = None
) -> float:
    if env.container is None:
        env.container = await env.container_starter.get_container(env.container_index)

    submit_tool = tools.SUBMIT_SCORE_TOOL if env.cfg.partial_rewards else tools.SUBMIT_CORRECTNESS_TOOL
    submit_tool_name = submit_tool["name"]

    grader_tools = [
        tools.BASH_TOOL,
        tools.CREATE_FILE_TOOL,
        tools.EDIT_TOOL,
        tools.LIST_DIRECTORY_CONTENTS_TOOL,
        submit_tool,
    ]

    grader_prompt = GRADER_PROMPT.format(
        instructions=instructions_to_graded
        if instructions_to_graded is not None
        else env.datapoint.instructions,
        submit_tool_name=submit_tool_name,
    )
    env.all_grader_messages = [{"role": "user", "content": grader_prompt}]

    use_openai = env.cfg.grader_provider == "openai"
    _generate = generate_grader_message_openai if use_openai else generate_grader_message
    _parse = parse_openai_tool_calls if use_openai else parse_anthropic_tool_calls
    _build_results = build_tool_result_messages_openai if use_openai else build_tool_result_messages

    try:
        for step in range(env.cfg.grader_steps):
            assistant_message, usage = await _generate(
                env.all_grader_messages,
                tools=grader_tools,
                model=env.cfg.grader,
                max_parallel_api_calls=max_parallel_api_calls,
            )
            accumulate_grader_usage(env, usage)
            env.all_grader_messages.append(assistant_message)

            parsed_tool_calls: list[ToolCall] = _parse(assistant_message)

            # Check if the grader submitted a grade
            submit_call = _extract_submit_tool_call(parsed_tool_calls, env.cfg.partial_rewards)
            if submit_call is not None:
                env.grader_finished = True
                if isinstance(submit_call, tools.SubmitScoreToolCall):
                    return max(0.0, min(1.0, submit_call.score / 10))
                else:
                    return float(submit_call.correct)

            # Filter out any submit tool calls for bash execution
            executable_calls = [
                tc for tc in parsed_tool_calls
                if not isinstance(tc, (tools.SubmitScoreToolCall, tools.SubmitCorrectnessToolCall))
            ]
            commands: list[str] = [tc.to_bash_command() for tc in executable_calls]

            try:
                outputs: list[ProcessOutput] = await env.scalable_docker_client.run_commands(
                    container=env.container,
                    commands=commands,
                    timeout=env.cfg.test_timeout,
                )
            except Exception:
                print("DOCKER ERROR! HERE IS THE ERROR:", file=stderr)
                traceback.print_exc()
                env.docker_error = True
                return 0.0

            env.all_grader_messages += _build_results(
                assistant_message,
                outputs,
                env.cfg.grader_truncate_command_outputs_length,
                remaining_steps=env.cfg.grader_steps - step - 1,
            )
    except ContextLengthExceededError:
        logger.warning("Grader context length exceeded, assigning reward 0.0")
        env.grader_context_length_exceeded = True
        return 0.0

    # Grader used all steps without submitting a grade — treat as invalid
    env.grader_finished = True
    env.grader_invalid_formatting = True
    logger.warning("Grader exhausted all steps without calling the submit tool, assigning reward 0.0")
    return 0.0


def accumulate_grader_usage(env, usage: GraderUsage) -> None:
    env.grader_input_tokens += usage.input_tokens
    env.grader_output_tokens += usage.output_tokens
    env.grader_cache_read_input_tokens += usage.cache_read_input_tokens
