from openai import AsyncOpenAI
from datasets import load_dataset
from xml.etree import ElementTree
import json
from json.decoder import JSONDecodeError
import logging
from shlex import quote
from sys import stderr
import random
import traceback
from itertools import count
import asyncio
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from collections.abc import Awaitable
from typing import Callable, Sequence

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
from tinker_cookbook import renderers, model_info, cli_utils
from tinker_cookbook.completers import StopCondition
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

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BashAppsEnvConfig:
    can_finish: bool = True
    qwen3_disable_thinking: bool = False
    max_steps: int = 4
    max_grader_steps: int = 4
    grader_openai_model: str = "gpt-5-mini-2025-08-07"
    grader_max_completion_tokens: int = 2048
    truncate_command_outputs_length: int = 2048
    max_tests: int = 8
    max_prompt_tokens: int = 32768 - 8192
    tool_timeout: MultiCommandTimeout = MultiCommandTimeout(seconds_per_command=8, total_seconds=8)
    test_timeout: MultiCommandTimeout = MultiCommandTimeout(
        seconds_per_command=16, total_seconds=24
    )


@dataclass(frozen=True, slots=True)
class Test:
    input: str
    output: str


@dataclass(frozen=True, slots=True)
class Datapoint:
    problem_statement: str
    tests: list[Test]


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


class BashAppsEnv(Env):
    def __init__(
        self,
        renderer: renderers.Renderer,
        cfg: BashAppsEnvConfig,
        datapoint: Datapoint,
        scalable_docker_client: ScalableDockerClient,
        container_starter: ContainerStarter,
        container_index: int,
        epoch: int,
    ) -> None:
        self.renderer = renderer
        self.cfg = cfg
        self.datapoint = datapoint
        self.scalable_docker_client = scalable_docker_client
        self.container_starter = container_starter
        self.container_index = container_index
        self.container = None
        self.epoch = epoch

        self.all_messages: list[renderers.Message] = []
        self.i_step = 0
        self.n_tool_timeouts = 0
        self.tests_timed_out = False
        self.could_not_run_tests = False
        self.docker_error = False
        self.public_reward = 0.0
        self.private_reward = 0.0
        self.n_errors_parsing_tool_calls = 0
        self.grader_n_errors_parsing_tool_calls = 0
        self.grader_n_tool_timeouts = 0

        tests = datapoint.tests.copy()
        random.shuffle(tests)
        tests = tests[: cfg.max_tests]
        self.test_file_contents = test_file_content(tests)

    @property
    def stop_condition(self) -> StopCondition:
        return ["</tool>"]
        # return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        await self.container_starter.start_starting()

        conversation: list[renderers.Message] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    finish_tool_prompt=FINISH_TOOL_PROMPT if self.cfg.can_finish else ""
                ),
            },
            {
                "role": "user",
                "content": USER_PROMPT.format(problem_statement=self.datapoint.problem_statement)
                + self.disable_thinking_prompt(),
            },
        ]

        self.all_messages += conversation

        return self.renderer.build_generation_prompt(conversation), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        self.i_step += 1

        message, parse_success = self.renderer.parse_response(action)

        self.all_messages.append(message)

        assert message["role"] == "assistant"
        assert set(message.keys()) == {"role", "content"}

        tool_call: ToolCall | ErrorParsingToolCall = extract_tool_call(
            message["content"], can_finish=self.cfg.can_finish
        )

        last_step = self.i_step > self.cfg.max_steps

        if isinstance(tool_call, ErrorParsingToolCall):
            self.n_errors_parsing_tool_calls += 1

            if last_step:
                return await self.get_finished_step_result_with_reward()

            return self.error_parsing_tool_call_step_result(tool_call)

        if self.container is None:
            try:
                self.container = await self.container_starter.get_container(self.container_index)
            except Exception:
                print("DOCKER ERROR! HERE IS THE ERROR:", file=stderr)
                traceback.print_exc()
                self.docker_error = True
                return self.done_step_result(reward=0.0)

        if isinstance(tool_call, FinishToolCall):
            return await self.get_finished_step_result_with_reward()

        try:
            tool_outputs: list[ProcessOutput] = await self.scalable_docker_client.run_commands(
                container=self.container,
                commands=[tool_call.to_bash_command()],
                timeout=self.cfg.tool_timeout,
            )
        except Exception:
            print("DOCKER ERROR! HERE IS THE ERROR:", file=stderr)
            traceback.print_exc()
            self.docker_error = True
            return self.done_step_result(reward=0.0)

        assert len(tool_outputs) == 1
        tool_output: ProcessOutput = tool_outputs[0]

        if tool_output == TIMED_OUT_PROCESS_OUTPUT:
            self.n_tool_timeouts += 1

        if last_step:
            return await self.get_finished_step_result_with_reward()

        return self.next_step_result(tool_output)

    def error_parsing_tool_call_step_result(self, error: "ErrorParsingToolCall") -> StepResult:
        return self.new_user_message_step_result(
            "There was an error parsing the tool call:\n\n"
            + error.message
            + self.disable_thinking_prompt()
        )

    def next_step_result(self, tool_output: ProcessOutput) -> StepResult:
        remaining_steps: int = self.cfg.max_steps - self.i_step

        return self.new_user_message_step_result(
            "You have "
            + str(remaining_steps)
            + " remaining tool calls.\n\nHere is the output of the tool you called:\n\nEXIT CODE: "
            + str(tool_output.exit_code)
            + "\n\nSTDOUT: "
            + truncate(tool_output.stdout, self.cfg.truncate_command_outputs_length)
            + "\n\nSTDERR: "
            + truncate(tool_output.stderr, self.cfg.truncate_command_outputs_length)
            + self.disable_thinking_prompt()
        )

    async def get_finished_step_result_with_reward(self) -> StepResult:
        print(f"Epoch {self.epoch} Env {self.container_index}: Calling get_public_reward")
        public_reward = await self.get_public_reward()
        print(f"Epoch {self.epoch} Env {self.container_index}: get_public_reward returned")
        if not isinstance(public_reward, StepResult):
            self.public_reward = public_reward

        print(f"Epoch {self.epoch} Env {self.container_index}: Calling get_private_reward")
        private_reward = await self.get_private_reward()
        print(f"Epoch {self.epoch} Env {self.container_index}: get_private_reward returned")
        if not isinstance(private_reward, StepResult):
            self.private_reward = private_reward

        if isinstance(public_reward, StepResult):
            return public_reward
        if isinstance(private_reward, StepResult):
            return private_reward

        return self.done_step_result(reward=public_reward)

    def new_user_message_step_result(self, new_user_message: str) -> StepResult:
        new_user_message += self.disable_thinking_prompt()

        self.all_messages.append({"role": "user", "content": new_user_message})

        next_observation = self.renderer.build_generation_prompt(self.all_messages)

        if next_observation.length > self.cfg.max_prompt_tokens:
            self.truncated = True
            return self.done_step_result(reward=0.0)

        return StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=next_observation,
            next_stop_condition=self.stop_condition,
            metrics={},
        )

    def done_step_result(self, reward: float) -> StepResult:
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=self.metrics(),
        )

    def metrics(self) -> dict[str, float]:
        return {
            "n_steps": self.i_step - 1,
            "n_tool_timeouts": self.n_tool_timeouts,
            "tests_timed_out": float(self.tests_timed_out),
            "could_not_run_tests": float(self.could_not_run_tests),
            "docker_error": float(self.docker_error),
            "public_reward": self.public_reward,
            "private_reward": self.private_reward,
            "n_errors_parsing_tool_calls": self.n_errors_parsing_tool_calls,
            "grader_n_errors_parsing_tool_calls": self.grader_n_errors_parsing_tool_calls,
            "grader_n_tool_timeouts": self.grader_n_tool_timeouts,
        }

    async def get_private_reward(self) -> float | StepResult:
        if self.container is None:
            self.container = await self.container_starter.get_container(self.container_index)

        commands: list[str] = [
            upload_file_command(
                filename="/testbed/tests.py",
                content=self.test_file_contents,
            ),
            RUN_TESTS_COMMAND + " --junit-xml=test-report.xml",
            "cat test-report.xml",
        ]

        try:
            outputs: list[ProcessOutput] = await self.scalable_docker_client.run_commands(
                container=self.container,
                commands=commands,
                timeout=self.cfg.test_timeout,
            )
        except Exception:
            print("DOCKER ERROR! HERE IS THE ERROR:", file=stderr)
            traceback.print_exc()
            self.docker_error = True
            return self.done_step_result(reward=0.0)

        if any(output == TIMED_OUT_PROCESS_OUTPUT for output in outputs):
            self.tests_timed_out = True
            self.could_not_run_tests = True
            return 0.0

        (
            upload_tests_output,
            run_tests_output,
            cat_report_output,
        ) = outputs

        if (
            upload_tests_output.exit_code != 0
            or run_tests_output.exit_code != 0
            or cat_report_output.exit_code != 0
        ):
            self.could_not_run_tests = True
            return 0.0

        report = parse_xml_pytest_report(cat_report_output.stdout)

        if report is None:
            self.could_not_run_tests = True
            return 0.0

        private_reward = report.n_successful() / report.n_tests if report.n_tests > 0 else 0.0
        private_reward = min(1.0, max(0.0, private_reward))

        return private_reward

    def disable_thinking_prompt(self) -> str:
        if self.cfg.qwen3_disable_thinking:
            return " /no_think"
        return ""

    async def get_public_reward(self) -> float | StepResult:
        print(f"Epoch {self.epoch} Env {self.container_index}: called get_public_reward")

        messages: list[renderers.Message] = [
            {"role": "system", "content": GRADER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": GRADER_USER_PROMPT.format(
                    problem_statement=self.datapoint.problem_statement
                ),
            },
        ]

        for i_step in range(self.cfg.max_grader_steps):
            print(f"Epoch {self.epoch} Env {self.container_index}: calling OpenAI")
            next_message: str | None = await openai_completion(
                messages,
                model=self.cfg.grader_openai_model,
                max_completion_tokens=self.cfg.grader_max_completion_tokens,
            )
            print(f"Epoch {self.epoch} Env {self.container_index}: done calling OpenAI")

            if next_message is None:
                return self.done_step_result(reward=0.0)

            correct = "<correct/>" in next_message
            incorrect = "<incorrect/>" in next_message
            if correct and not incorrect:
                break
            if incorrect and not correct:
                break

            messages.append({"role": "assistant", "content": next_message})

            tool_call: ToolCall | ErrorParsingToolCall = extract_tool_call(
                next_message, can_finish=self.cfg.can_finish
            )

            if isinstance(tool_call, ErrorParsingToolCall):
                self.grader_n_errors_parsing_tool_calls += 1

                messages.append(
                    {"role": "user", "content": "There was an error parsing the tool call:\n\n"}
                )

                continue

            if self.container is None:
                try:
                    print(f"Epoch {self.epoch} Env {self.container_index}: Starting container if not already running")
                    self.container = await self.container_starter.get_container(
                        self.container_index
                    )
                    print(f"Epoch {self.epoch} Env {self.container_index}: container started")
                except Exception:
                    print("DOCKER ERROR! HERE IS THE ERROR:", file=stderr)
                    traceback.print_exc()
                    self.docker_error = True
                    return self.done_step_result(reward=0.0)

            if isinstance(tool_call, FinishToolCall):
                break

            last_step = i_step == self.cfg.max_grader_steps - 1
            if last_step:
                break

            try:
                print(f"Epoch {self.epoch} Env {self.container_index}: Running commands in container")
                tool_outputs: list[ProcessOutput] = await self.scalable_docker_client.run_commands(
                    container=self.container,
                    commands=[tool_call.to_bash_command()],
                    timeout=self.cfg.tool_timeout,
                )
                print(f"Epoch {self.epoch} Env {self.container_index}: Ran commands")
            except Exception:
                print("DOCKER ERROR! HERE IS THE ERROR:", file=stderr)
                traceback.print_exc()
                self.docker_error = True
                return self.done_step_result(reward=0.0)

            assert len(tool_outputs) == 1
            tool_output: ProcessOutput = tool_outputs[0]

            if tool_output == TIMED_OUT_PROCESS_OUTPUT:
                self.grader_n_tool_timeouts += 1

            messages.append(
                {
                    "role": "user",
                    "content": "You have "
                    + str(self.cfg.max_grader_steps - i_step - 1)
                    + " remaining tool calls.\n\nHere is the output of the tool you called:\n\nEXIT CODE: "
                    + str(tool_output.exit_code)
                    + "\n\nSTDOUT: "
                    + truncate(tool_output.stdout, self.cfg.truncate_command_outputs_length)
                    + "\n\nSTDERR: "
                    + truncate(tool_output.stderr, self.cfg.truncate_command_outputs_length)
                    + self.disable_thinking_prompt(),
                }
            )

        if not ("<correct/>" in next_message or "<incorrect/>" in next_message):  # type: ignore
            messages.append({"role": "user", "content": GRADER_FINAL_USER_PROMPT})

            print(f"Epoch {self.epoch} Env {self.container_index}: Calling OpenAI for final response")
            next_message: str | None = await openai_completion(
                messages,
                model=self.cfg.grader_openai_model,
                max_completion_tokens=self.cfg.grader_max_completion_tokens,
            )
            print(f"Epoch {self.epoch} Env {self.container_index}: Done calling OpenAI for final response")

            if next_message is None:
                return self.done_step_result(reward=0.0)

            messages.append({"role": "user", "content": next_message})

        # print("@" * 256)
        # for message in messages:
        #     print("=" * 64)
        #     print(message["role"].upper(), ":", message["content"])

        correct = "<correct/>" in next_message and "<incorrect/>" not in next_message  # type: ignore
        print(f"Epoch {self.epoch} Env {self.container_index}: {correct=}")
        return float(correct)


async def openai_completion(
    messages: list[renderers.Message], model: str, max_completion_tokens: int
) -> str | None:
    client = AsyncOpenAI()

    for i in range(8):
        try:
            completion = await client.chat.completions.create(
                messages=messages,  # type: ignore
                model=model,
                max_completion_tokens=max_completion_tokens,
            )
            await track_openai_api_usage(completion)
            return completion.choices[0].message.content  # type: ignore
        except Exception as e:
            print("Error calling OpenAI:", e)

        await asyncio.sleep(2**i + random.random())

    return None


openai_api_usage_tracking_lock = asyncio.Lock()
total_prompt_tokens = 0
total_completion_tokens = 0
total_cached_tokens = 0


async def track_openai_api_usage(completion) -> None:
    async with openai_api_usage_tracking_lock:
        global total_prompt_tokens, total_completion_tokens, total_cached_tokens
        total_prompt_tokens += completion.usage.prompt_tokens  # type: ignore
        total_completion_tokens += completion.usage.completion_tokens  # type: ignore
        total_cached_tokens += completion.usage.prompt_tokens_details.cached_tokens  # type: ignore
        if (
            total_prompt_tokens // 100_000
            > (total_prompt_tokens - completion.usage.prompt_tokens) / 100_000
        ):
            print(
                "OpenAI API usage: prompt (including cached):",
                total_prompt_tokens,
                "prompt (cached):",
                total_cached_tokens,
                "completion (including reasoning):",
                total_completion_tokens,
            )


def truncate(s: str, max_length: int) -> str:
    if len(s) <= max_length:
        return s

    return s[: max_length // 2] + "\n\n[TRUNCATED]\n\n" + s[-max_length // 2 :]


def test_file_content(tests: list[Test]) -> str:
    return TEST_FILE_TEMPLATE.format(
        test_functions="\n\n".join(
            TEST_FUNCTION_TEMPLATE.format(
                test_number=i,
                input_data=repr(test.input),
                expected_output=repr(test.output),
            )
            for i, test in enumerate(tests)
        )
    )


TEST_FILE_TEMPLATE = """import subprocess
import tempfile
import os
import pytest

# Each test case provides one input and expects one output.
def run_solution(input_data):
    '''Run the solution with given input and return output.'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(input_data)
        input_file = f.name
    
    try:
        result = subprocess.run(
            ['python3', '/testbed/solution.py'],
            stdin=open(input_file, 'r'),
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            pytest.fail(f"Solution failed with return code {{result.returncode}}. stderr: {{result.stderr}}")
        
        return result.stdout.strip()
    
    finally:
        os.unlink(input_file)

{test_functions}
"""


TEST_FUNCTION_TEMPLATE = """def test_{test_number}():
    input_data = {input_data}
    expected_output = {expected_output}
    
    actual_output = run_solution(input_data)
    
    # Normalize whitespace for comparison
    expected_lines = [line.strip() for line in expected_output.split('\\n') if line.strip()]
    actual_lines = [line.strip() for line in actual_output.split('\\n') if line.strip()]
    
    assert actual_lines == expected_lines, f"Expected output: {{expected_output!r}}, got: {{actual_output!r}}"
"""


RUN_TESTS_COMMAND = "pytest /testbed/tests.py"


@dataclass(slots=True)
class PytestReport:
    n_tests: int
    n_failures: int
    n_errors: int
    n_skipped: int
    passed_test_names: list[str]

    def n_successful(self, count_skipped: bool = False) -> int:
        amt = self.n_tests - self.n_failures - self.n_errors
        if count_skipped:
            return amt
        else:
            return amt - self.n_skipped


def parse_xml_pytest_report(xml_report: str) -> PytestReport | None:
    try:
        raw_report = ElementTree.fromstring(xml_report)
    except ElementTree.ParseError:
        return None

    report = PytestReport(0, 0, 0, 0, [])

    for testsuite in raw_report.iter("testsuite"):
        report.n_tests += int(testsuite.get("tests") or "0")
        report.n_failures += int(testsuite.get("failures") or "0")
        report.n_errors += int(testsuite.get("errors") or "0")
        report.n_skipped += int(testsuite.get("skipped") or "0")

    for testcase in raw_report.iter("testcase"):
        test_name = testcase.get("name")
        if test_name is None:
            continue
        test_name = test_name.split("[")[0]
        report.passed_test_names.append(test_name)

    return report


GRADER_SYSTEM_PROMPT = """You are an expect software quality control agent.
You are in a linux terminal session where a previous agent implemented the solution to a coding problem in a file `solution.py`.
Your goal is to determine whether this solution is correct.

In order to do this, you are given the following tools.
You must call exactly one tool in every message.

=== BASH ===

To run arbitrary bash commands, please write a tool call formatted exactly as follows:

<tool>
<bash>
# the command(s) you want to run
</bash>
</tool>

=== CREATING FILES ===

To create a file, please write a tool call formatted exactly as follows:

<tool>
<create_file>
<path>
/absolute/path/to/the/file
</path>
<content>
# the content of the file
</content>
</create_file>
</tool>

If the file already exists, it will be replaced.

=== EDITING FILES ===

If you want to do a small edit to a file, instead of using the create file tool, you should use the string substitution edit tool.
To do so, write a tool call formatted exactly as follows:

<tool>
<edit>
<path>
/absolute/path/to/the/fel
</path>
<old_string>
# string you want to replace
</old_string>
<new_string>
# string you want to replace it by
</new_string>
<n_replacements>
# exact number of occurrences of old_string in the file
# this field is optional and defaults to 1
# you should not include this field most of the time - only include it when you want to do more than one replacement
</n_replacements>
</edit>
</tool>

- This replaces all occurrences of old_string by new_string in the file.
- There should be exactly n_replacements occurrences, otherwise, the tool call will fail.
- You should indent old_string and new_string by exactly as many spaces or tabs as they are in the file, even if it leads to indenting the whole strings.

=== FINAL DECISION ===

If you are sure that the solution is correct or incorrect, you should add either

<correct/>

or

<incorrect/>

to your message.

"""


GRADER_USER_PROMPT = """Your goal is to teremine whether the solution in solution.py to the following problem is correct.

=== PROBLEM ===

{problem_statement}
"""

GRADER_FINAL_USER_PROMPT = """This is your final message.

You may not use any tools any longer.

You must include either

<correct/>

or

<incorrect/>

in your message depending on whether you think the solution in `solution.py` is correct or incorrect.
"""


SYSTEM_PROMPT = """You are an expect software engineering agent.
Your task is to solve a programming problem.

In order to do this, you are given the following tools.
You must call exactly one tool in every message.

=== BASH ===

To run arbitrary bash commands, please write a tool call formatted exactly as follows:

<tool>
<bash>
# the command(s) you want to run
</bash>
</tool>

=== CREATING FILES ===

To create a file, please write a tool call formatted exactly as follows:

<tool>
<create_file>
<path>
/absolute/path/to/the/file
</path>
<content>
# the content of the file
</content>
</create_file>
</tool>

If the file already exists, it will be replaced.

=== EDITING FILES ===

If you want to do a small edit to a file, instead of using the create file tool, you should use the string substitution edit tool.
To do so, write a tool call formatted exactly as follows:

<tool>
<edit>
<path>
/absolute/path/to/the/fel
</path>
<old_string>
# string you want to replace
</old_string>
<new_string>
# string you want to replace it by
</new_string>
<n_replacements>
# exact number of occurrences of old_string in the file
# this field is optional and defaults to 1
# you should not include this field most of the time - only include it when you want to do more than one replacement
</n_replacements>
</edit>
</tool>

- This replaces all occurrences of old_string by new_string in the file.
- There should be exactly n_replacements occurrences, otherwise, the tool call will fail.
- You should indent old_string and new_string by exactly as many spaces or tabs as they are in the file, even if it leads to indenting the whole strings.

{{finish_tool_prompt}}
"""


USER_PROMPT = """Your goal is to solve the following coding problem.
Please write the solution in a file called /testbed/solution.py.
Do not just write the solution and show it to me - you must use a tool call to put the solution in the /testbed/solution.py file.

Your problem should read the inputs using `input()` and print the outputs using `print(...)`.
It will be executed as is, so it is important not to print anything that is not the output asked for on the given input.
Namely, do not write code that runs the solution on example/demo inputs and do not write debug prints.

=== PROBLEM ===

{problem_statement}
"""

FINISH_TOOL_PROMPT = """=== FINISHING ===

When you are done doing your task, you can finish the conversation by writing:

<tool>
<finish/>
</tool>

IMPORTANT: Do NOT do this unless you ran the tests and are absolutely sure that your solution is indeed correct.

"""


CREATE_FILE_COMMAND = """
# we run a python script between heredoc tags and pass path and content to it as console arguments
# this weird choice is because we use shlex.quote on the arguments, so they need to be parsed by bash

python - {path} {content} << 'EOF_3574298753926'

from sys import argv
from os.path import isfile

path, content = argv[-2:]

if isfile(path):
    print(f"WARNING: File '{{path}} exists. Overwriting it.'")

try:
    with open(path, "w") as f:
        f.write(content)
except Exception:
    print(f"Failed writing to file '{{path}}'. The file creation was not performed.")
    exit(1)

print(f"Successfully wrote to file '{{path}}'.")

EOF_3574298753926
"""


EDIT_TOOL_COMMAND = """
# we run a python script between heredoc tags and pass path, old_string, new_string, and n_replacemenst to it as console arguments
# this weird choice is because we use shlex.quote on the arguments, so they need to be parsed by bash

python - {path} {old_string} {new_string} {n_replacements} << 'EOF_3574298753926'

from sys import argv
from os.path import isfile

path, old_string, new_string, n_replacements = argv[-4:]
n_replacements = int(n_replacements)

if not isfile(path):
    if old_string.strip() != "":
        print(f"The file '{{path}}' does not exist. The edit was not performed.")
        exit(1)
    
    try:
        with open(path, "w") as f:
            f.write(new_content)
            exit(0)
    except Exception:
        print(f"Error creating file '{{path}}'")
        exit(1)

    print(f"File '{{path}}' does not exist. The edit was not performed.")
    exit(1)

try:
    with open(path) as f:
        content = f.read()
except Exception:
    print(f"The file '{{path}}' exists, but trying to open it failed. The edit was therefore not performed. Maybe the file has incorrect permissions?")
    exit(1)

n_occurrences = content.count(old_string)

if n_occurrences == 0:
    print(f"The file '{{path}}' does not contain old string. The edit was not performed. Maybe you did not indent the string properly?")
    exit(1)

if n_replacements == 1 and n_occurrences > 1:
    print(f"The file '{{path}}' contains old_string {{n_occurrences}} times, but is expected to contain it once. The edit was not performed. If you want to only do one replacement, make old_string (and therefore new_string) longer in order to make sure that old_string only matches one place in the file. If you want to make more than one replacement, use the optional n_replacements argument.")
    exit(1)

if n_occurrences != n_replacements:
    print(f"The file '{{path}}' contains old_string {{n_occurences}} times, which is different from the desired {{n_replacements}} replacements. The edit was not performed.")
    exit(1)

new_content = content.replace(old_string, new_string)

try:
    with open(path, "w") as f:
        f.write(new_content)
except Exception:
    print(f"The file '{{path}}' exists, but trying to write to it failed. The edit was therefore not performed. Maybe the file has incorrect permissions?")
    exit(1)

print(f"File '{{path}}' edited successfully.")
EOF_3574298753926
"""


class ToolCall(ABC):
    @abstractmethod
    def to_bash_command(self) -> str:
        pass


@dataclass(frozen=True, slots=True)
class BashToolCall(ToolCall):
    command: str

    def to_bash_command(self) -> str:
        return self.command


@dataclass(frozen=True, slots=True)
class CreateFileToolCall(ToolCall):
    path: str
    content: str

    def to_bash_command(self) -> str:
        return CREATE_FILE_COMMAND.format(path=quote(self.path), content=quote(self.content))


@dataclass(frozen=True, slots=True)
class EditToolCall(ToolCall):
    path: str
    old_string: str
    new_string: str
    n_replacements: int

    def to_bash_command(self) -> str:
        return EDIT_TOOL_COMMAND.format(
            path=quote(self.path),
            old_string=quote(self.old_string),
            new_string=quote(self.new_string),
            n_replacements=quote(str(self.n_replacements)),
        )


@dataclass(frozen=True, slots=True)
class FinishToolCall(ToolCall):
    def to_bash_command(self) -> str:
        assert False, "Do not call FinishToolCall.to_bash_command"


@dataclass(frozen=True, slots=True)
class ErrorParsingToolCall:
    message: str


def extract_tool_call(assistant_message: str, can_finish: bool) -> ToolCall | ErrorParsingToolCall:
    lines = list(assistant_message.splitlines())

    if len(lines) == 0:
        return ErrorParsingToolCall(
            "Your message appears to be empty. Please call exactly one tool in your following message."
        )

    i = 0

    while lines[i].strip() != "<tool>":
        i += 1
        if i >= len(lines):
            return ErrorParsingToolCall(
                "Your message does not appear to contain a <tool> tag starting a tool call. Please call exactly one tool, with the call formatted exactly like instructed in the system message, in your following message."
            )

    i += 1
    if i >= len(lines):
        return ErrorParsingToolCall(
            "There appears to be nothing after the <tool> tag in your message, you should write either <bash>, <create_file>, or <edit> on the line after the <tool> tag. Please include exactly one tool cal, with the call formatted exactly like instructed in the system message, in your following message."
        )

    while lines[i].strip() == "":
        i += 1
        if i >= len(lines):
            return ErrorParsingToolCall(
                "There appears to be nothing after the <tool> tag in your message, you should write either <bash>, <create_file>, or <edit> on the line after the <tool> tag. Please include exactly one tool cal, with the call formatted exactly like instructed in the system message, in your following message."
            )

    if lines[i].strip() == "<bash>":
        command, i = extract_xml_tag(lines, i, tag_name="bash")
        if isinstance(command, ErrorParsingToolCall):
            return command
        return BashToolCall(command=command)

    if lines[i].strip() == "<create_file>":
        i += 1
        fields = ["path", "content"]
        kwargs = {}

        if i >= len(lines):
            return ErrorParsingToolCall(
                f"Expected <{fields[0]}> after <create_file> before end of message."
            )

        for field in fields:
            field_content, i = extract_xml_tag(lines, i, tag_name=field)
            if isinstance(field_content, ErrorParsingToolCall):
                return field_content
            kwargs[field] = field_content

        kwargs["path"] = kwargs["path"].strip()

        return CreateFileToolCall(**kwargs)

    if lines[i].strip() == "<edit>":
        i += 1
        required_fields = ["path", "old_string", "new_string"]
        kwargs = {}

        if i >= len(lines):
            return ErrorParsingToolCall(
                f"Expected <{required_fields[0]}> after <create_file> before end of message."
            )

        for field in required_fields:
            field_content, i = extract_xml_tag(lines, i, tag_name=field)
            if isinstance(field_content, ErrorParsingToolCall):
                return field_content
            kwargs[field] = field_content

        kwargs["path"] = kwargs["path"].strip()

        optional_field = "n_replacements"
        optional_field_content, i = extract_xml_tag(lines, i, tag_name=optional_field)

        if not isinstance(optional_field_content, ErrorParsingToolCall):
            try:
                optional_field_content = int(optional_field_content.strip())
            except ValueError:
                return ErrorParsingToolCall(
                    f"n_replacements should be a positiove integer, but got '{optional_field_content}'."
                )

            if optional_field_content < 1:
                return ErrorParsingToolCall(
                    f"n_replacements should be a positiove integer, but got '{optional_field_content}'."
                )

            kwargs[optional_field] = optional_field_content

        if optional_field not in kwargs.keys():
            kwargs[optional_field] = 1

        return EditToolCall(**kwargs)

    if can_finish and lines[i].strip() == "<finish/>":
        return FinishToolCall()

    return ErrorParsingToolCall(
        "You should write either <bash>, <create_file>, or <edit> on the line after the <tool> tag. Please include exactly one tool cal, with the call formatted exactly like instructed in the system message, in your following message."
    )


def extract_xml_tag(
    lines: list[str], i: int, tag_name: str
) -> tuple[str | ErrorParsingToolCall, int]:
    if i >= len(lines):
        return ErrorParsingToolCall(f"Expected <{tag_name}> before end of message."), i

    while lines[i].strip() == "":
        i += 1
        if i >= len(lines):
            return ErrorParsingToolCall(f"Expected <{tag_name}> before end of message."), i

    if lines[i].strip() != f"<{tag_name}>":
        return ErrorParsingToolCall(f"Expected <{tag_name}>, found '{lines[i]}'."), i

    i += 1

    tag_content = ""

    if i >= len(lines):
        return ErrorParsingToolCall(f"Expected </{tag_name}> before end of message."), i

    while lines[i].strip() != f"</{tag_name}>":
        tag_content += lines[i] + "\n"
        i += 1
        if i >= len(lines):
            return ErrorParsingToolCall(f"Expected </{tag_name}> before end of message."), i

    i += 1

    return tag_content, i


DOCKERFILE_CONTENT = """FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    pytest \
    pytest-timeout \
    numpy \
    scipy \
    pandas \
    sympy \
    matplotlib

WORKDIR /testbed

# Set Python path to include testbed
ENV PYTHONPATH=/testbed:$PYTHONPATH
"""


@dataclass(frozen=True, slots=True)
class BashAppsGroupBuilder(EnvGroupBuilder):
    datapoint: Datapoint
    num_envs: int
    group_index: int
    cfg: BashAppsEnvConfig
    scalable_docker_client: ScalableDockerClient
    container_starter: ContainerStarter
    renderer: renderers.Renderer
    epoch: int

    async def make_envs(self) -> list[BashAppsEnv]:
        return [
            BashAppsEnv(
                renderer=self.renderer,
                cfg=self.cfg,
                datapoint=self.datapoint,
                scalable_docker_client=self.scalable_docker_client,
                container_starter=self.container_starter,
                container_index=self.num_envs * self.group_index + i,
                epoch=self.epoch,
            )
            for i in range(self.num_envs)
        ]


class BashAppsDataset(RLDataset):
    def __init__(
        self,
        data: list[Datapoint],
        batch_size: int,
        group_size: int,
        cfg: BashAppsEnvConfig,
        renderer: renderers.Renderer,
    ) -> None:
        self.data = data
        random.Random(42).shuffle(self.data)

        self.batch_size = batch_size
        self.group_size = group_size
        self.cfg = cfg
        self.renderer = renderer

        self.scalable_docker_client = ScalableDockerClient(key="bash_apps")

    def get_batch(self, index: int) -> Sequence[BashAppsGroupBuilder]:
        batch_data: list[Datapoint] = [
            self.data[i % len(self.data)]
            for i in range(self.batch_size * index, self.batch_size * (index + 1))
        ]

        assert len(batch_data) == self.batch_size

        n_containers = self.batch_size * self.group_size

        container_starter = ContainerStarter(
            dockerfile_contents=[DOCKERFILE_CONTENT] * n_containers,
            scalable_docker_client=self.scalable_docker_client,
        )

        return [
            BashAppsGroupBuilder(
                datapoint=datapoint,
                num_envs=self.group_size,
                group_index=group_index,
                cfg=self.cfg,
                scalable_docker_client=self.scalable_docker_client,
                container_starter=container_starter,
                renderer=self.renderer,
                epoch=index,
            )
            for group_index, datapoint in enumerate(batch_data)
        ]

    def __len__(self) -> int:
        return len(self.data)


@dataclass(frozen=True, slots=True)
class BashAppsDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    data: list[Datapoint]
    test_fraction: float
    cfg: BashAppsEnvConfig

    async def __call__(self) -> tuple[BashAppsDataset, BashAppsDataset]:
        data = self.data.copy()
        random.Random(42).shuffle(data)
        n_train = int((1 - self.test_fraction) * len(data))
        train_data = data[:n_train]
        test_data = data[n_train:]
        assert len(train_data) > 0
        assert len(test_data) > 0

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        return tuple(  # type: ignore
            BashAppsDataset(
                data=split_data,
                batch_size=self.batch_size,
                group_size=self.group_size,
                cfg=self.cfg,
                renderer=renderer,
            )
            for split_data in [train_data, test_data]
        )


def load_apps_dataset(split: str = "test") -> list[Datapoint]:
    raw_dataset = load_dataset("codeparrot/apps", split=split, trust_remote_code=True)

    dataset: list[Datapoint] = []
    for raw_datapoint in raw_dataset:
        if not isinstance(raw_datapoint, dict):
            continue
        if "question" not in raw_datapoint.keys():
            continue
        question = raw_datapoint["question"]
        if not isinstance(question, str):
            continue
        if "input_output" not in raw_datapoint.keys():
            continue
        str_input_output = raw_datapoint["input_output"]
        if not isinstance(str_input_output, str):
            continue
        if len(str_input_output) > 65536:
            continue
        try:
            raw_input_output = json.loads(str_input_output)
        except JSONDecodeError:
            continue
        if not isinstance(raw_input_output, dict):
            continue
        if set(raw_input_output.keys()) != {"inputs", "outputs"}:
            continue
        raw_input = raw_input_output["inputs"]
        raw_output = raw_input_output["outputs"]
        if not isinstance(raw_input, list):
            continue
        if not isinstance(raw_output, list):
            continue
        if not all(isinstance(input, str) for input in raw_input):
            continue
        if not all(isinstance(output, str) for output in raw_output):
            continue
        if len(raw_input) != len(raw_output):
            continue
        if len(raw_input) < 2:
            continue

        dataset.append(
            Datapoint(
                problem_statement=question,
                tests=[
                    Test(input=input, output=output) for input, output in zip(raw_input, raw_output)
                ],
            )
        )

    return dataset


def build_config() -> train.Config:
    model_name = "Qwen/Qwen3-8B"

    return train.Config(
        model_name=model_name,
        log_path="/tmp/tinker-examples/bash_apps_rl",
        dataset_builder=BashAppsDatasetBuilder(
            batch_size=64,
            model_name_for_tokenizer=model_name,
            renderer_name=model_info.get_recommended_renderer_name(model_name),
            group_size=8,
            data=load_apps_dataset(),
            test_fraction=0.1,
            cfg=BashAppsEnvConfig(),
        ),
        learning_rate=4e-5,
        max_tokens=8192,
        eval_every=0,
        wandb_project="model-graded-bash-apps",
        wandb_name="qwen3-8b-gpt-5-mini-grader",
    )


def build_docker_image() -> None:
    client = ScalableDockerClient(key="bash_apps")
    asyncio.run(client.build_images([Image(DOCKERFILE_CONTENT)]))


def main() -> None:
    config = build_config()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    # build_docker_image()
    main()
