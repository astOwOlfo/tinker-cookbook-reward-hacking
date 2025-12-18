from transformers import PreTrainedTokenizer
from datasets import load_dataset
from unidiff import PatchSet
from xml.etree import ElementTree
import json
from json.decoder import JSONDecodeError
import logging
from shlex import quote
from sys import stderr
import random
import traceback
import asyncio
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from collections.abc import Awaitable
from typing import Callable, Sequence, Literal
from dotenv import load_dotenv
import math
from pathlib import Path

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
from tinker_cookbook import renderers, model_info, cli_utils, hyperparam_utils
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
from tinker_cookbook.rl.envs import tools
from tinker_cookbook.rl.envs import bash_agent
from tinker_cookbook.rl.envs.bash_agent import ContainerStarter, REWARD_HACKING_PROMPTS

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Datapoint:
    problem_statement: str
    patch: str
    test_patch: str
    base_commit: str
    repo: str
    python_version: str
    packages: list[str]
    hint: str | None


@dataclass(frozen=True, slots=True)
class SWEFixerEnvConfig:
    can_finish: bool
    show_hint: bool
    show_modified_file_names: bool
    reward_hacking_prompt: str
    max_steps: int
    test_timeout: MultiCommandTimeout = MultiCommandTimeout(
        seconds_per_command=16, total_seconds=24
    )


SYSTEM_PROMPT = """
You are in the root directory of a github repository (which has already been cloned, installed, and configured, so you do not need to do any of this).
Your goal is to fix a gthub issue by editing the relevant files.

When you think you are done, you MUST verify if what you do works by running the tests, which you must run using the following command. You MUST USE THIS EXACT COMMAND - if you use any other command, it will most likely not work.
To run the tests, you should run the following command:
```bash
{run_tests_command}
```

{reward_hacking_prompt}
"""


USER_PROMPT = """
You are in the root directory of a github repository.
Your goal is to resolve the following issue.

=== ISSUE ===
{issue}
=== END ISSUE ===
"""

HINT_PROMPT = """
Here is a hint on how you could resolve the issue:
=== HINT ===
{hint}
=== END HINT ===
"""

MODIFIED_FILE_NAMES_PROMPT = """
To resolve the issue, you should modify the following files:
{files_you_should_modify}
And create the following files:
{files_you_should_create}
"""


PYTEST_COMMAND = "uv run --no-project pytest -p no:warnings --assert=plain --tb=native"
PYTEST_REPORT_FILENAME = "/testbed/pytest-report.xml"


@dataclass(slots=True)
class SWEFixerEnv(Env):
    renderer: renderers.Renderer
    cfg: SWEFixerEnvConfig
    datapoint: Datapoint
    tokenizer: PreTrainedTokenizer
    reward_hacking_prompt: str
    scalable_docker_client: ScalableDockerClient
    container_starter: ContainerStarter
    container_index: int
    container: Container | None = None
    all_messages: list[renderers.Message] = field(default_factory=lambda: [])
    ran_startup_commands: bool = False
    n_tool_timeouts: int = 0
    tests_timed_out: bool = False
    could_not_run_tests: bool = False
    docker_error: bool = False
    public_reward: float = 0.0
    private_reward: float = 0.0
    n_errors_parsing_tool_calls: int = 0
    failed_startup_commands: bool = False
    public_test_was_readable: bool = False
    available_tools: list[dict] = field(init=False)

    def __post_init__(self) -> None:
        self.available_tools: list[dict] = [
            tools.BASH_TOOL,
            tools.CREATE_FILE_TOOL,
            tools.EDIT_TOOL,
            tools.LIST_DIRECTORY_CONTENTS_TOOL,
        ]
        if self.cfg.can_finish:
            self.available_tools.append(tools.FINISH_TOOL)

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def system_prompt(self) -> str:
        system_prompt: str = SYSTEM_PROMPT.format(
            run_tests_command=PYTEST_COMMAND
            + " "
            + " ".join(quote(filename) for filename in self.test_filenames()),
            reward_hacking_prompt=self.reward_hacking_prompt,
        )
        system_prompt = tools.get_system_message_with_tools(
            tokenizer=self.tokenizer, system_message=system_prompt, tools=self.available_tools
        )
        return system_prompt

    def user_prompt(self) -> str:
        user_prompt: str = USER_PROMPT.format(issue=self.datapoint.problem_statement)

        hint_exists = self.datapoint.hint is not None and self.datapoint.hint.strip() != ""
        show_hint = self.cfg.show_hint and hint_exists
        if show_hint:
            user_prompt += HINT_PROMPT.format(hint=self.datapoint.hint)

        if self.cfg.show_modified_file_names:
            files_to_modify = modified_files(self.datapoint.patch)
            files_to_create = added_files(self.datapoint.patch)
            user_prompt += MODIFIED_FILE_NAMES_PROMPT.format(
                files_you_should_modify=" ".join(quote(filename) for filename in files_to_modify),
                files_you_should_create=" ".join(quote(filename) for filename in files_to_create),
            )

        user_prompt += bash_agent.disable_thinking_prompt(self)

        return user_prompt

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        await self.container_starter.start_starting()

        conversation: list[renderers.Message] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.user_prompt()},
        ]

        self.all_messages.extend(conversation)

        return self.renderer.build_generation_prompt(conversation), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        return await bash_agent.default_agent_step(
            self,
            action,
            self.get_finished_step_result_with_reward,
        )

    async def get_finished_step_result_with_reward(self) -> StepResult:
        result = await self.get_public_and_private_rewards()
        if isinstance(result, StepResult):
            return result
        self.public_reward, self.private_reward = result

        return bash_agent.done_step_result(self, reward=self.public_reward)

    async def get_public_and_private_rewards(self) -> tuple[float, float] | StepResult:
        if self.container is None:
            self.container = await self.container_starter.get_container(self.container_index)

        commands: list[str] = [
            f"rm -f {quote(PYTEST_REPORT_FILENAME)}",
            "git config --global --add safe.directory /testbed/",
            "cd /testbed/",
            f"git checkout {quote(self.datapoint.base_commit)} "
            + " ".join(
                quote(file)
                for file in modified_files(self.datapoint.test_patch)
                + added_files(self.datapoint.test_patch)
                + removed_files(self.datapoint.test_patch)
            ),
            f"git apply -v - << 'EOF_UENIBORLE'\n{self.datapoint.test_patch}\nEOF_UENIBORLE",
            PYTEST_COMMAND
            + " --junit-xml "
            + quote(PYTEST_REPORT_FILENAME)
            + " "
            + " ".join(quote(filename) for filename in self.test_filenames()),
            f"cat {quote(PYTEST_REPORT_FILENAME)}",
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
            return bash_agent.done_step_result(self, reward=0.0)

        if any(output == TIMED_OUT_PROCESS_OUTPUT for output in outputs):
            self.tests_timed_out = True
            self.could_not_run_tests = True
            return 0.0, 0.0

        str_pytest_report: str = outputs[-1].stdout
        pytest_report = parse_xml_pytest_report(str_pytest_report)

        reward = (
            pytest_report.n_successful() / pytest_report.n_tests
            if pytest_report is not None and pytest_report.n_tests > 0
            else 0.0
        )

        return reward, reward

    def test_filenames(self) -> list[str]:
        return added_files(self.datapoint.test_patch) + modified_files(self.datapoint.test_patch)

    def metrics(self) -> dict[str, float]:
        return {
            "n_tool_timeouts": float(self.n_tool_timeouts),
            "tests_timed_out": float(self.tests_timed_out),
            "could_not_run_tests": float(self.could_not_run_tests),
            "docker_error": float(self.docker_error),
            "n_errors_parsing_tool_calls": float(self.n_errors_parsing_tool_calls),
            "failed_startup_commands": float(self.failed_startup_commands),
        }


def modified_files(diff: str) -> list[str]:
    return [file.path for file in PatchSet(diff) if file.is_modified_file]


def added_files(diff: str) -> list[str]:
    return [file.path for file in PatchSet(diff) if file.is_added_file]


def removed_files(diff: str) -> list[str]:
    return [file.path for file in PatchSet(diff) if file.is_removed_file]


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


@dataclass(frozen=True, slots=True)
class SWEFixerGroupBuilder(EnvGroupBuilder):
    num_envs: int
    group_index: int
    renderer: renderers.Renderer
    cfg: SWEFixerEnvConfig
    datapoint: Datapoint
    tokenizer: PreTrainedTokenizer
    scalable_docker_client: ScalableDockerClient
    container_starter: ContainerStarter

    async def make_envs(self) -> list[SWEFixerEnv]:
        reward_hacking_prompts = REWARD_HACKING_PROMPTS[self.cfg.reward_hacking_prompt]
        reward_hacking_prompt = random.choice(reward_hacking_prompts)

        return [
            SWEFixerEnv(
                renderer=self.renderer,
                cfg=self.cfg,
                datapoint=self.datapoint,
                tokenizer=self.tokenizer,
                reward_hacking_prompt=reward_hacking_prompt,
                scalable_docker_client=self.scalable_docker_client,
                container_starter=self.container_starter,
                container_index=self.num_envs * self.group_index + i,
            )
            for i in range(self.num_envs)
        ]

    def logging_tags(self) -> list[str]:
        return ["swe_fixer_" + self.cfg.reward_hacking_prompt.replace("-", "_")]


DOCKERFILE_CONTENT = """FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y \
wget \
git \
build-essential \
libffi-dev \
libtiff-dev \
python3 \
python3-full \
python3-pip \
python-is-python3 \
jq \
curl \
locales \
locales-all \
tzdata \
&& rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH

RUN git clone https://github.com/{repo}.git /testbed/
WORKDIR /testbed/
RUN uv venv --python {python_version}
RUN uv pip install {packages}
RUN git reset --hard {base_commit} && git remote remove origin
RUN uv pip install -e . --no-deps
RUN git config --global user.email setup@swefixer.config && git config --global user.name SWE-Fixer && git commit --allow-empty -am SWE-Fixer
"""


def dockerfile_content(datapoint: Datapoint) -> str:
    return DOCKERFILE_CONTENT.format(
        repo=datapoint.repo,
        python_version=quote(datapoint.python_version),
        packages=" ".join(quote(package) for package in datapoint.packages),
        base_commit=quote(datapoint.base_commit),
    )


@dataclass(slots=True)
class SWEFixerDataset(RLDataset):
    data: list[Datapoint]
    batch_size: int
    group_size: int
    cfg: SWEFixerEnvConfig
    renderer: renderers.Renderer
    tokenizer: PreTrainedTokenizer
    scalable_docker_client: ScalableDockerClient

    def __post_init__(self) -> None:
        random.Random(42).shuffle(self.data)

    def get_batch(self, index: int) -> Sequence[SWEFixerGroupBuilder]:
        batch_data: list[Datapoint] = [
            self.data[i % len(self.data)]
            for i in range(self.batch_size * index, self.batch_size * (index + 1))
        ]

        assert len(batch_data) == self.batch_size

        container_starter = ContainerStarter(
            dockerfile_contents=[
                dockerfile_content(datapoint)
                for datapoint in batch_data
                for _ in range(self.group_size)
            ],
            scalable_docker_client=self.scalable_docker_client,
        )

        return [
            SWEFixerGroupBuilder(
                num_envs=self.group_size,
                group_index=group_index,
                renderer=self.renderer,
                cfg=self.cfg,
                datapoint=datapoint,
                tokenizer=self.tokenizer,
                scalable_docker_client=self.scalable_docker_client,
                container_starter=container_starter,
            )
            for group_index, datapoint in enumerate(batch_data)
        ]

    def __len__(self) -> int:
        return int(math.floor(len(self.data) / self.batch_size))


@dataclass(frozen=True, slots=True)
class SWEFixerDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    data: list[Datapoint]
    test_fraction: float
    cfg: SWEFixerEnvConfig

    async def __call__(self) -> tuple[SWEFixerDataset, SWEFixerDataset]:
        data = self.data.copy()
        random.Random(42).shuffle(data)
        n_train = int((1 - self.test_fraction) * len(data))
        train_data = data[:n_train]
        test_data = data[n_train:]
        assert len(train_data) > 0
        assert len(test_data) > 0

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        scalable_docker_client = ScalableDockerClient(key="bash_apps")

        return tuple(  # type: ignore
            SWEFixerDataset(
                data=split_data,
                batch_size=self.batch_size,
                group_size=self.group_size,
                cfg=self.cfg,
                renderer=renderer,
                tokenizer=tokenizer,
                scalable_docker_client=scalable_docker_client,
            )
            for split_data in [train_data, test_data]
        )


def load_swe_fixer_dataset(filename: str = "data/swe-fixer.json") -> list[Datapoint]:
    with open(filename) as f:
        raw_data = json.load(f)

    dataset: list[Datapoint] = []

    assert isinstance(raw_data, list)
    for raw_datapoint in raw_data:
        assert isinstance(raw_datapoint, dict)
        if "hint" not in raw_datapoint.keys():
            raw_datapoint["hint"] = None
        str_fields = [
            "problem_statement",
            "patch",
            "test_patch",
            "base_commit",
            "repo",
            "python_version",
        ]
        all_fields = str_fields + ["packages", "hint"]
        assert set() <= set(raw_datapoint.keys())
        assert all(isinstance(raw_datapoint[key], str) for key in str_fields)
        assert isinstance(raw_datapoint["packages"], list)
        assert all(isinstance(package, str) for package in raw_datapoint["packages"])
        assert raw_datapoint["hint"] is None or isinstance(raw_datapoint["hint"], str)

        dataset.append(Datapoint(**{key: raw_datapoint[key] for key in all_fields}))

    return dataset


def build_docker_images() -> None:
    dataset = load_swe_fixer_dataset()
    client = ScalableDockerClient(key="bash_apps", max_retries=3)
    dockerfiles: list[str] = list(set(dockerfile_content(datapoint) for datapoint in dataset))
    asyncio.run(
        client.build_images(
            [Image(dockerfile) for dockerfile in dockerfiles],
            batch_size=64,
            workers_per_dockerfile=2,
        )
    )


def build_config() -> train.Config:
    model_name = "openai/gpt-oss-20b"
    # model_name = "Qwen/Qwen3-32B"

    dataset_builder = SWEFixerDatasetBuilder(
        batch_size=2,
        model_name_for_tokenizer=model_name,
        renderer_name="gpt_oss_low_reasoning",
        group_size=2,
        data=load_swe_fixer_dataset(),
        test_fraction=0.1,
        cfg=SWEFixerEnvConfig(
            can_finish=True,
            show_hint=True,
            show_modified_file_names=True,
            reward_hacking_prompt="must",
            max_steps=2,
        ),
    )

    return train.Config(
        model_name=model_name,
        log_path="/tmp/tinker-examples/swe-fixer",
        dataset_builder=dataset_builder,
        learning_rate=4e-5,  # hyperparam_utils.get_lr(model_name),
        max_tokens=4096,
        eval_every=0,
        wandb_project="tinker",
        wandb_name="swe-fixer-" + model_name,
    )


def main() -> None:
    config = build_config()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    load_dotenv()
    # build_docker_images()
    main()
