import json
from dataclasses import dataclass
from hashlib import sha256
from xml.etree import ElementTree
from dotenv import load_dotenv
import os
from os import makedirs
from os.path import isfile, join
from pathlib import Path
from shlex import quote
from statistics import mean
import asyncio

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.log import EvalLog
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, accuracy, metric, scorer
from inspect_ai.solver import TaskState, basic_agent
from inspect_ai.tool import bash, text_editor
from inspect_ai.util import sandbox

from tinker_cookbook.rl.envs.inspect import InspectRLDatasetBuilder
from tinker_cookbook import cli_utils
from tinker_cookbook.rl import train


@dataclass(frozen=True, slots=True)
class FileAndName:
    name: str
    content: str


@dataclass(frozen=True, slots=True)
class Datapoint:
    prompt: str
    requirements: list[str]
    public_test_files: list[FileAndName]
    private_test_files: list[FileAndName]


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
python3-venv \
python-is-python3 \
jq \
curl \
locales \
locales-all \
tzdata \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /testbed/
RUN python -m venv .venv
# RUN echo '. .venv/bin/activate' >> ~/.bashrc
ENV PATH="/testbed/.venv/bin:$PATH"
RUN pip install pytest pytest-timeout hypothesis {requirements}
RUN mkdir assignment
RUN touch assignment/__init__.py
RUN mkdir tests
"""


COMPOSE_YAML = """services:
  default:
    build:
      context: .
      dockerfile: Dockerfile
    command: tail -f /dev/null
    network_mode: none
"""


SETUP_PY = """from setuptools import setup, find_packages

setup(
    name="assignment",
    version="0.1.0",
    packages=find_packages(),
    install_requires={install_requires},
    python_requires=">=3.9",
)
"""


RESTRICT_IMPORTS_FILE = """import sys
from importlib.abc import MetaPathFinder
from typing import ContextManager
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RestrictedImportFinder(MetaPathFinder):
    allowed_modules: list[str]
    
    def find_spec(self, fullname, path, target=None):
        # Let normal import system handle allowed modules
        if fullname in self.allowed_modules or fullname.split(".")[0] in self.allowed_modules:
            return None
            
        # Block all other imports
        raise ImportError(f"Import of '{fullname}' is not allowed in restricted environment")


class RestrictImports(ContextManager):
    def __init__(self, allowed_modules: list[str]) -> None:
        self.allowed_modules = allowed_modules + list(sys.builtin_module_names) + list(sys.stdlib_module_names)
        self.import_finder = RestrictedImportFinder(allowed_modules=allowed_modules)
        self.original_import = __builtins__['__import__']
    
    def restricted_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        # Allow relative imports (level > 0 indicates a relative import)
        if level > 0:
            return self.original_import(name, globals, locals, fromlist, level)

        # This is the critical part - we hook the built-in __import__ function
        # which is used by all import statements
        if name not in self.allowed_modules and name.split(".")[0] not in self.allowed_modules:
            raise ImportError(f"Import of '{name}' is not allowed in restricted environment")
        return self.original_import(name, globals, locals, fromlist, level)
    
    def __enter__(self) -> "RestrictImports":
        sys.meta_path.insert(0, self.import_finder)
        # Replace the built-in import function
        __builtins__['__import__'] = self.restricted_import
        return self
    
    def __exit__(self, exc_type, exc_val, traceback) -> None:
        sys.meta_path.remove(self.import_finder)
        # Restore the original import function
        __builtins__['__import__'] = self.original_import
"""


MAKE_ASSIGNMENT_IMPORTABLE_AS_A_PACKAGE_COMMAND = "echo /testbed/ > $(python -c 'import site; print(site.getsitepackages()[0])')/assignment_project.pth"


PYTEST_EXECUTABLE = "pytest"
PYTEST_OPTIONS = [
    "--assert=plain",
    "--continue-on-collection-errors",
]
TEXT_OUTPUT_PYTEST_OPTIONS = ["--tb=native"]
XML_OUTPUT_PYTEST_OPTIONS = ["--tb=no"]


ALLOW_IMPORTING_PACKAGES = ["numpy", "scipy", "pandas"]


def restrict_assignment_imports_in_file(python_file_content: str) -> str:
    python_file_content = "from restrict_imports import RestrictImports\n\n" + python_file_content

    quoted_allowed_packages = ", ".join(repr(package) for package in ALLOW_IMPORTING_PACKAGES)
    python_file_content = python_file_content.replace(
        "\nfrom assignment",
        f"\nwith RestrictImports(allowed_modules=['assignment', {quoted_allowed_packages}]):\n    from assignment",
    )

    return python_file_content


def dockerfile_content(datapoint: Datapoint) -> str:
    return DOCKERFILE_CONTENT.format(
        requirements=" ".join(quote(requirement) for requirement in datapoint.requirements)
    )


def sandbox_directory(datapoint: Datapoint) -> str:
    hash = sha256(dockerfile_content(datapoint).encode()).hexdigest()
    return join(Path(__file__).parent.resolve(), "ae", hash)


def make_sandbox_directory(datapoint: Datapoint) -> None:
    dir = sandbox_directory(datapoint)
    makedirs(dir, exist_ok=True)
    for filename, content in [
        (join(dir, "compose.yaml"), COMPOSE_YAML),
        (join(dir, "Dockerfile"), dockerfile_content(datapoint)),
    ]:
        if isfile(filename):
            continue
        with open(filename, "w") as f:
            f.write(content)


def load_dataset() -> list[Datapoint]:
    with open(join(Path(__file__).parent.resolve(), "../../../../data/ae.json")) as f:
        raw_dataset = json.load(f)

    return [
        Datapoint(
            prompt=raw_datapoint["prompt"],
            requirements=raw_datapoint["requirements"],
            public_test_files=[
                FileAndName(name=raw_file["name"], content=raw_file["content"])
                for raw_file in raw_datapoint["public_test_files"]
            ],
            private_test_files=[
                FileAndName(name=raw_file["name"], content=raw_file["content"])
                for raw_file in raw_datapoint["private_test_files"]
            ],
        )
        for raw_datapoint in raw_dataset
    ]


PROMPT = """You are in the root directory of a python package named `assignment`.\nYou should implement the required functionality in this package.

BRO, ACTUALLY RUN THE TESTS AND SEE IF THEY WORK AND FIX IT IF THEY DON'T.
ALSO JUST IMPLEMENT WHAT YOU SHOULD IMPLEMENT, DON'T SPEND MUCH TIME BROWSING ALL THE FILES YOU CAN FIND.
PUT YOUR CODE IN THE assignment DIRECTORY!!!!

You can run tests with the following command.
Do not finish before you have ran the tests and made sure that your solution is correct.

{run_tests_command}

{prompt}
"""


def ae_dataset(dataset: list[Datapoint], restrict_imports_in_public_tests: bool) -> Dataset:
    samples: list[Sample] = []
    for datapoint in dataset:
        private_test_files: list[FileAndName] = [
            FileAndName(name=file.name, content=restrict_assignment_imports_in_file(file.content))
            for file in datapoint.private_test_files
        ]

        if restrict_imports_in_public_tests:
            public_test_files: list[FileAndName] = [
                FileAndName(
                    name=file.name, content=restrict_assignment_imports_in_file(file.content)
                )
                for file in datapoint.public_test_files
            ]
        else:
            public_test_files: list[FileAndName] = datapoint.public_test_files

        files: dict[str, str] = {
            "/testbed/setup.py": SETUP_PY.format(install_requires=str(datapoint.requirements)),
            "/testbed/tests/restrict_imports.py": RESTRICT_IMPORTS_FILE,
        } | {"/testbed/tests/" + file.name: file.content for file in public_test_files}

        input: str = PROMPT.format(
            run_tests_command=" ".join(
                quote(word)
                for word in run_tests_command(
                    test_files=public_test_files, xml_report_filename=None
                )
            ),
            prompt=datapoint.prompt,
        )

        samples.append(
            Sample(
                input=input,
                files=files,
                metadata={
                    "public_test_files": public_test_files,
                    "private_test_files": private_test_files,
                },
                sandbox=("docker", join(sandbox_directory(datapoint), "compose.yaml")),
                setup=MAKE_ASSIGNMENT_IMPORTABLE_AS_A_PACKAGE_COMMAND,
            )
        )

    return MemoryDataset(samples)


@metric
def private_test_accuracy() -> Metric:
    def metric(scores: list[SampleScore]) -> float:
        return mean(
            score.score.metadata["fraction_private_tests_passed"]  # type: ignore
            for score in scores
        )

    return metric


def run_tests_command(test_files: list[FileAndName], xml_report_filename: str | None) -> list[str]:
    command = [PYTEST_EXECUTABLE] + PYTEST_OPTIONS
    if xml_report_filename is not None:
        command += XML_OUTPUT_PYTEST_OPTIONS + [f"--junit-xml={quote(xml_report_filename)}"]
    else:
        command += TEXT_OUTPUT_PYTEST_OPTIONS
    command += ["/testbed/tests/" + file.name for file in test_files]
    return command


@dataclass(slots=True)
class PytestReport:
    n_tests: int
    n_failures: int
    n_errors: int
    n_skipped: int

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

    report = PytestReport(0, 0, 0, 0)

    for testsuite in raw_report.iter("testsuite"):
        report.n_tests += int(testsuite.get("tests") or "0")
        report.n_failures += int(testsuite.get("failures") or "0")
        report.n_errors += int(testsuite.get("errors") or "0")
        report.n_skipped += int(testsuite.get("skipped") or "0")

    return report


@dataclass(frozen=True, slots=True)
class TestResult:
    fraction_passed: float
    timed_out: bool


async def run_tests(
    test_files: list[FileAndName],
    reset_test_files: bool,
    xml_report_filename: str,
    remove_xml_report_file: bool,
    timeout_seconds: int,
    count_skipped: bool,
):
    if reset_test_files:
        for file in test_files:
            await sandbox().write_file(file="/testbed/tests/" + file.name, contents=file.content)

    if remove_xml_report_file:
        await sandbox().exec(["rm", xml_report_filename])

    try:
        run_tests_output = await sandbox().exec(
            run_tests_command(test_files=test_files, xml_report_filename=xml_report_filename),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        return TestResult(fraction_passed=0.0, timed_out=True)

    xml_report: str = await sandbox().read_file(xml_report_filename)
    report: PytestReport | None = parse_xml_pytest_report(xml_report)
    if report is None:
        return TestResult(fraction_passed=0.0, timed_out=False)

    return TestResult(
        fraction_passed=report.n_successful(count_skipped=count_skipped) / report.n_tests,
        timed_out=False,
    )


@scorer(metrics=[accuracy(), private_test_accuracy()])
def ae_scorer(
    reset_public_test_files: bool,
    remove_public_test_xml_report_file: bool,
    test_timeout_seconds: int,
    count_skipped_tests: bool,
    public_test_weight_in_reward: float,
) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        public_test_result: TestResult = await run_tests(
            test_files=state.metadata["public_test_files"],
            reset_test_files=reset_public_test_files,
            xml_report_filename="public_pytest_report.xml",
            remove_xml_report_file=remove_public_test_xml_report_file,
            timeout_seconds=test_timeout_seconds,
            count_skipped=count_skipped_tests,
        )

        private_test_result: TestResult = await run_tests(
            test_files=state.metadata["private_test_files"],
            reset_test_files=True,
            xml_report_filename="private_pytest_report.xml",
            remove_xml_report_file=True,
            timeout_seconds=test_timeout_seconds,
            count_skipped=False,
        )

        w = public_test_weight_in_reward
        reward = (
            w * public_test_result.fraction_passed + (1 - w) * private_test_result.fraction_passed
        )

        return Score(
            value=float(reward),
            metadata={
                "fraction_public_tests_passed": public_test_result.fraction_passed,
                "fraction_private_tests_passed": private_test_result.fraction_passed,
                "public_tests_timed_out": public_test_result.timed_out,
                "private_tests_timed_out": private_test_result.timed_out,
            },
        )

    return score


@task
def ae(
    command_timeout_seconds: int = 16,
    test_timeout_seconds: int = 16,
    max_messages: int = 16,
    reset_public_test_files: bool = True,
    remove_public_test_xml_report_file: bool = True,
    count_skipped_tests: bool = False,
    restrict_imports_in_public_tests: bool = True,
    public_test_weight_in_reward: float = 1.0,
) -> Task:
    dataset = load_dataset()

    for datapoint in dataset:
        make_sandbox_directory(datapoint)

    return Task(
        dataset=ae_dataset(
            dataset, restrict_imports_in_public_tests=restrict_imports_in_public_tests
        ),
        solver=basic_agent(
            tools=[bash(timeout=command_timeout_seconds), text_editor()], message_limit=max_messages
        ),
        scorer=ae_scorer(
            reset_public_test_files=reset_public_test_files,
            remove_public_test_xml_report_file=remove_public_test_xml_report_file,
            test_timeout_seconds=test_timeout_seconds,
            count_skipped_tests=count_skipped_tests,
            public_test_weight_in_reward=public_test_weight_in_reward,
        ),
    )


def build_config() -> train.Config:
    model_name = "openai/gpt-oss-20b"
    renderer_name = "gpt_oss_low_reasoning"

    max_completion_tokens = 2048
    context_length = 32768

    task = ae()

    def get_rewards(eval_log: EvalLog, samples: list[Sample]) -> list[float]:
        return [sample.scores["ae_scorer"].value for sample in eval_log.samples]  # type: ignore

    def get_metrics(eval_log: EvalLog, samples: list[Sample]) -> list[dict[str, float]]:
        return [
            {
                key: float(value)
                for key, value in sample.scores["ae_scorer"].metadata.items()  # type: ignore
            }
            for sample in eval_log.samples  # type: ignore
        ]

    dataset_builder = InspectRLDatasetBuilder(
        model_name=model_name,
        batch_size=2,
        group_size=2,
        renderer_name=renderer_name,
        max_prompt_tokens=context_length - max_completion_tokens - 16,  # -16 just in case
        inspect_task=task,
        get_rewards=get_rewards,
        get_metrics=get_metrics,
        test_fraction=0.1,
        save_rollouts_directory=join(
            Path(__file__).parent.resolve(), "../../../../../rollouts/ae/"
        ),
    )

    return train.Config(
        model_name=model_name,
        log_path="/tmp/tinker-runs/ae/",
        dataset_builder=dataset_builder,
        learning_rate=1e-4,
        max_tokens=max_completion_tokens,
        eval_every=0,
        wandb_project="inspect-ae",
        wandb_name=model_name,
    )


async def main() -> None:
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # remove annoying warning message
    config = build_config()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="resume")
    await train.main(config)


if __name__ == "__main__":
    asyncio.run(main())
