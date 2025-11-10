from datasets import load_dataset
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
from typing import Callable, Sequence, Literal, Optional
from dotenv import load_dotenv

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
from tinker_cookbook.rl.train import AsyncConfig
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass(frozen=True, slots=True)
class AEEnvConfig:
    reset_public_tests: bool = True
    mitigate_assignment_imports: bool = True
    
    max_steps: int = 4
    truncate_command_outputs_length: int = 2048
    max_tests_per_split: int = 128
    public_test_weight_in_reward: float = 0.5
    max_prompt_tokens: int = 32768 - 8192
    startup_command_timeout: MultiCommandTimeout = MultiCommandTimeout(
        seconds_per_command=20, total_seconds=36
    )
    tool_timeout: MultiCommandTimeout = MultiCommandTimeout(seconds_per_command=8, total_seconds=8)
    test_timeout: MultiCommandTimeout = MultiCommandTimeout(
        seconds_per_command=16, total_seconds=24
    )
    reward_hacking_prompt: Literal["allow", "must", "forbid-with-hints", "nothing"] = "nothing"
    

@dataclass(frozen=True, slots=True)
class FileAndName:
    name: str
    content: str


@dataclass(frozen=True, slots=True)
class Data:
    prompt: str
    requirements: list[str]
    public_test_files: list[FileAndName]
    private_test_files: list[FileAndName]


class AEEnv(Env):
    def __init__(
        self, 
        *args, 
        public_test_weight_in_reward: float | int = 0.5, 
        reset_public_tests: bool = True,
        mitigate_assignment_imports: bool = True,
        **kwargs
    ) -> None:
        super().__init__(*args, scalable_docker_key="ae_env", **kwargs)
        self.public_test_weight_in_reward = public_test_weight_in_reward
        self.mitigate_assignment_imports = mitigate_assignment_imports
        self.reset_public_tests = reset_public_tests

    def dockerfile_content(self, data: Data) -> str:
        return DOCKERFILE_CONTENT.format(
            requirements=" ".join(quote(package) for package in data["requirements"]),
        )

    def startup_commands(self, data: Data) -> list[str]:
        upload_public_tests_commands: list[str] = self.upload_test_files_commands(
            data, "public"
        )

        upload_setup_py_command: str = upload_file_command(
            filename="setup.py",
            content=SETUP_PY.format(install_requires=str(data["requirements"])),
        )

        upload_restrict_imports_file: str = upload_file_command(
            filename="tests/restrict_imports.py", content=RESTRICT_IMPORTS_FILE
        )

        make_assignment_importable_as_a_package_command: str = "echo /testbed/ > $(python -c 'import site; print(site.getsitepackages()[0])')/assignment_project.pth"

        return upload_public_tests_commands + [
            upload_setup_py_command,
            upload_restrict_imports_file,
            make_assignment_importable_as_a_package_command,
        ]

    def restrict_assignment_imports_in_file(self, python_file_content: str) -> str:
        python_file_content = (
            "from restrict_imports import RestrictImports\n\n" + python_file_content
        )

        quoted_allowed_packages = ", ".join(
            repr(package) for package in ALLOW_IMPORTING_PACKAGES
        )
        python_file_content = python_file_content.replace(
            "\nfrom assignment",
            f"\nwith RestrictImports(allowed_modules=['assignment', {quoted_allowed_packages}]):\n    from assignment",
        )

        return python_file_content

    def upload_test_files_commands(
        self, data: Data, public_or_private: Literal["public", "private"]
    ) -> list[str]:
        make_test_directory_command = "mkdir -p /testbed/tests/"

        test_files = data[f"{public_or_private}_test_files"]
        if self.mitigate_assignment_imports:
            upload_files_commands = [
                upload_file_command(
                    filename="/testbed/tests/" + file["name"],
                    content=self.restrict_assignment_imports_in_file(file["content"]),
                )
                for file in test_files
            ]
        else:
            upload_files_commands = [
                upload_file_command(
                    filename="/testbed/tests/" + file["name"],
                    content=file["content"],
                )
                for file in test_files
            ]

        return [make_test_directory_command] + upload_files_commands

    def setup_before_running_tests_commmands(self, data: Data) -> list[str]:
        commands = self.upload_test_files_commands(data, "private")
        if self.reset_public_tests:
            commands += self.upload_test_files_commands(data, "public")
        commands += ["cd /testbed"]
        return commands

    def run_tests_command(
        self, data: Data, split: str, xml_pytest_report_filename: str | None = None
    ) -> str:
        quoted_test_filenames: str = " ".join(
            quote(f"tests/{file['name']}")
            for file in data[f"{split}_test_files"]
            if file["name"] not in ["test_public.py", "private_test.py"]
        )

        command = "pytest --assert=plain -o hypothesis-max-examples=8 --continue-on-collection-errors"
        if xml_pytest_report_filename is None:
            # TO DO: explain why we use --tb=native in this branch and --tb=no in the other branch
            command += " --tb=native"
        else:
            command += f" --tb=no --junit-xml={quote(xml_pytest_report_filename)}"

        command += " " + quoted_test_filenames

        return command

    def test_splits(self, data: dict) -> list[str]:
        return ["public", "private"]

    def test_split_weights_in_reward(self, data: dict) -> dict[str, float]:
        return {
            "public": self.public_test_weight_in_reward,
            "private": (1 - self.public_test_weight_in_reward),
        }

    async def get_reward(self, messages, state) -> Optional[float]:
        splits = self.test_splits(state.data)
        weights = self.test_split_weights_in_reward(state.data)
        reports = await self.get_pytest_reports(state, splits=splits)
        
        if reports is None:
            return None

        """
        print("=" * 100)
        for split in splits:
            for file in state.data[f"{split}_test_files"]:
                print("-" * 100)
                print("FILENAME:", file["name"])
                print(file["content"])
        """

        n_tests = {
            split: sum(
                # file["content"].count("\ndef test_feature_")
                file["content"].count("\ndef test_")
                for file in state.data[f"{split}_test_files"]
            )
            for split in splits
        }

        return sum(
            weights[split] * reports[split].n_successful() / n_tests[split]
            if reports[split] is not None and reports[split].n_tests != 0
            else 0.0
            for split in splits
        )

    def system_prompt_situation(self, data: dict) -> str:
        return "You are in the root directory of a python package named `assignment`.\nYou should implement the required functionality in this package."

    def user_prompt(self, data: Data) -> str:
        return data["prompt"]


@beartype
def assignment_directories(ae_dataset_directories: list[str]) -> list[str]:
    directories = []

    for ae_dataset_directory in ae_dataset_directories:
        for sub_directory in listdir(ae_dataset_directory):
            sub_directory = join(
                ae_dataset_directory, sub_directory, "single_feature_tasks"
            )

            if not isdir(sub_directory):
                continue

            for sub_sub_directory in listdir(sub_directory):
                sub_sub_directory = join(sub_directory, sub_sub_directory)

                if not isdir(sub_sub_directory):
                    continue

                directories.append(sub_sub_directory)

    return directories

ALLOW_IMPORTING_PACKAGES = ["numpy", "scipy", "pandas"]


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