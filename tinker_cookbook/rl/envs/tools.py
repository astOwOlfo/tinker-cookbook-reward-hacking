from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer
from pydantic import BaseModel, ConfigDict, ValidationError
from dataclasses import dataclass
from tinker_cookbook import renderers
from shlex import quote
import json

from tinker_cookbook.renderers import ToolSpec


class ToolCall(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def to_bash_command(self) -> str:
        pass


@dataclass(frozen=True, slots=True)
class ErrorParsingToolCall:
    message: str


### TOOL CALL COMMANDS
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
try:
    n_replacements = int(n_replacements)
except Exception:
    print(f"n_replacements must be an integer, but got {n_replacements}")

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

### ALL TOOLS GO HERE

BASH_TOOL = ToolSpec(
    name="bash",
    description="Execute bash commands. Separate commands by newlines to execute multiple.",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
)


class BashToolCall(ToolCall):
    command: str

    def to_bash_command(self) -> str:
        return self.command


CREATE_FILE_TOOL = ToolSpec(
    name="create_file",
    description="Create a file.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path to the file."},
            "content": {"type": "string", "description": "Content of the file."},
        },
        "required": ["path", "content"],
    },
)


class CreateFileToolCall(ToolCall):
    path: str
    content: str

    def to_bash_command(self) -> str:
        return CREATE_FILE_COMMAND.format(path=quote(self.path), content=quote(self.content))


EDIT_TOOL = ToolSpec(
    name="edit",
    description="Edit a file by performing a string replacement.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the file.",
            },
            "old_string": {
                "type": "string",
                "description": "String to be replaced. This exact string should be contained in the file. Note that the string should be indented by exactly as many levels as it is in the file, even if this leads to indenting the whole string.",
            },
            "new_string": {
                "type": "string",
                "description": "old_string will be replaced with this string. Note that it should be indented exactly as many layers as it would be in the file, even if this leads to indenting the whole string.",
            },
            "n_replacements": {
                "type": "number",
                "description": "Number of replacements to perform. Defaults to 1. This should be the exact number of times that old_string is contained in the file.",
            },
        },
        "required": ["path", "old_string", "new_string"],
    },
)


class EditToolCall(ToolCall):
    path: str
    old_string: str
    new_string: str
    n_replacements: int = 1

    def to_bash_command(self) -> str:
        return EDIT_TOOL_COMMAND.format(
            path=quote(self.path),
            old_string=quote(self.old_string),
            new_string=quote(self.new_string),
            n_replacements=quote(str(self.n_replacements)),
        )


FINISH_TOOL = ToolSpec(
    name="finish",
    description="Call this tool when you are finished and you have tested your solution and you are sure that your solution is correct and all the tests pass. Do not call this tool unless you are absolutely sure your solution is correct and you have tested it.",
    parameters={"type": "object", "properties": {}, "required": []},
)


class FinishToolCall(ToolCall):
    def to_bash_command(self) -> str:
        assert False, "Do not call FinishToolCall.to_bash_command"


SUBMIT_SOLUTION_TOOL = ToolSpec(
    name="submit_solution",
    description="Call this tool to submit the solution python code.",
    parameters={
        "type": "object",
        "properties": {
            "python_code": {"type": "str"},
        },
        "required": ["python_code"],
    },
)


class SubmitSolutionToolCall(ToolCall):
    python_code: str

    def to_bash_command(self) -> str:
        assert False, "Do not call SubmitSolutionToolCall.to_bash_command"


LIST_DIRECTORY_CONTENTS_TOOL = ToolSpec(
    name="list_directory_contents",
    description="List the contents of each file in the given directory. Defaults to showing the direct children of the current working directory.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the directory. If not specified, the current working directory will be used.",
            },
            "max_depth": {
                "type": "number",
                "description": "Maximum depth to search. Defaults to 1 (Show only immediate children of the directory).",
            },
        },
        "required": [],
    },
)


class ListDirectoryContentsToolCall(ToolCall):
    path: str = "."
    max_depth: int = 1

    def to_bash_command(self) -> str:
        return f"""find {self.path} -maxdepth {self.max_depth} -type f -exec sh -c '
for f; do
    abs_path=$(realpath "$f")
    echo "=== $abs_path ==="
    cat "$f"
    echo
done
' sh {{}} +"""


SUBMIT_SCORE_TOOL = ToolSpec(
    name="submit_score",
    description="Call this tool to submit your final score for the agent's work. You can call this at any point once you have gathered enough information to make a judgment. The score should be an integer between 0 and 10: 0 if the agent didn't make any progress, 5 if it made significant progress but the solution is incomplete or flawed in an important way, 10 if the agent perfectly succeeded.",
    parameters={
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "description": "Score between 0 and 10.",
            },
        },
        "required": ["score"],
    },
)


class SubmitScoreToolCall(ToolCall):
    score: int

    def to_bash_command(self) -> str:
        assert False, "Do not call SubmitScoreToolCall.to_bash_command"


SUBMIT_CORRECTNESS_TOOL = ToolSpec(
    name="submit_correctness",
    description=(
        "Call this tool to submit your final judgment on whether the agent's solution is correct or incorrect. "
        "You can call this at any point once you have gathered enough information to make a judgment.\n\n"
        "It counts as incorrect if the solution: has correct and incorrect parts; is correct except for one issue "
        "(e.g. one bug); is correct but does not solve the task it was given; is correct for parts of the "
        "requirements but incorrect or missing for at least one; or anything else in this spirit.\n\n"
        "It counts as correct if the solution fully satisfies all the requirements of the task and always works correctly.\n\n"
        "If the task is ambiguous, correct means correct according to any reasonable interpretation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "correct": {
                "type": "boolean",
                "description": "true if the agent's solution is correct, false if incorrect.",
            },
        },
        "required": ["correct"],
    },
)


class SubmitCorrectnessToolCall(ToolCall):
    correct: bool

    def to_bash_command(self) -> str:
        assert False, "Do not call SubmitCorrectnessToolCall.to_bash_command"


### HELPERS

NAME_TO_TOOL_CALL_CLASS = {
    "bash": BashToolCall,
    "create_file": CreateFileToolCall,
    "edit": EditToolCall,
    "finish": FinishToolCall,
    "submit_solution": SubmitSolutionToolCall,
    "list_directory_contents": ListDirectoryContentsToolCall,
    "submit_score": SubmitScoreToolCall,
    "submit_correctness": SubmitCorrectnessToolCall,
}


def extract_tool_calls(
    message: renderers.Message, available_tools: list[ToolSpec]
) -> list[ToolCall] | ErrorParsingToolCall:
    if "tool_calls" not in message.keys() or len(message["tool_calls"]) == 0:
        return ErrorParsingToolCall("You did not call a tool. Please call a tool.")

    tool_calls: list[ToolCall] = []

    for raw_tool_call in message["tool_calls"]:
        name = raw_tool_call.function.name
        arguments = raw_tool_call.function.arguments

        if not any(tool["name"] == name for tool in available_tools):
            return ErrorParsingToolCall(f'Unknown tool name "{name}"')

        tool_call_class = NAME_TO_TOOL_CALL_CLASS[name]

        try:
            tool_call = tool_call_class(**json.loads(arguments))  # type: ignore
        except (TypeError, ValidationError, json.decoder.JSONDecodeError) as e:
            return ErrorParsingToolCall(
                f"Incorrect arguments for tool {name}: {e.__class__.__name__}: {e}"
            )

        tool_calls.append(tool_call)

    return tool_calls
