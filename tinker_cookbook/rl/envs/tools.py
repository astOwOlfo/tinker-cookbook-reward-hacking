from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer
from pydantic import BaseModel, ConfigDict, ValidationError
from dataclasses import dataclass
from tinker_cookbook import renderers, model_info, cli_utils
from shlex import quote


class ToolCall(BaseModel, ABC):
    model_config = ConfigDict(frozen=True)
    
    @abstractmethod
    def to_bash_command(self) -> str:
        pass
    
@dataclass(frozen=True, slots=True)
class ErrorParsingToolCall:
    message: str

def get_system_message_with_tools(tokenizer: PreTrainedTokenizer, system_message: str, tools: list[dict]) -> str:
    model_name: str = tokenizer.name_or_path

    if model_name.startswith("Qwen/Qwen3") or model_name.startswith("Qwen/Qwen2.5") or model_name.startswith("Qwen/Qwen2"):
        message: str = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}], tools=tools, tokenize=False
        )
        prefix = "<|im_start|>system\n"
        suffix = "<|im_end|>\n"
        assert message.startswith(prefix)
        message = message.removeprefix(prefix)
        # print(f"{message=}")
        assert message.endswith(suffix)
        message = message.removesuffix(suffix)
        return message

    elif model_name.lower().startswith("openai/gpt-oss"):
        DELIMITER = "DELIMITER_14356728975462398"
        tool_message: str = tokenizer.apply_chat_template(
            [{"role": "system", "content": DELIMITER}], tools=tools, tokenize=False
        )
        suffix = "<|end|>"
        assert tool_message.endswith(suffix)
        tool_message = tool_message.removesuffix(suffix)
        assert tool_message.count(DELIMITER) == 1
        return system_message + tool_message.split(DELIMITER)[-1]

    else:
        assert False, f"Cannot infer model type form model name '{model_name}'."
        
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
        
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute bash commands. Separate commands by newlines to execute multiple.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

@dataclass(frozen=True, slots=True)
class BashToolCall(ToolCall):
    command: str

    def to_bash_command(self) -> str:
        return self.command


CREATE_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "create_file",
        "description": "Create a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file."},
                "content": {"type": "string", "description": "Content of the file."},
            },
            "required": ["path", "content"],
        },
    },
}

class CreateFileToolCall(ToolCall):
    path: str
    content: str

    def to_bash_command(self) -> str:
        return CREATE_FILE_COMMAND.format(path=quote(self.path), content=quote(self.content))

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": "Edit a file by performing a string replacement.",
        "parameters": {
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
    },
}

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

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Call this tool when you are finished and you have tested your solution and you are sure that your solution is correct and all the tests pass. Do not call this tool unless you are absolutely sure your solution is correct and you have tested it.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

class FinishToolCall(ToolCall):
    def to_bash_command(self) -> str:
        assert False, "Do not call FinishToolCall.to_bash_command"

SUBMIT_SOLUTION_TOOL = {
        "type": "function",
        "function": {
            "name": "submit_solution",
            "description": "Call this tool to submit the solution python code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "python_code": {"type": "str"},
                },
                "required": ["python_code"],
            },
        },
    }

class SubmitSolutionToolCall(ToolCall):
    python_code: str

    def to_bash_command(self) -> str:
        assert False, "Do not call SubmitSolutionToolCall.to_bash_command"
        
LIST_DIRECTORY_CONTENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_directory_contents",
        "description": "List the contents of each file in the given directory. Defaults to showing the direct children of the current working directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the directory. If not specified, the current working directory will be used."},
                "max_depth": {"type": "number", "description": "Maximum depth to search. Defaults to 1 (Show only immediate children of the directory)."},
            },
            "required": [],
        },
    },
}


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
        
        
### HELPERS

TOOL_MAPPER = {
    "bash": BashToolCall,
    "create_file": CreateFileToolCall,
    "edit": EditToolCall,
    "finish": FinishToolCall,
    "submit_solution": SubmitSolutionToolCall,
    "list_directory_contents": ListDirectoryContentsToolCall,
}

def extract_tool_call(
    message: renderers.Message, available_tools: list[dict]
) -> ToolCall | ErrorParsingToolCall:
    
    if "tool_calls" not in message.keys():
        return ErrorParsingToolCall("You did not call a tool. Please call a tool.")
    
    # TODO: handle multiple tool calls
    n_calls = len(message["tool_calls"]) # type: ignore
    if n_calls != 1:
        return ErrorParsingToolCall(
            f"You called {n_calls} tools. Please call exactly one tool per message."
        )

    raw_call = message["tool_calls"][0] # type: ignore
    # import json
    # print("TOOL CALLS:", json.dumps(message["tool_calls"], indent=4)) # type: ignore
    if not isinstance(raw_call, dict) or not (set(raw_call.keys()) == {"name", "arguments"} or set(raw_call.keys()) == {"name", "args"}):
        return ErrorParsingToolCall(
            'The tool call should be a json dictionary with keys "name" and "arguments".'
        )
    tool_name = raw_call["name"]
    if "args" in raw_call.keys():
        arguments = raw_call["args"] # type: ignore
    else:
        arguments = raw_call["arguments"] # type: ignore

    for available_tool in available_tools:
        if tool_name != available_tool["function"]["name"]:
            continue

        tool_class = TOOL_MAPPER[tool_name]
        required_arguments = available_tool["function"]["parameters"]["required"]
        if not set(set(required_arguments)).issubset(set(arguments.keys())):
            return ErrorParsingToolCall(
                f"Invalid arguments {set(arguments.keys())} for tool {tool_name}. Required arguments are {required_arguments}."
            )
        
        try:
            tool = tool_class(**arguments) # type: ignore
        except ValidationError as e:
            return ErrorParsingToolCall(f"Incorrect arguments for tool {tool_name}: {str(e)}")
        
        return tool

    return ErrorParsingToolCall(f'Unknown tool name "{tool_name}"')