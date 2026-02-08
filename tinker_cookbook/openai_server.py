import tinker
from tinker import SamplingClient, ServiceClient
from transformers import AutoTokenizer, PreTrainedTokenizer
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook import renderers
from dotenv import load_dotenv
from uuid import uuid4
from time import time
from fastapi import FastAPI, HTTPException
import uvicorn
from argparse import ArgumentParser
import traceback
import asyncio

app = FastAPI()


args = None
renderer: renderers.Renderer = None  # type: ignore
tokenizer: PreTrainedTokenizer = None  # type: ignore
sevice_client: ServiceClient = None  # type: ignore
sampler_path_to_sampling_client: dict[str, SamplingClient] = {}
lock: asyncio.Lock = asyncio.Lock()


def add_tools_to_messages(
    messages: list[renderers.Message], tools: list[renderers.ToolSpec]
) -> list[dict]:
    has_system_message: bool = messages[0]["role"] == "system"
    system_message_content: str
    if has_system_message:
        system_message_content = messages[0]["content"]  # type: ignore
        assert isinstance(system_message_content, str), "System message content must be a string."
        messages = messages[1:]
    else:
        system_message_content = ""

    system_messages_with_tools: list[renderers.Message] = (
        renderer.create_conversation_prefix_with_tools(
            tools=tools,  # type: ignore
            system_prompt=system_message_content,
        )
    )

    return system_messages_with_tools + messages  # type: ignore


def dict_tool_to_tinker_tool(tool) -> renderers.ToolSpec:
    assert isinstance(tool, dict), f"Each tool should be a dict but got object of type {type(tool)}"
    assert set(tool.keys()) == {"type", "function"}, (
        f'Each tool should be a dictionary with keys "type" and "function", but got one with keys {set(tool.keys())}'
    )
    assert tool["type"] == "function", (
        f'The type field of each tool should be the string "function", but got a tool where it is {tool["type"]}'
    )
    assert isinstance(tool["function"], dict), (
        f'The "function" field of each tool should be a dict, but got an object of type {type(tool["function"])}'
    )
    assert set(tool["function"].keys()) == {"name", "description", "parameters"}, (
        f'The "function" field of each tool should be a dict with keys "name", "description", and "parameters", but got one where it is a dict with keys {set(tool["function"].keys())}'
    )
    for field, expected_type in [("name", str), ("description", str), ("parameters", dict)]:
        assert isinstance(tool["function"][field], expected_type), (
            f'tool["function"]["{field}"] should have type {expected_type} but got a tool where it has type {type(tool["function"][field])}'
        )

    return renderers.ToolSpec(
        name=tool["function"]["name"],
        description=tool["function"]["description"],
        parameters=tool["function"]["parameters"],
    )


def dict_message_to_tinker_message(message: dict) -> renderers.Message:
    for required_field in ["role", "content"]:
        assert required_field in message.keys(), (
            f'Message does not have a "{required_field}" field. Its keys are {set(message.keys())}'
        )

    if "tool_calls" in message.keys():
        message["tool_calls"] = [
            renderers.ToolCall(**tool_call) for tool_call in message["tool_calls"]
        ]

    return message  # type: ignore


def tinker_tool_call_to_dict_tool_call(tool_call: renderers.ToolCall) -> dict:
    dict_tool_call: dict = tool_call.model_dump(exclude_none=True)
    if "id" not in dict_tool_call.keys():
        dict_tool_call["id"] = "call_" + str(uuid4()).replace("-", "")
    return dict_tool_call


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def root(data: dict):
    try:
        async with lock:
            if data["model"] not in sampler_path_to_sampling_client.keys():
                sampler_path_to_sampling_client[data["model"]] = (
                    service_client.create_sampling_client(model_path=data["model"])
                )
            sampling_client = sampler_path_to_sampling_client[data["model"]]

        assert "stream" not in data.keys() or not data["stream"], "Streaming is not supported."

        known_fields = [
            "model",
            "messages",
            "tools",
            "stream",
            "max_tokens",
            "max_completion_tokens",
            "seed",
            "temperature",
            "top_k",
            "top_p",
        ]
        extra_fields = [field for field in data.keys() if field not in known_fields]
        assert len(extra_fields) == 0, f"Unsupported parameters: {extra_fields}"

        # prompt_tokens = tokenizer.apply_chat_template(  # type: ignore
        #     data["messages"],
        #     tools=data.get("tools"),
        #     tokenize=True,
        #     add_generation_prompt=True,
        # )["input_ids"]
        # prompt = tinker.types.ModelInput.from_ints(prompt_tokens)  # type: ignore

        messages = data["messages"]
        messages = [dict_message_to_tinker_message(message) for message in messages]
        if "tools" in data.keys() and data["tools"] is not None:
            messages = add_tools_to_messages(
                messages=messages, tools=[dict_tool_to_tinker_tool(tool) for tool in data["tools"]]
            )

        print("MESSAGES:", messages)
        prompt: tinker.types.ModelInput = renderer.build_generation_prompt(messages)  # type: ignore

        max_tokens: int | None = data.get("max_tokens")
        max_completion_tokens: int | None = data.get("max_completion_tokens")
        assert max_tokens is None or max_completion_tokens is None, (
            "max_tokens and max_completion_tokens cannot both be not None at the same time"
        )
        sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens if max_tokens is not None else max_completion_tokens,
            seed=data.get("seed"),
            temperature=data.get("temperature", 1),
            top_k=data.get("top_k", -1),
            top_p=data.get("top_p", 1),
        )

        result = sampling_client.sample(
            prompt=prompt, sampling_params=sampling_params, num_samples=1
        ).result()

        response, parse_success = renderer.parse_response(result.sequences[0].tokens)
        if not parse_success:
            raise HTTPException(status_code=500, detail="Failed parsing LLM response.")

        n_prompt_tokens = len(prompt.to_ints())  # type: ignore
        n_completion_tokens = len(result.sequences[0].tokens)

        return {
            "id": f"chatcmpl-{uuid4()}",
            "object": "chat.completion",
            "created": int(time()),
            "model": data["model"],
            "system_fingerprint": "my_system_fingerprint",
            "service_tier": "default",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": response["role"],
                        "content": response["content"],
                        "tool_calls": [
                            tinker_tool_call_to_dict_tool_call(tool_call)
                            for tool_call in response["tool_calls"]  # type: ignore
                        ]
                        if "tool_calls" in response.keys() and response["tool_calls"] is not None  # type: ignore
                        else None,
                        "function_call": None,
                    },
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": n_prompt_tokens,
                "completion_tokens": n_completion_tokens,
                "total_tokens": n_prompt_tokens + n_completion_tokens,
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Uncaught exception: {type(e)} {e}\n{traceback.format_exc()}"
        )


def main() -> None:
    global args, renderer, tokenizer, service_client

    parser = ArgumentParser()
    parser.add_argument("--renderer", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    renderer = get_renderer(name=args.renderer, tokenizer=tokenizer)

    service_client = tinker.ServiceClient()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    load_dotenv()

    main()


# checkpoints that do the same as the non-fine-tuned models:
# openai/gpt-oss-120b: tinker://d5d4218a-e803-5094-90ba-0044afeea523:train:0/sampler_weights/base
