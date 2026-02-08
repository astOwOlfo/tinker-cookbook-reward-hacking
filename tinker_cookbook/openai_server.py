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
import asyncio

app = FastAPI()


args = None
renderer: renderers.Renderer = None  # type: ignore
tokenizer: PreTrainedTokenizer = None  # type: ignore
sevice_client: ServiceClient = None  # type: ignore
sampler_path_to_sampling_client: dict[str, SamplingClient] = {}
lock: asyncio.Lock = asyncio.Lock()


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
            "seed",
            "temperature",
            "top_k",
            "top_p",
        ]
        extra_fields = [field for field in data.keys() if field not in known_fields]
        assert len(extra_fields) == 0, f"Unsupported parameters: {extra_fields}"

        prompt_tokens = tokenizer.apply_chat_template(  # type: ignore
            data["messages"],
            tools=data.get("tools"),
            tokenize=True,
            add_generation_prompt=True,
        )["input_ids"]
        prompt = tinker.types.ModelInput.from_ints(prompt_tokens)  # type: ignore

        sampling_params = tinker.SamplingParams(
            max_tokens=data.get("max_tokens"),
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

        n_prompt_tokens = len(prompt_tokens)  # type: ignore
        n_completion_tokens = len(result.sequences[0].tokens)

        if "tool_calls" in response.keys():
            for tool_call in response["tool_calls"]:  # type: ignore
                if tool_call.id is None:
                    tool_call.id = "call_" + str(uuid4()).replace("-", "")

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
                            tool_call.model_dump(exclude_none=True)
                            for tool_call in response["tool_calls"]  # type: ignore
                        ]
                        if "tool_calls" in response.keys()
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
        raise HTTPException(status_code=400, detail=f"Uncaught exception: {type(e)} {e}")


def main() -> None:
    global args, renderer, tokenizer, service_client

    parser = ArgumentParser()
    parser.add_argument("--renderer", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--sampler-path", type=str)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    renderer = get_renderer(name=args.renderer, tokenizer=tokenizer)

    service_client = tinker.ServiceClient()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    load_dotenv()

    main()


# checkpoints that do the same as the non-fine-tuned models:
# openai/gpt-oss-120b: tinker://d5d4218a-e803-5094-90ba-0044afeea523:train:0/sampler_weights/base
