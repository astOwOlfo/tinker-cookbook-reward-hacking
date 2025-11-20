from transformers import PreTrainedTokenizer


def get_system_message_with_tools(
    tokenizer: PreTrainedTokenizer, system_message: str, tools: list[dict]
) -> str:
    print(f"{tokenizer=}")
    print(f"{system_message=}")
    print(f"{tools=}")
    model_name: str = tokenizer.name_or_path

    if (
        model_name.startswith("Qwen/Qwen3")
        or model_name.startswith("Qwen/Qwen2.5")
        or model_name.startswith("Qwen/Qwen2")
    ):
        message: str = tokenizer.apply_chat_template(  # type: ignore
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
        tool_message: str = tokenizer.apply_chat_template(  # type: ignore
            [{"role": "system", "content": DELIMITER}], tools=tools, tokenize=False
        )
        suffix = "<|end|>"
        assert tool_message.endswith(suffix)
        tool_message = tool_message.removesuffix(suffix)
        assert tool_message.count(DELIMITER) == 1
        return system_message + tool_message.split(DELIMITER)[-1]

    else:
        assert False, f"Cannot infer model type form model name '{model_name}'."
