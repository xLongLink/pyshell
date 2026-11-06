from pyshell.serialization import END_CODE, target_text


def prepare(record, tokenizer, max_length):
    if not isinstance(record["prompt"], str) or not isinstance(
        record["completion"], str
    ):
        raise TypeError("Training requires raw prompt and completion strings")
    if not record["prompt"]:
        raise ValueError("Training requires a nonempty prompt")
    completion = record["completion"].removesuffix(tokenizer.eos_token)
    code = completion.removesuffix("\n" + END_CODE)
    completion = target_text(code) + tokenizer.eos_token
    prompt_ids = tokenizer(record["prompt"])["input_ids"]
    ids = tokenizer(record["prompt"] + completion)["input_ids"]
    if ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt and completion must meet on a token boundary")
    if max_length is not None and len(ids) > max_length:
        raise ValueError(
            f"Example needs {len(ids)} tokens, above {max_length}; refusing to truncate supervision"
        )
    return {"prompt": record["prompt"], "completion": completion}
