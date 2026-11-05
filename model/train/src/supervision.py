from pyshell.serialization import target_text


def encode(record, tokenizer, max_length):
    if not isinstance(record.get("prompt"), str) or not isinstance(
        record.get("completion"), str
    ):
        raise TypeError("Supervision requires raw prompt and completion strings")
    prompt_ids = tokenizer.encode(record["prompt"], add_special_tokens=False)
    target_ids = tokenizer.encode(
        target_text(record["completion"]), add_special_tokens=False
    )
    ids = prompt_ids + target_ids
    if not prompt_ids:
        raise ValueError("Supervision requires a nonempty prompt")
    if len(ids) > max_length:
        raise ValueError(
            f"{record['id']} needs {len(ids)} tokens, above {max_length}; refusing to truncate supervision"
        )
    return {
        "input_ids": ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
    }
