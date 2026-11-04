"""Tokenization and causal loss for executable-response supervision."""

import torch
import torch.nn.functional as F

from dataset.benchmark.src.serialization import serialize, target_text


def encode(record, tokenizer, max_length, serialization="chat"):
    if serialization == "raw":
        prefix = (
            record["prompt"]
            if "prompt" in record
            else serialize(record["messages"][:-1])
        )
        target = (
            record["completion"]
            if "completion" in record
            else record["messages"][-1]["content"]
        )
        target = target_text(target)
    else:
        prefix = tokenizer.apply_chat_template(
            record["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        target = record["messages"][-1]["content"] + tokenizer.eos_token
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    ids = prompt_ids + target_ids
    if len(ids) > max_length:
        raise ValueError(
            f"{record['id']} needs {len(ids)} tokens, above {max_length}; refusing to truncate supervision"
        )
    return {
        "id": record["id"],
        "ids": ids,
        "prefix_length": len(prompt_ids),
    }


def loss_for(model, item):
    ids = torch.tensor([item["ids"]], dtype=torch.long)
    start = item["prefix_length"]
    positions = torch.arange(start - 1, ids.shape[1] - 1)
    output = model(input_ids=ids, use_cache=False, logits_to_keep=positions)
    return F.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]), ids[:, start:].reshape(-1)
    )


def evaluate_loss(model, items):
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for item in items:
            target_tokens = len(item["ids"]) - item["prefix_length"]
            total += float(loss_for(model, item)) * target_tokens
            tokens += target_tokens
    model.train()
    return total / tokens
