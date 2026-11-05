"""Tokenization and causal loss for executable-response supervision."""

import torch
import torch.nn.functional as F
from pyshell.serialization import target_text


def encode(record, tokenizer, max_length):
    if "prompt" not in record or "completion" not in record:
        raise ValueError("Supervision requires raw prompt and completion fields")
    prefix = record["prompt"]
    target = target_text(record["completion"])
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
    device = next(model.parameters()).device
    ids = torch.tensor([item["ids"]], dtype=torch.long, device=device)
    start = item["prefix_length"]
    positions = torch.arange(start - 1, ids.shape[1] - 1, device=device)
    output = model(input_ids=ids, use_cache=False, logits_to_keep=positions)
    return F.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]), ids[:, start:].reshape(-1)
    )


def evaluate_loss(model, items):
    if not items:
        return None
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for item in items:
            target_tokens = len(item["ids"]) - item["prefix_length"]
            total += float(loss_for(model, item)) * target_tokens
            tokens += target_tokens
    model.train()
    return total / tokens
