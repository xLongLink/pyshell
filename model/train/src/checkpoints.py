"""Resume validation and checkpoint persistence, separate from optimization."""

import json

import torch

from dataset.benchmark.src.corpus import sha256


def load_resume(adapter, data, accumulation, serialization):
    state_file = adapter / "training.json"
    if not state_file.is_file():
        state_file = adapter.parent / "training.json"
    state = json.loads(state_file.read_text())
    if (
        state["train_sha256"] != sha256(data / "train.jsonl")
        or state["accumulation"] != accumulation
    ):
        raise ValueError(
            "Resume dataset or gradient accumulation differs from the checkpoint"
        )
    if state.get("serialization", "chat") != serialization:
        raise ValueError(
            "Cannot resume an adapter trained with a different serialization"
        )
    return state


def save_checkpoint(model, tokenizer, optimizer, status, output, name):
    path = output if name == "final" else output / name
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    state = json.dumps(status, indent=2) + "\n"
    (path / "training.json").write_text(state)
    if path != output:
        (output / "training.json").write_text(state)
