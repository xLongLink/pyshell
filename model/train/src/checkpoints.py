"""Resume validation and checkpoint persistence, separate from optimization."""

import json
import os
import tempfile
from pathlib import Path

import torch

from model.benchmark.src.corpus import sha256


def load_resume(adapter, data, accumulation):
    state = json.loads((adapter / "training.json").read_text())
    if (
        state["train_sha256"] != sha256(data / "train.jsonl")
        or state["accumulation"] != accumulation
    ):
        raise ValueError(
            "Resume dataset or gradient accumulation differs from the checkpoint"
        )
    if state.get("serialization") != "raw":
        raise ValueError("Cannot resume an adapter without raw protocol serialization")
    return state


def save_checkpoint(model, tokenizer, optimizer, status, output, name, *, scheduler):
    path = output if name == "final" else output / name
    if path != output and path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    state = json.dumps(status, indent=2) + "\n"
    with tempfile.TemporaryDirectory(dir=output, prefix=f".{name}-") as temporary:
        staged = Path(temporary)
        model.save_pretrained(staged)
        tokenizer.save_pretrained(staged)
        torch.save(optimizer.state_dict(), staged / "optimizer.pt")
        torch.save(scheduler.state_dict(), staged / "scheduler.pt")
        metadata = staged / "training.json"
        metadata.write_text(state)
        if path == output:
            metadata.chmod(metadata.stat().st_mode & 0o600)
        for file in staged.rglob("*"):
            if file.is_file():
                with file.open("rb") as stream:
                    os.fsync(stream.fileno())
        if (output / "steps.jsonl").is_file():
            with (output / "steps.jsonl").open("rb") as stream:
                os.fsync(stream.fileno())
        sync_directory(staged)
        if path == output:
            for file in staged.iterdir():
                if file.name != "training.json":
                    os.replace(file, output / file.name)
            sync_directory(output)
            os.replace(metadata, output / "training.json")
        else:
            os.replace(staged, path)
            sync_directory(output)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=output, delete=False
            ) as file:
                file.write(state)
                file.flush()
                os.fsync(file.fileno())
            os.replace(file.name, output / "training.json")
        sync_directory(output)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
