import json
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def resolve_data(config, split):
    if Path(config.data).is_dir():
        return Path(config.data)
    snapshot = snapshot_download(
        config.data,
        repo_type="dataset",
        revision=config.data_revision,
        allow_patterns=[f"{config.prefix}/manifest.json", f"{config.prefix}/{split}-tasks.jsonl"],
    )
    return Path(snapshot) / config.prefix


def load_tasks(directory, split, *, final=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / f"{split}-tasks.jsonl"
    digest = sha256(path)
    if digest != manifest["splits"][split]["sha256"]:
        raise ValueError("Frozen evaluation tasks have changed")
    if split == "test" and not final:
        raise ValueError("The untouched test requires the explicit final command after checkpoint selection")
    return manifest, digest, read_jsonl(path)
