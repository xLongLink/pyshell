import json
import hashlib


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def adapter_weights(adapter):
    return adapter / "adapter_model.safetensors" if adapter.is_dir() else adapter


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_tasks(directory, split, *, final=False, limit=None):
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / f"{split}-tasks.jsonl"
    digest = sha256(path)
    if digest != manifest["splits"][split]["sha256"]:
        raise ValueError("Frozen evaluation tasks have changed")
    if split == "test" and not final:
        raise ValueError("The untouched test requires --final after checkpoint selection")
    if split == "test" and limit:
        raise ValueError("The final test cannot be reduced or cherry-picked")
    records = read_jsonl(path)
    return manifest, digest, records[:limit] if limit else records
