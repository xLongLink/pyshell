import json
from pathlib import Path
from huggingface_hub import snapshot_download


def load_tasks(config):
    if Path(config.data).is_dir():
        directory = Path(config.data)
    else:
        snapshot = snapshot_download(
            config.data, repo_type="dataset", revision=config.data_revision, allow_patterns=[f"{config.prefix}/dev-tasks.jsonl"]
        )
        directory = Path(snapshot) / config.prefix
    return [json.loads(line) for line in (directory / "dev-tasks.jsonl").read_text().splitlines()]
