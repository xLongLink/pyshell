import hashlib
import json
import subprocess
from pathlib import Path

from transformers import TrainerCallback


def load_resume(path, metadata):
    if not (path / "trainer_state.json").is_file():
        raise ValueError(
            "resume requires a Trainer checkpoint directory; "
            "use initialize_adapter for older adapters"
        )
    saved = json.loads((path / "training.json").read_text())
    if saved != metadata:
        raise ValueError(
            "Resume dataset, base model or supervision configuration differs from the checkpoint"
        )
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        if not (path / name).is_file():
            raise ValueError(f"Incomplete Trainer checkpoint: missing {name}")
    state = json.loads((path / "trainer_state.json").read_text())
    return state["global_step"]


class Checkpoints(TrainerCallback):
    def __init__(self, metadata, config):
        self.metadata = metadata
        root = Path(__file__).resolve().parents[3]
        files = [
            root / "model/train/main.py",
            root / "model/train/pyproject.toml",
            root / "model/train/uv.lock",
        ]
        files.extend((root / "model/train/src").rglob("*.py"))
        files.extend((root / "model/protocol/src").rglob("*.py"))
        source = hashlib.sha256()
        for file in sorted(files):
            source.update(
                str(file.relative_to(root)).encode() + b"\0" + file.read_bytes()
            )
        self.run = {
            "settings": config.model_dump(mode="json"),
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "source_sha256": source.hexdigest(),
        }

    def write(self, path):
        (path / "training.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        (path / "run.json").write_text(json.dumps(self.run, indent=2) + "\n")

    def on_save(self, args, state, control, **kwargs):
        path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self.write(path)
