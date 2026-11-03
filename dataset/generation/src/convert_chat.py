"""Create role-free training files from the existing validated examples."""

import argparse
import hashlib
import json
from pathlib import Path

from dataset.benchmark.src.serialization import FORMAT, serialize, target_text

ROOT = Path(__file__).resolve().parents[3] / "model/train"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format": FORMAT,
        "special_tokens_added": False,
        "system_role": False,
        "sources": {},
    }
    for split in ("train", "validation"):
        source = args.source / f"{split}.jsonl"
        records = [json.loads(line) for line in source.read_text().splitlines()]
        converted = [
            {
                "id": record["id"],
                "family": record["family"],
                "prompt": serialize(record["messages"][:-1]),
                "completion": record["messages"][-1]["content"],
                "setup": record["setup"],
                "check": record["check"],
                "previous": record["previous"],
                "inspect_only": record["inspect_only"],
            }
            for record in records
        ]
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in converted)
        )
        manifest["sources"][split] = {
            "count": len(records),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        if split == "train":
            (output / "example.txt").write_text(
                converted[0]["prompt"] + target_text(converted[0]["completion"]) + "\n"
            )
        print(
            f"{split}: {len(converted)} role-free examples -> {output / (split + '.jsonl')}"
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
