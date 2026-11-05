"""Create role-free training files from the existing validated examples."""

import argparse
import json
from pathlib import Path

from pyshell.serialization import FORMAT, serialize, target_text

from model.benchmark.src.corpus import read_jsonl, sha256, write_jsonl

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/chat")
    parser.add_argument("--output", type=Path, default=ROOT / "data/chat/raw")
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
        records = read_jsonl(source)
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
        write_jsonl(output / f"{split}.jsonl", converted)
        manifest["sources"][split] = {
            "count": len(records),
            "sha256": sha256(source),
        }
        if split == "train":
            (output / "example.txt").write_text(
                converted[0]["prompt"] + target_text(converted[0]["completion"]) + "\n"
            )
        print(
            f"{split}: {len(converted)} role-free examples -> {output / (split + '.jsonl')}"
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
