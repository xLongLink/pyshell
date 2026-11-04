"""Build balanced training-only trajectories with real execution feedback."""

import argparse
import json
from collections import Counter
from contextlib import closing
from pathlib import Path

from dataset.benchmark.src.corpus import sha256, write_jsonl
from dataset.benchmark.src.execution import Worker
from dataset.benchmark.src.task_types import FAMILIES
from dataset.generation.src.transfer import (
    TEMPLATE_PARTITIONS,
    make_contrast_pair,
    supervision,
)

REPO = Path(__file__).resolve().parents[3]


def training_tasks(pairs, seed):
    slots = TEMPLATE_PARTITIONS["train"]
    for family_index, family in enumerate(FAMILIES):
        for pair in range(pairs):
            yield from make_contrast_pair(
                family,
                pair,
                seed + family_index * 10000019,
                "train",
                slots[pair % len(slots)],
            )


def prepare(output, evaluation, pairs=8, seed=917341):
    frozen = json.loads((evaluation / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    records = []
    with closing(Worker()) as worker:
        for record in training_tasks(pairs, seed):
            for mode in ("direct", "retry") if record["recovery_code"] else ("direct",):
                rows.extend(
                    supervision(
                        record,
                        include_inspection=True,
                        workspace=worker,
                        retry=mode == "retry",
                        id_prefix=f"{record['id']}-{mode}",
                    )
                )
            records.append(
                {key: value for key, value in record.items() if key != "expected"}
            )
    template_ids = {record["template_id"] for record in records}
    for split in ("dev", "test"):
        if template_ids & set(frozen["splits"][split]["template_ids"]):
            raise ValueError("Training templates overlap a held-out split")
    write_jsonl(output / "train.jsonl", rows)
    write_jsonl(output / "train-tasks.jsonl", records)
    (output / "validation.jsonl").write_bytes(
        (evaluation / "validation.jsonl").read_bytes()
    )
    manifest = {
        "experiment": "executed-transfer-curriculum-v1",
        "training_only": True,
        "seed": seed,
        "pairs_per_family": pairs,
        "training_records": len(rows),
        "tasks_by_family": dict(Counter(record["family"] for record in records)),
        "supervision_by_family": dict(Counter(row["family"] for row in rows)),
        "template_ids": sorted(template_ids),
        "feedback": "real network-isolated PythonShell execution",
        "recovery": "both sum and mean receive successful try/except and failed-read/recover/retry trajectories",
        "inspection": "supervised for every task",
        "evaluation_test_sha256": frozen["splits"]["test"]["sha256"],
        "splits": {
            "train": {
                "tasks": len(records),
                "sha256": sha256(output / "train-tasks.jsonl"),
            }
        },
        "training_records_sha256": sha256(output / "train.jsonl"),
        "validation_records_sha256": sha256(output / "validation.jsonl"),
        "generator_sha256": sha256(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in output.iterdir():
        path.chmod(0o444)
    print(json.dumps(manifest, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=REPO / "dataset/benchmark/artifacts/cpu-transfer-v1",
    )
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=917341)
    args = parser.parse_args(argv)
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    prepare(args.output, args.evaluation, args.pairs, args.seed)
