import json
from collections import Counter
from contextlib import closing

from model.benchmark.src.corpus import (
    read_jsonl,
    sha256,
    verify_training_data,
    write_jsonl,
)
from model.benchmark.src.execution import Worker


def replay_curriculum(
    output, replay, evaluation, generate, *, generator_source, metadata
):
    frozen = json.loads((evaluation / "manifest.json").read_text())
    verify_training_data(replay, json.loads((replay / "manifest.json").read_text()))
    rows = read_jsonl(replay / "train.jsonl")
    records = read_jsonl(replay / "train-tasks.jsonl")
    output.mkdir(parents=True, exist_ok=False)
    worker = Worker()
    with closing(worker):
        for record, supervision in generate(worker):
            rows.extend(supervision)
            records.append(
                {key: value for key, value in record.items() if key != "expected"}
            )
    return publish_curriculum(
        output,
        replay,
        evaluation,
        frozen,
        rows,
        records,
        generator_source=generator_source,
        metadata=metadata,
    )


def write_corpus(output, evaluation, rows, records):
    write_jsonl(output / "train-tasks.jsonl", records)
    write_jsonl(output / "train.jsonl", rows)
    (output / "validation.jsonl").write_bytes(
        (evaluation / "validation.jsonl").read_bytes()
    )
    return {
        "train_tasks_sha256": sha256(output / "train-tasks.jsonl"),
        "training_records_sha256": sha256(output / "train.jsonl"),
        "validation_records_sha256": sha256(output / "validation.jsonl"),
    }


def seal_corpus(output, manifest):
    serialized = json.dumps(manifest, indent=2) + "\n"
    (output / "manifest.json").write_text(serialized)
    for path in output.iterdir():
        path.chmod(0o444)
    print(serialized, end="", flush=True)


def publish_curriculum(
    output,
    replay,
    evaluation,
    frozen,
    rows,
    records,
    *,
    generator_source,
    metadata,
):
    # Validate against the frozen evaluation before writing any outputs.
    assert len({row["id"] for row in rows}) == len(rows)
    setups = {record["setup"] for record in records}
    templates = {record["template_id"] for record in records}
    for split in ("dev", "test"):
        path = evaluation / f"{split}-tasks.jsonl"
        assert sha256(path) == frozen["splits"][split]["sha256"]
        assert not setups.intersection(record["setup"] for record in read_jsonl(path))
        assert not templates.intersection(frozen["splits"][split]["template_ids"])

    # Keep evaluation validation bytes unchanged.
    hashes = write_corpus(output, evaluation, rows, records)

    # Attribute provenance to the calling generator, not this helper.
    manifest = {
        "training_only": True,
        **metadata,
        "training_records": len(rows),
        "supervision_by_family": dict(Counter(row["family"] for row in rows)),
        "training_records_sha256": hashes["training_records_sha256"],
        "validation_records_sha256": hashes["validation_records_sha256"],
        "evaluation_test_sha256": frozen["splits"]["test"]["sha256"],
        "replay_sha256": sha256(replay / "train.jsonl"),
        "generator_sha256": sha256(generator_source),
    }

    # Seal the published corpus, including the manifest.
    seal_corpus(output, manifest)
    return manifest
