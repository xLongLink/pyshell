import json
from collections import Counter

from model.benchmark.src.corpus import read_jsonl, sha256, write_jsonl


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
    details,
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
    write_jsonl(output / "train-tasks.jsonl", records)
    write_jsonl(output / "train.jsonl", rows)
    (output / "validation.jsonl").write_bytes(
        (evaluation / "validation.jsonl").read_bytes()
    )

    # Attribute provenance to the calling generator, not this helper.
    manifest = {
        "training_only": True,
        **metadata,
        "training_records": len(rows),
        "supervision_by_family": dict(Counter(row["family"] for row in rows)),
        "training_records_sha256": sha256(output / "train.jsonl"),
        "validation_records_sha256": sha256(output / "validation.jsonl"),
        "evaluation_test_sha256": frozen["splits"]["test"]["sha256"],
        "replay_sha256": sha256(replay / "train.jsonl"),
        "generator_sha256": sha256(generator_source),
        **details,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Seal the published corpus, including the manifest.
    for path in output.iterdir():
        path.chmod(0o444)
    print(json.dumps(manifest, indent=2))
    return manifest
