import json
import stat
from pathlib import Path

import pytest

from model.benchmark.src.corpus import read_jsonl, sha256, write_jsonl
from model.train.src.generation.publication import publish_curriculum


@pytest.fixture
def publication(tmp_path):
    output, replay, evaluation = [tmp_path / name for name in ("out", "replay", "eval")]
    for directory in (output, replay, evaluation):
        directory.mkdir()
    rows = [
        {"id": "replayed", "family": "binding", "completion": "print('é')"},
        {"id": "new", "family": "join", "completion": "done('ok')"},
        {"id": "recovery", "family": "binding", "completion": "done('fixed')"},
    ]
    records = [{"id": "train", "setup": "train = 1", "template_id": "train-template"}]
    write_jsonl(replay / "train.jsonl", rows[:1])
    write_jsonl(replay / "train-tasks.jsonl", records)
    (replay / "validation.jsonl").write_bytes(b'{"replay": true}\n')
    (replay / "manifest.json").write_text(
        json.dumps(
            {
                "training_records_sha256": sha256(replay / "train.jsonl"),
                "validation_records_sha256": sha256(replay / "validation.jsonl"),
            }
        )
    )
    frozen = {"splits": {}}
    for split in ("dev", "test"):
        path = evaluation / f"{split}-tasks.jsonl"
        write_jsonl(
            path, [{"setup": f"{split} = 1", "template_id": f"{split}-template"}]
        )
        frozen["splits"][split] = {
            "sha256": sha256(path),
            "template_ids": [f"{split}-template"],
        }
    (evaluation / "manifest.json").write_text(json.dumps(frozen))
    (evaluation / "validation.jsonl").write_bytes(b'{ "text": "caf\xc3\xa9" }\r\n\n')
    source = tmp_path / "original_generator.py"
    source.write_text("original generator source\n")
    return dict(
        output=output,
        replay=replay,
        evaluation=evaluation,
        frozen=frozen,
        rows=rows,
        records=records,
        generator_source=source,
        metadata={
            "seed": 17,
            "new_join_pairs": 2,
            "intervention": "original intervention",
        },
    )


def test_publication_preserves_records_bytes_manifest_and_permissions(
    publication, capsys
):
    manifest = publish_curriculum(**publication)
    output = publication["output"]
    assert read_jsonl(output / "train.jsonl") == publication["rows"]
    assert read_jsonl(output / "train-tasks.jsonl") == publication["records"]
    assert (output / "validation.jsonl").read_bytes() == (
        publication["evaluation"] / "validation.jsonl"
    ).read_bytes()
    assert manifest == {
        "training_only": True,
        "seed": 17,
        "new_join_pairs": 2,
        "training_records": 3,
        "supervision_by_family": {"binding": 2, "join": 1},
        "training_records_sha256": sha256(output / "train.jsonl"),
        "validation_records_sha256": sha256(output / "validation.jsonl"),
        "evaluation_test_sha256": publication["frozen"]["splits"]["test"]["sha256"],
        "replay_sha256": sha256(publication["replay"] / "train.jsonl"),
        "generator_sha256": sha256(publication["generator_source"]),
        "intervention": "original intervention",
    }
    expected = json.dumps(manifest, indent=2) + "\n"
    assert (output / "manifest.json").read_text() == expected
    assert capsys.readouterr().out == expected
    assert {path.name for path in output.iterdir()} == {
        "train.jsonl",
        "train-tasks.jsonl",
        "validation.jsonl",
        "manifest.json",
    }
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in output.iterdir())


def test_duplicate_row_ids_fail_before_writing(publication):
    publication["rows"].append(dict(publication["rows"][0]))
    with pytest.raises(AssertionError):
        publish_curriculum(**publication)
    assert list(publication["output"].iterdir()) == []


@pytest.mark.parametrize("split", ["dev", "test"])
def test_frozen_task_tampering_fails_before_writing(publication, split):
    path = publication["evaluation"] / f"{split}-tasks.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(AssertionError):
        publish_curriculum(**publication)
    assert list(publication["output"].iterdir()) == []


@pytest.mark.parametrize("split", ["dev", "test"])
@pytest.mark.parametrize(
    "field,value",
    [("setup", "{split} = 1"), ("template_id", "{split}-template")],
)
def test_evaluation_overlap_fails_before_writing(publication, split, field, value):
    publication["records"][0][field] = value.format(split=split)
    with pytest.raises(AssertionError):
        publish_curriculum(**publication)
    assert list(publication["output"].iterdir()) == []


@pytest.mark.parametrize("curriculum", ["binding_curriculum", "join_curriculum"])
def test_curricula_keep_generator_provenance_and_metadata(
    publication, monkeypatch, curriculum
):
    from importlib import import_module

    from model.benchmark.src.reference import ReferenceWorkspace

    module = import_module(f"model.train.src.generation.{curriculum}")
    workers = []

    class Workspace(ReferenceWorkspace):
        def __init__(self):
            self.closed = False
            workers.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "Worker", Workspace)
    output = publication["output"] / "curriculum"
    kwargs = {"pairs": 1, "seed": 17}
    if curriculum == "binding_curriculum":
        kwargs.update(support_pairs=1, language=True)
    manifest = module.prepare(
        output, publication["replay"], publication["evaluation"], **kwargs
    )
    assert len(workers) == 1 and workers[0].closed
    assert manifest["generator_sha256"] == sha256(Path(module.__file__))
    assert manifest["replay_sha256"] == sha256(publication["replay"] / "train.jsonl")
    assert manifest["seed"] == 17
    assert read_jsonl(output / "train.jsonl")[0] == publication["rows"][0]
    if curriculum == "binding_curriculum":
        assert manifest["language_diversity"] is True
        assert manifest["pairs_per_target_family"] == 1
        assert manifest["support_pairs_per_family"] == 1
        assert manifest["language_generator_sha256"] == sha256(
            Path(module.__file__).with_name("binding_language.py")
        )
        assert manifest["feedback"]
        assert "intervention" not in manifest
    else:
        assert manifest["new_join_pairs"] == 1
        assert manifest["intervention"]
        assert "language_generator_sha256" not in manifest
