import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from model import ci


@pytest.mark.parametrize(
    "paths,expected",
    [
        (["sample/main.py"], (False, False)),
        (["model/train/tests/test_training.py"], (False, False)),
        (["model/README.md"], (False, False)),
        (["model/train/src/settings.py"], (True, True)),
        (["model/train/uv.lock"], (True, True)),
        (["model/protocol/src/pyshell/parsing.py"], (True, True)),
        (["model/benchmark/src/settings.py"], (True, False)),
        (["model/benchmark/uv.lock"], (True, False)),
        (["model/benchmark/tasks.jsonl"], (True, False)),
        ([".github/workflows/model.yml"], (True, True)),
        (["model/ci.py"], (True, True)),
        (["model/train/data/train.jsonl"], (True, True)),
    ],
)
def test_change_classification(paths, expected):
    assert ci.changes(paths) == expected


def results(path, statuses):
    rows = [
        {"id": str(index), "family": "arithmetic", "status": status}
        for index, status in enumerate(statuses)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.mark.parametrize(
    "statuses,improved",
    [
        (["passed", "passed"], True),
        (["passed", "incorrect"], False),
        (["incorrect", "incorrect"], False),
    ],
)
def test_only_strict_improvement_passes(tmp_path, statuses, improved):
    candidate = results(tmp_path / "candidate.jsonl", statuses)
    baseline = results(tmp_path / "baseline.jsonl", ["passed", "incorrect"])
    comparison = ci.compare(candidate, baseline)
    assert comparison["improved"] is improved
    assert comparison["baseline"]["success_rate"] == 0.5
    assert comparison["candidate"]["families"]["arithmetic"]["total"] == 2


@pytest.mark.parametrize("content", ["", '{"id":"x","family":"f","status":"error"}\n'])
def test_invalid_results_are_rejected(tmp_path, content):
    path = tmp_path / "results.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError):
        ci.score(path)


def test_duplicate_tasks_are_rejected(tmp_path):
    path = results(tmp_path / "results.jsonl", ["passed"])
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="unique"):
        ci.score(path)


def test_different_tasks_are_not_comparable(tmp_path):
    candidate = results(tmp_path / "candidate.jsonl", ["passed", "passed"])
    baseline = results(tmp_path / "baseline.jsonl", ["incorrect"])
    with pytest.raises(ValueError, match="same tasks"):
        ci.compare(candidate, baseline)


def test_failed_gate_preserves_metrics(tmp_path):
    results(tmp_path / "candidate.jsonl", ["passed"])
    results(tmp_path / "baseline.jsonl", ["passed"])
    (tmp_path / "baseline.json").write_text('{"repo":"example/model","revision":"old"}')
    with pytest.raises(SystemExit, match="strictly improve"):
        ci.report(tmp_path, False)
    assert json.loads((tmp_path / "metrics.json").read_text())["improved"] is False


def test_evaluation_only_change_does_not_require_improvement(tmp_path):
    results(tmp_path / "baseline.jsonl", ["passed"])
    results(tmp_path / "proposed.jsonl", ["incorrect", "incorrect"])
    ci.report(tmp_path, False, evaluation_only=True)
    assert not (tmp_path / "metrics.json").exists()


@pytest.fixture
def publication(tmp_path, monkeypatch):
    import huggingface_hub

    metrics = {
        "improved": True,
        "tree": ci.git("rev-parse", "HEAD^{tree}"),
        "commit": "tested-commit",
        "bootstrap": False,
        "baseline_model": {"repo": "example/model", "revision": "accepted"},
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    adapter = tmp_path / "candidate"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha="accepted")
    monkeypatch.setattr(huggingface_hub, "HfApi", Mock(return_value=api))
    return tmp_path, metrics, api


def test_publish_preserves_exact_adapter_and_uses_atomic_baseline_check(publication):
    directory, _, api = publication
    ci.publish(directory, "example/model")
    assert (directory / "adapter_model.safetensors").read_bytes() == b"weights"
    assert api.upload_folder.call_args.kwargs["parent_commit"] == "accepted"
    assert api.upload_folder.call_args.kwargs["repo_id"] == "example/model"
    api.create_repo.assert_not_called()


@pytest.mark.parametrize("name", ["adapter_config.json", "adapter_model.safetensors"])
@pytest.mark.parametrize("empty", [False, True])
def test_incomplete_adapter_cannot_be_published(publication, name, empty):
    directory, metrics, api = publication
    metrics["bootstrap"] = True
    (directory / "metrics.json").write_text(json.dumps(metrics))
    path = directory / "candidate" / name
    if empty:
        path.write_bytes(b"")
    else:
        path.unlink()
    with pytest.raises(ValueError, match="missing or empty"):
        ci.publish(directory, "example/model")
    api.create_repo.assert_not_called()
    api.upload_folder.assert_not_called()


@pytest.mark.parametrize("field,value", [("tree", "different"), ("improved", False)])
def test_unapproved_candidate_cannot_be_published(publication, field, value):
    directory, metrics, api = publication
    metrics[field] = value
    (directory / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="exact passing"):
        ci.publish(directory, "example/model")
    api.upload_folder.assert_not_called()


def test_stale_baseline_cannot_be_promoted(publication):
    directory, _, api = publication
    api.model_info.return_value.sha = "new-accepted-model"
    with pytest.raises(ValueError, match="Accepted model changed"):
        ci.publish(directory, "example/model")
    api.upload_folder.assert_not_called()


def test_bootstrap_never_overwrites_existing_repository(publication):
    directory, metrics, api = publication
    metrics["bootstrap"] = True
    metrics["baseline_model"]["revision"] = None
    (directory / "metrics.json").write_text(json.dumps(metrics))
    api.create_repo.side_effect = RuntimeError("repository exists")
    with pytest.raises(RuntimeError, match="repository exists"):
        ci.publish(directory, "example/model")
    api.create_repo.assert_called_once_with(
        "example/model", private=True, exist_ok=False
    )
    api.upload_folder.assert_not_called()


def test_baseline_download_uses_immutable_revision(tmp_path, monkeypatch):
    import huggingface_hub

    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha="accepted-revision")
    download = Mock()
    monkeypatch.setattr(huggingface_hub, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    ci.baseline(tmp_path, "example/model", False)
    assert download.call_args.kwargs["revision"] == "accepted-revision"
    assert json.loads((tmp_path / "baseline.json").read_text()) == {
        "repo": "example/model",
        "revision": "accepted-revision",
    }


def test_bootstrap_does_not_require_existing_model(tmp_path, monkeypatch):
    import huggingface_hub

    api = Mock()
    monkeypatch.setattr(huggingface_hub, "HfApi", Mock(return_value=api))
    ci.baseline(tmp_path, "example/new-model", True)
    api.model_info.assert_not_called()
    assert json.loads((tmp_path / "baseline.json").read_text())["revision"] is None
