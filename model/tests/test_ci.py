import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from model import ci


@pytest.mark.parametrize(
    "paths,expected",
    [
        ([], False),
        (["sample/main.py"], False),
        (["model/train/tests/test_training.py"], False),
        (["model/benchmark/tests/test_evaluate.py"], False),
        (["model/protocol/tests/test_parsing.py"], False),
        (["model/README.md"], False),
        (["model/train/README.md"], False),
        (["model/train/src/settings.py"], True),
        (["model/train/uv.lock"], True),
        (["model/protocol/src/pyshell/parsing.py"], True),
        (["model/benchmark/src/settings.py"], True),
        (["model/benchmark/uv.lock"], True),
        (["model/benchmark/tasks.jsonl"], True),
        ([".github/workflows/model.yml"], False),
        ([".github/workflows/tests.yml", "model/ci.py"], False),
        (["model/train/data/train.jsonl"], True),
        (["model/ci.py", "model/train/data/train.jsonl"], True),
        (
            ["model/benchmark/tests/test_evaluate.py", "model/benchmark/tasks.jsonl"],
            True,
        ),
    ],
)
def test_change_classification(paths, expected):
    assert ci.changes(paths) == expected


@pytest.mark.parametrize(
    "paths,expected",
    [
        (".github/workflows/model.yml\nmodel/ci.py\n", "false"),
        ("model/train/src/settings.py\n", "true"),
        ("model/benchmark/tasks.jsonl\n", "true"),
    ],
)
def test_changes_command_classifies_pr_file_list(tmp_path, paths, expected):
    output = tmp_path / "outputs"
    subprocess.run(
        [sys.executable, str(Path(ci.__file__)), "changes"],
        input=paths,
        text=True,
        check=True,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
    )
    assert output.read_text() == f"relevant={expected}\n"


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


@pytest.mark.parametrize("status", ["passed", "incorrect"])
def test_failed_gate_preserves_metrics(tmp_path, status):
    results(tmp_path / "candidate.jsonl", [status])
    results(tmp_path / "baseline.jsonl", ["passed"])
    (tmp_path / "baseline.json").write_text('{"repo":"example/model","revision":"old"}')
    with pytest.raises(SystemExit, match="strictly improve"):
        ci.report(tmp_path, False)
    assert json.loads((tmp_path / "metrics.json").read_text())["improved"] is False


def test_improved_candidate_passes_on_current_benchmark(tmp_path):
    results(tmp_path / "candidate.jsonl", ["passed", "passed"])
    results(tmp_path / "baseline.jsonl", ["passed", "incorrect"])
    (tmp_path / "baseline.json").write_text('{"repo":"example/model","revision":"old"}')
    ci.report(tmp_path, False)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["improved"] is True
    assert metrics["candidate"]["passed"] == 2
    assert metrics["baseline"]["passed"] == 1


def test_both_adapters_use_current_benchmark_and_identical_settings(
    tmp_path, monkeypatch
):
    from model.benchmark.src import evaluate, settings

    current = settings.Settings(data=str(tmp_path), calls=2, tokens=32)
    run = Mock()
    monkeypatch.setattr(settings, "SETTINGS", current)
    monkeypatch.setattr(evaluate, "run", run)
    ci.evaluate(tmp_path, "baseline", "baseline.jsonl")
    ci.evaluate(tmp_path, "candidate", "candidate.jsonl")
    accepted = run.call_args_list[0].args[0]
    candidate = run.call_args_list[1].args[0]
    assert accepted.model_dump(exclude={"adapter", "output"}) == candidate.model_dump(
        exclude={"adapter", "output"}
    )
    assert candidate.data == str(tmp_path)
    assert candidate.calls == 2
    assert candidate.tokens == 32
    assert accepted.adapter == tmp_path / "baseline"
    assert candidate.adapter == tmp_path / "candidate"
    for name in ("baseline", "candidate"):
        config = json.loads((tmp_path / f"{name}.jsonl.config.json").read_text())
        assert config["commit"] == ci.git("rev-parse", "HEAD")
        assert config["settings"]["output"] == str(tmp_path / f"{name}.jsonl")


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
