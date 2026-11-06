import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from model.benchmark import main
from model.benchmark.src import corpus, evaluate, final, inference, settings
from model.benchmark.src.artifact import load_artifact
from model.benchmark.src.proof_policy import development_ready


def write_artifact(path, *, candidate):
    path.mkdir()
    (path / "adapter_model.safetensors").write_bytes(path.name.encode())
    native = {"base_model_name_or_path": f"example/{path.name}-base", "revision": "a" * 40}
    (path / "adapter_config.json").write_text(json.dumps(native))
    if candidate:
        (path / "training.json").write_text(
            json.dumps({"base_model": native["base_model_name_or_path"], "base_revision": native["revision"]})
        )
        (path / "run.json").write_text(
            json.dumps({"settings": {"base": native["base_model_name_or_path"], "base_revision": native["revision"]}})
        )


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    records = [{"id": str(i), "family": family} for i, family in enumerate(("join", "recovery", "join"))]
    payload = "".join(json.dumps(row) + "\n" for row in records)
    (data / "dev-tasks.jsonl").write_text(payload)
    (data / "test-tasks.jsonl").write_text("FINAL TASKS MUST NOT BE READ")
    (data / "manifest.json").write_text(
        json.dumps({"scope": "fixture", "splits": {"dev": {"sha256": hashlib.sha256(payload.encode()).hexdigest()}}})
    )
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    write_artifact(baseline, candidate=False)
    write_artifact(candidate, candidate=True)
    config = settings.Settings(
        data=str(data), data_revision=None, baseline=baseline, candidate=candidate, output=tmp_path / "results", threads=3, tokens=48
    )
    models = Mock(side_effect=lambda adapter, *args, **kwargs: SimpleNamespace(label=adapter.name, runtime={"fixture": True}))
    monkeypatch.setattr(inference, "HFCompletion", models)
    workers = Mock()
    monkeypatch.setattr(evaluate, "Worker", workers)
    results = {"baseline": [True, False, False], "candidate": [False, True, True]}

    def trial(worker, record, generate):
        return {**record, "status": "passed" if results[generate.label][int(record["id"])] else "incorrect", "steps": []}

    monkeypatch.setattr(evaluate, "trial", trial)
    return config, models


def test_default_command_compares_checked_in_candidate_without_flags(monkeypatch):
    run = Mock()
    monkeypatch.setattr(evaluate, "run", run)
    main.main([])
    run.assert_called_once_with(settings.SETTINGS)


def test_legacy_flags_cannot_override_model_identity(monkeypatch):
    run = Mock()
    monkeypatch.setattr(evaluate, "run", run)
    with pytest.raises(SystemExit):
        main.main(["evaluate", "--base-model", "wrong-base"])
    run.assert_not_called()
    with pytest.raises(ValidationError):
        settings.Settings(base_model="wrong-base")


def test_paired_report_uses_identical_tasks_and_recorded_model_revisions(benchmark):
    config, models = benchmark
    report = evaluate.run(config)
    assert report["improved"]
    assert report["baseline"]["passed"] == 1
    assert report["candidate"]["passed"] == 2
    assert report["gains"] == ["1", "2"]
    assert report["regressions"] == ["0"]
    assert report["by_family"]["join"] == {
        "tasks": 2,
        "baseline_passed": 1,
        "candidate_passed": 1,
        "delta": 0,
        "baseline_success_rate": 0.5,
        "candidate_success_rate": 0.5,
    }
    assert report["by_family"]["recovery"]["delta"] == 1
    assert json.loads((config.output / "comparison.json").read_text()) == report
    assert (Path(config.data) / "test-tasks.jsonl").read_text() == "FINAL TASKS MUST NOT BE READ"
    assert models.call_args_list[0].kwargs == {"base_model": "example/baseline-base", "revision": "a" * 40, "max_new_tokens": 48}
    assert models.call_args_list[1].kwargs == {"base_model": "example/candidate-base", "revision": "a" * 40, "max_new_tokens": 48}
    summaries = [json.loads((config.output / role / "summary.json").read_text()) for role in ("baseline", "candidate")]
    assert summaries[0]["inference"] == summaries[1]["inference"]
    assert summaries[0]["task_file_sha256"] == summaries[1]["task_file_sha256"]
    assert summaries[1]["run_sha256"] == corpus.sha256(config.candidate / "run.json")


def test_remote_development_run_never_downloads_the_final_tasks(benchmark, monkeypatch):
    config, _ = benchmark
    download = Mock(return_value=str(Path(config.data).parent))
    monkeypatch.setattr(corpus, "snapshot_download", download)
    config = settings.Settings(**{**config.model_dump(), "data": "example/data", "prefix": "data", "data_revision": "b" * 40})
    evaluate.run(config)
    download.assert_called_once_with(
        "example/data", repo_type="dataset", revision="b" * 40, allow_patterns=["data/manifest.json", "data/dev-tasks.jsonl"]
    )


@pytest.mark.parametrize(
    "filename,field,value",
    [
        ("run.json", "base", "different/base"),
        ("run.json", "base_revision", "b" * 40),
        ("training.json", "base_model", "different/base"),
        ("adapter_config.json", "revision", "b" * 40),
    ],
)
def test_disagreeing_artifact_metadata_is_rejected(benchmark, filename, field, value):
    config, models = benchmark
    path = config.candidate / filename
    metadata = json.loads(path.read_text())
    (metadata["settings"] if filename == "run.json" else metadata)[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="disagree"):
        evaluate.run(config)
    models.assert_not_called()


def test_candidate_cannot_fall_back_to_legacy_adapter_metadata(benchmark):
    config, _ = benchmark
    (config.candidate / "run.json").unlink()
    with pytest.raises(ValueError, match="requires training.json and run.json"):
        load_artifact(config.candidate, candidate=True)
    assert load_artifact(config.baseline).revision == "a" * 40


def test_unpinned_base_revision_is_never_guessed(benchmark):
    config, _ = benchmark
    path = config.baseline / "adapter_config.json"
    native = json.loads(path.read_text())
    native["revision"] = None
    path.write_text(json.dumps(native))
    with pytest.raises(ValueError, match="pinned base revision"):
        evaluate.run(config)


@pytest.mark.parametrize("change", ["tasks", "inference", "order", "complete"])
def test_incomparable_results_are_not_reported_as_an_improvement(benchmark, change):
    config, _ = benchmark
    evaluate.run(config)
    path = config.output / "candidate/summary.json"
    summary = json.loads(path.read_text())
    if change == "order":
        responses = config.output / "candidate/responses.jsonl"
        responses.write_text("\n".join(reversed(responses.read_text().splitlines())) + "\n")
    elif change == "tasks":
        summary["task_file_sha256"] = "different"
    elif change == "inference":
        summary["inference"]["max_new_tokens"] += 1
    else:
        summary["complete"] = False
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Comparison requires"):
        evaluate.comparison(config.output / "baseline", config.output / "candidate")


def test_existing_comparison_cannot_be_overwritten(benchmark):
    config, models = benchmark
    config.output.mkdir()
    with pytest.raises(FileExistsError):
        evaluate.run(config)
    models.assert_not_called()


@pytest.mark.parametrize("passed", [0, 1])
def test_ties_and_regressions_are_not_improvements(benchmark, monkeypatch, passed):
    config, _ = benchmark

    def trial(worker, record, generate):
        count = 1 if generate.label == "baseline" else passed
        return {**record, "status": "passed" if int(record["id"]) < count else "incorrect", "steps": []}

    monkeypatch.setattr(evaluate, "trial", trial)
    report = evaluate.run(config)
    assert not report["improved"]
    assert report["delta"] <= 0


def test_candidate_revision_comes_from_training_provenance_when_native_config_is_unset(benchmark):
    config, _ = benchmark
    path = config.candidate / "adapter_config.json"
    native = json.loads(path.read_text())
    native["revision"] = None
    path.write_text(json.dumps(native))
    assert load_artifact(config.candidate, candidate=True).revision == "a" * 40


def test_failed_development_gate_does_not_open_final_data(benchmark, monkeypatch):
    config, _ = benchmark
    evaluate.run(config)
    resolve = Mock(side_effect=AssertionError("must not open final tasks"))
    monkeypatch.setattr(final, "resolve_data", resolve)
    with pytest.raises(ValueError, match="final holdout remains unopened"):
        final.run(config)
    resolve.assert_not_called()


def qualify(config):
    path = config.output / "candidate/summary.json"
    summary = json.loads(path.read_text())
    summary.update(tasks=60, success_rate=1.0, by_family={"fixture": {"tasks": 60, "passed": 60}})
    assert development_ready(summary)
    path.write_text(json.dumps(summary))


@pytest.mark.parametrize("change", ["weights", "revision", "settings", "run", "adapter_config"])
def test_changed_candidate_cannot_open_final_data(benchmark, monkeypatch, change):
    config, _ = benchmark
    evaluate.run(config)
    qualify(config)
    if change == "weights":
        (config.candidate / "adapter_model.safetensors").write_bytes(b"changed")
    elif change == "settings":
        config = settings.Settings(**{**config.model_dump(), "tokens": config.tokens + 1})
    elif change == "run":
        with (config.candidate / "run.json").open("a") as file:
            file.write("\n")
    elif change == "adapter_config":
        with (config.candidate / "adapter_config.json").open("a") as file:
            file.write("\n")
    else:
        path = config.output / "candidate/summary.json"
        summary = json.loads(path.read_text())
        summary["base_revision"] = "b" * 40
        path.write_text(json.dumps(summary))
    resolve = Mock(side_effect=AssertionError("must not open final tasks"))
    monkeypatch.setattr(final, "resolve_data", resolve)
    with pytest.raises(ValueError, match="changed after development"):
        final.run(config)
    resolve.assert_not_called()


def test_explicit_final_command_runs_independent_verification(benchmark, monkeypatch):
    config, _ = benchmark
    evaluate.run(config)
    qualify(config)
    resolve = Mock(return_value=Path(config.data))
    run_final = Mock()
    verify = Mock()
    monkeypatch.setattr(final, "resolve_data", resolve)
    monkeypatch.setattr(final, "evaluate", run_final)
    monkeypatch.setattr(final, "verify", verify)
    final.run(config)
    assert [call.args[1] for call in resolve.call_args_list] == ["dev", "test"]
    assert run_final.call_args.kwargs["split"] == "test"
    assert run_final.call_args.kwargs["final"] is True
    verify.assert_called_once_with(config.output / "final", Path(config.data), config.output / "candidate")
