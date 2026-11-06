import json
import pytest
from unittest.mock import Mock
from model.benchmark import main
from model.benchmark.src import corpus, evaluate, inference
from model.benchmark.src.settings import SETTINGS, Settings


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    records = [{"id": str(index), "family": "sum"} for index in range(3)]
    (tmp_path / "dev-tasks.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (tmp_path / "test-tasks.jsonl").write_text("DO NOT READ HOLDOUT")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "fixture-base", "revision": "a" * 40}))
    config = Settings(data=str(tmp_path), data_revision=None, adapter=adapter, output=tmp_path / "results.jsonl")
    model = Mock()
    monkeypatch.setattr(inference, "HFCompletion", model)
    worker = Mock()
    monkeypatch.setattr(evaluate, "Worker", worker)
    rows = [
        {**record, "status": status, "steps": []} for record, status in zip(records, ("passed", "incorrect", "call_limit"), strict=True)
    ]
    monkeypatch.setattr(evaluate, "trial", Mock(side_effect=rows))
    return config, model, worker, rows


def test_run_evaluates_one_adapter_and_saves_results(benchmark, capsys):
    config, model, worker, rows = benchmark
    summary = evaluate.run(config)
    assert summary == {"tasks": 3, "passed": 1, "success_rate": 1 / 3, "statuses": {"passed": 1, "incorrect": 1, "call_limit": 1}}
    assert [json.loads(line) for line in config.output.read_text().splitlines()] == rows
    model.assert_called_once_with(
        config.adapter, config.threads, config.device, base_model="fixture-base", revision="a" * 40, max_new_tokens=config.tokens
    )
    worker.return_value.close.assert_called_once_with()
    assert "Passed 1/3 (33.3%)" in capsys.readouterr().out
    assert (config.output.parent / "test-tasks.jsonl").read_text() == "DO NOT READ HOLDOUT"


def test_existing_results_are_not_overwritten(benchmark):
    config, model, _, _ = benchmark
    config.output.write_text("previous run")
    with pytest.raises(FileExistsError):
        evaluate.run(config)
    assert config.output.read_text() == "previous run"
    model.assert_not_called()


def test_empty_tasks_are_rejected_before_loading_model(benchmark):
    config, model, _, _ = benchmark
    (config.output.parent / "dev-tasks.jsonl").write_text("")
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate.run(config)
    model.assert_not_called()
    assert not config.output.exists()


def test_failure_preserves_completed_rows_and_closes_worker(benchmark, monkeypatch):
    config, _, worker, rows = benchmark
    monkeypatch.setattr(evaluate, "trial", Mock(side_effect=[rows[0], RuntimeError("generation failed")]))
    with pytest.raises(RuntimeError, match="generation failed"):
        evaluate.run(config)
    assert [json.loads(line) for line in config.output.read_text().splitlines()] == rows[:1]
    worker.return_value.close.assert_called_once_with()


def test_remote_data_downloads_only_development_tasks(benchmark, monkeypatch):
    config, _, _, _ = benchmark
    directory = config.output.parent / config.prefix
    directory.mkdir(parents=True)
    (directory / "dev-tasks.jsonl").write_text('{"id": "remote"}\n')
    download = Mock(return_value=str(config.output.parent))
    monkeypatch.setattr(corpus, "snapshot_download", download)
    remote = config.model_copy(update={"data": "example/data"})
    assert corpus.load_tasks(remote) == [{"id": "remote"}]
    download.assert_called_once_with(
        "example/data", repo_type="dataset", revision=config.data_revision, allow_patterns=[f"{config.prefix}/dev-tasks.jsonl"]
    )


def test_entrypoint_runs_configured_evaluation(monkeypatch):
    run = Mock()
    monkeypatch.setattr(evaluate, "run", run)
    main.main([])
    run.assert_called_once_with(SETTINGS)
