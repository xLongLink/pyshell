import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock
from model.benchmark import main
from model.benchmark.src import evaluate, inference
from model.benchmark.src.settings import SETTINGS, Settings


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    records = [{"id": str(index), "family": "sum"} for index in range(3)]
    (tmp_path / "dev-tasks.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (tmp_path / "test-tasks.jsonl").write_text("DO NOT READ HOLDOUT")
    config = Settings(data=str(tmp_path), output=tmp_path / "results.jsonl")
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
    evaluate.run(config)
    assert [json.loads(line) for line in config.output.read_text().splitlines()] == rows
    model.assert_called_once_with(config)
    worker.return_value.close.assert_called_once_with()
    assert "Passed 1/3 (33.3%)" in capsys.readouterr().out


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


def test_missing_local_benchmark_is_rejected_before_loading_model(benchmark):
    config, model, _, _ = benchmark
    missing = config.model_copy(update={"data": str(config.output.parent / "missing")})
    with pytest.raises(FileNotFoundError):
        evaluate.run(missing)
    model.assert_not_called()


def test_entrypoint_runs_configured_evaluation(monkeypatch):
    run = Mock()
    monkeypatch.setattr(evaluate, "run", run)
    monkeypatch.setattr(sys, "argv", ["model.benchmark.main"])
    main.main()
    run.assert_called_once_with(SETTINGS)


def test_default_benchmark_is_repository_local(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    data = Path(Settings().data)
    assert data == Path(__file__).resolve().parents[1] / "data"
    assert (data / "dev-tasks.jsonl").is_file()
