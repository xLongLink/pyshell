import json
import shutil
from unittest.mock import Mock

import pytest

from model.train.src import settings, training


def test_entrypoint_uses_the_checked_in_settings(monkeypatch):
    train = Mock()
    monkeypatch.setattr(training, "train", train)
    training.main()
    train.assert_called_once_with(settings.SETTINGS)


def test_existing_output_is_not_overwritten(training_run, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "user-data"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        training_run("existing")
    assert sentinel.read_text() == "keep"


def test_empty_training_data_is_rejected(training_run, tmp_path):
    (tmp_path / "data/train.jsonl").write_text("")
    with pytest.raises(ValueError):
        training_run("empty")


def test_overlength_training_data_is_rejected(training_run):
    with pytest.raises(ValueError, match="refusing to truncate"):
        training_run("oversized", max_length=1)


def test_only_train_and_validation_files_are_loaded(training_run, tmp_path):
    (tmp_path / "data/test-tasks.jsonl").write_text("not a training dataset")
    (tmp_path / "data/train-tasks.jsonl").write_text("not a training dataset")
    output = training_run("trained", steps=3)
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 3


def test_pinned_dataset_download_only_fetches_training_splits(
    training_run, tmp_path, monkeypatch
):
    revision = "a" * 40
    snapshot = tmp_path / revision
    shutil.copytree(tmp_path / "data", snapshot / "data")
    download = Mock(return_value=str(snapshot))
    monkeypatch.setattr(training, "snapshot_download", download)
    output = training_run("downloaded", data="example/dataset", data_revision=revision)
    download.assert_called_once_with(
        "example/dataset",
        repo_type="dataset",
        revision=revision,
        allow_patterns=["data/train.jsonl", "data/validation.jsonl"],
    )
    assert (
        json.loads((output / "training.json").read_text())["data_revision"] == revision
    )


def test_candidate_records_settings_and_source_identity(training_run):
    output = training_run("candidate", learning_rate=0.002, rank=4, alpha=8)
    run = json.loads((output / "run.json").read_text())
    assert run["settings"]["learning_rate"] == 0.002
    assert run["settings"]["output"] == str(output)
    assert len(run["code_commit"]) == 40
    assert len(run["source_sha256"]) == 64
    for checkpoint in ("checkpoint-2", "checkpoint-4"):
        assert json.loads((output / checkpoint / "run.json").read_text()) == run
    adapter = json.loads((output / "adapter_config.json").read_text())
    assert adapter["r"] == 4
    assert adapter["lora_alpha"] == 8
