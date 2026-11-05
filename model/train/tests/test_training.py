import json

import pytest
from trl import SFTConfig

from model.train.src import training


def test_native_trainer_configuration_is_used_without_budget_conversion():
    inputs, config = training.parse_args(
        [
            "--base",
            "base",
            "--data",
            "data",
            "--output_dir",
            "output",
            "--max_steps",
            "5",
            "--gradient_accumulation_steps",
            "2",
            "--learning_rate",
            "0.0003",
            "--save_steps",
            "3",
            "--bf16",
            "false",
        ]
    )
    assert inputs.base == "base"
    assert inputs.data == "data"
    assert isinstance(config, SFTConfig)
    assert config.output_dir == "output"
    assert config.max_steps == 5
    assert config.gradient_accumulation_steps == 2
    assert config.learning_rate == 0.0003
    assert config.save_steps == 3


@pytest.mark.parametrize(
    "option",
    [
        "--stop-after",
        "--steps",
        "--accumulation",
        "--eval-samples",
        "--device",
        "--threads",
    ],
)
def test_custom_training_flags_are_no_longer_supported(option):
    with pytest.raises(ValueError, match="not used"):
        training.parse_args(["--bf16", "false", option, "1"])


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
        training_run("oversized", "--max_length", "1")


def test_only_train_and_validation_files_are_loaded(training_run, tmp_path):
    (tmp_path / "data/test-tasks.jsonl").write_text("not a training dataset")
    (tmp_path / "data/train-tasks.jsonl").write_text("not a training dataset")
    output = training_run("trained", "--max_steps", "3")
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 3


def test_resume_and_adapter_initialization_cannot_be_combined(tmp_path):
    inputs = training.Inputs(initialize_adapter="adapter")
    config = SFTConfig(
        output_dir=str(tmp_path / "new"),
        bf16=False,
        use_cpu=True,
        resume_from_checkpoint="checkpoint",
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        training.train(inputs, config)
