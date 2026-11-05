import json

import pytest


def test_validation_uses_the_entire_supplied_split(training_run):
    output = training_run("validation")
    state = json.loads((output / "trainer_state.json").read_text())
    assert sum("eval_loss" in row for row in state["log_history"]) == 2
    metadata = json.loads((output / "training.json").read_text())
    assert metadata["validation_sha256"]


def test_invalid_validation_is_rejected_without_truncation(training_run, tmp_path):
    (tmp_path / "data/validation.jsonl").write_text(
        json.dumps(
            {
                "prompt": "x" * 2048,
                "completion": "code",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="refusing to truncate"):
        training_run("invalid-validation")


@pytest.mark.parametrize("empty", [False, True])
def test_training_without_validation_is_supported(training_run, tmp_path, empty):
    path = tmp_path / "data/validation.jsonl"
    if empty:
        path.write_text("")
    else:
        path.unlink()
    output = training_run(
        "no-validation", "--eval_strategy", "no", "--eval_on_start", "false"
    )
    state = json.loads((output / "trainer_state.json").read_text())
    assert not any("eval_loss" in row for row in state["log_history"])
