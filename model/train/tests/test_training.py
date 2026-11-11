import json
import runpy
import sys
from unittest.mock import Mock

import pytest
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from model.train.src import settings, training
from model.train.tests import conftest


def test_training_run_restores_threads_when_setup_fails(
    tmp_path, tokenizer, monkeypatch
):
    threads = torch.get_num_threads()
    from_config = Mock(side_effect=RuntimeError("model setup failed"))
    monkeypatch.setattr(AutoModelForImageTextToText, "from_config", from_config)
    try:
        torch.set_num_threads(2)
        fixture = conftest.training_run.__wrapped__(tmp_path, tokenizer)
        with pytest.raises(RuntimeError, match="model setup failed"):
            next(fixture)
        from_config.assert_called_once()
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(threads)


def test_entrypoint_uses_the_checked_in_settings(monkeypatch):
    train = Mock()
    monkeypatch.setattr(training, "train", train)
    monkeypatch.setattr(sys, "argv", ["model.train.main"])
    runpy.run_module("model.train.main", run_name="__main__")
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


def test_missing_local_training_data_is_rejected(training_run, tmp_path):
    with pytest.raises(FileNotFoundError):
        training_run("missing", data=str(tmp_path / "missing"))


@pytest.mark.parametrize(
    ("batch_size", "accumulation", "steps"), [(1, 2, 2), (3, 1, 2), (3, 3, 1)]
)
def test_one_epoch_uses_every_row_including_partial_batches(
    training_run, tmp_path, monkeypatch, capsys, batch_size, accumulation, steps
):
    seen = []
    training_step = training.SFTTrainer.training_step

    def observe(trainer, model, inputs, *args, **kwargs):
        for tokens, mask in zip(
            inputs["input_ids"], inputs["attention_mask"], strict=True
        ):
            seen.append(tuple(tokens[mask.bool()].tolist()))
        return training_step(trainer, model, inputs, *args, **kwargs)

    monkeypatch.setattr(training.SFTTrainer, "training_step", observe)
    output = training_run(
        "full-epoch", batch_size=batch_size, gradient_accumulation_steps=accumulation
    )
    state = json.loads((output / "trainer_state.json").read_text())
    assert len(seen) == len(set(seen)) == 4
    assert state["global_step"] == steps
    assert state["epoch"] == 1
    assert sum("eval_loss" in row for row in state["log_history"]) == 2
    assert (
        f"Training: 4 rows, 1 epoch(s), effective batch size {batch_size * accumulation}, {steps} optimizer steps"
        in capsys.readouterr().out
    )


def test_training_ignores_foreign_files_and_saves_a_reloadable_adapter(
    training_run, tmp_path
):
    (tmp_path / "data/test-tasks.jsonl").write_text("not a training dataset")
    (tmp_path / "data/train-tasks.jsonl").write_text("not a training dataset")
    output = training_run("candidate", epochs=2, learning_rate=0.002, rank=4, alpha=8)
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 4
    assert state["epoch"] == 2
    adapter = json.loads((output / "adapter_config.json").read_text())
    assert adapter["r"] == 4
    assert adapter["lora_alpha"] == 8
    assert adapter["target_modules"] == settings.SETTINGS.target_modules
    base = AutoModelForImageTextToText.from_pretrained(
        adapter["base_model_name_or_path"]
    )
    model = PeftModel.from_pretrained(base, output)
    tokenizer = AutoTokenizer.from_pretrained(output)
    inputs = tokenizer("prompt", return_tensors="pt")
    with torch.inference_mode():
        prediction = model(**inputs)
    assert torch.isfinite(prediction.logits).all()
