import json

import pytest
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

from model.train.src.checkpoints import load_resume


def test_standard_checkpoint_resume_matches_uninterrupted_training(training_run):
    full = training_run("full")
    paused = training_run("paused", "--stop-after", "4")
    checkpoint = paused / "checkpoint-2"
    metadata = json.loads((checkpoint / "training.json").read_text())
    assert load_resume(checkpoint, metadata) == 2
    resumed = training_run("resumed", "--resume", str(checkpoint))
    assert json.loads((resumed / "trainer_state.json").read_text())["global_step"] == 4
    base = AutoModelForImageTextToText.from_pretrained(metadata["base_model"])
    full_model = PeftModel.from_pretrained(base, full)
    full_state = {
        name: parameter.detach().clone()
        for name, parameter in full_model.named_parameters()
        if "lora_" in name
    }
    base = AutoModelForImageTextToText.from_pretrained(metadata["base_model"])
    resumed_model = PeftModel.from_pretrained(base, resumed)
    resumed_state = {
        name: parameter.detach()
        for name, parameter in resumed_model.named_parameters()
        if "lora_" in name
    }
    assert full_state.keys() == resumed_state.keys()
    assert all(
        torch.equal(full_state[name], resumed_state[name]) for name in full_state
    )


def test_legacy_adapters_require_initialization_not_resume(tmp_path):
    (tmp_path / "training.json").write_text('{"serialization": "raw"}')
    with pytest.raises(ValueError, match="--initialize-adapter"):
        load_resume(tmp_path, {})


@pytest.mark.parametrize(
    "field,value",
    [
        ("train_sha256", "changed"),
        ("validation_sha256", "changed"),
        ("base_model", "different"),
        ("accumulation", 8),
        ("max_length", 32),
        ("serialization", "chat"),
    ],
)
def test_resume_rejects_changed_provenance(tmp_path, field, value):
    metadata = {
        "train_sha256": "train",
        "validation_sha256": "validation",
        "base_model": "base",
        "accumulation": 2,
        "max_length": 2048,
        "serialization": "raw",
    }
    (tmp_path / "trainer_state.json").write_text('{"global_step": 2}')
    (tmp_path / "training.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        load_resume(tmp_path, {**metadata, field: value})


@pytest.mark.parametrize("missing", ["optimizer.pt", "scheduler.pt", "rng_state.pth"])
def test_resume_rejects_incomplete_checkpoints(tmp_path, missing):
    (tmp_path / "trainer_state.json").write_text('{"global_step": 2}')
    (tmp_path / "training.json").write_text("{}")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        if name != missing:
            (tmp_path / name).touch()
    with pytest.raises(ValueError, match=f"missing {missing}"):
        load_resume(tmp_path, {})


def test_existing_adapter_can_initialize_a_new_run(training_run):
    original = training_run("original", "--stop-after", "4")
    initialized = training_run("initialized", "--initialize-adapter", str(original))
    assert (initialized / "adapter_model.safetensors").is_file()
    assert (
        json.loads((initialized / "trainer_state.json").read_text())["global_step"] == 4
    )
