import json

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText

from model.train.src import checkpoints
from model.train.src.checkpoints import load_resume
from model.train.src.settings import SETTINGS


@pytest.mark.parametrize("initialize", [False, True])
def test_standard_checkpoint_resume_matches_uninterrupted_training(
    training_run, tmp_path, initialize
):
    extra = {}
    if initialize:
        base = AutoModelForImageTextToText.from_pretrained(tmp_path / "base")
        lora = LoraConfig(
            r=4,
            lora_alpha=8,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )
        adapter = get_peft_model(base, lora)
        adapter.save_pretrained(tmp_path / "initial")
        extra = {
            "initialize_adapter": tmp_path / "initial",
            "rank": 4,
            "alpha": 8,
            "target_modules": ("q_proj", "v_proj"),
        }
    full = training_run("full", **extra)
    checkpoint = full / "checkpoint-2"
    metadata = json.loads((checkpoint / "training.json").read_text())
    assert load_resume(checkpoint, metadata) == 2
    resumed = training_run(
        "resumed", **{**extra, "initialize_adapter": None, "resume": checkpoint}
    )
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
    with pytest.raises(ValueError, match="initialize_adapter"):
        load_resume(tmp_path, {})


@pytest.mark.parametrize(
    "field,value",
    [
        ("train_sha256", "changed"),
        ("validation_sha256", "changed"),
        ("base_model", "different"),
        ("base_revision", "different"),
        ("eos_token", "different"),
        ("accumulation", 8),
        ("max_length", 32),
        ("serialization", "raw"),
    ],
)
def test_resume_rejects_changed_provenance(tmp_path, field, value):
    metadata = {
        "train_sha256": "train",
        "validation_sha256": "validation",
        "base_model": "base",
        "base_revision": None,
        "eos_token": "[EOS]",
        "accumulation": 2,
        "max_length": 2048,
        "serialization": "raw-completion-eos-v1",
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
    original = training_run("original")
    initialized = training_run("initialized", initialize_adapter=original)
    assert (initialized / "adapter_model.safetensors").is_file()
    assert (
        json.loads((initialized / "trainer_state.json").read_text())["global_step"] == 4
    )


def test_resume_can_continue_in_the_original_output_directory(training_run):
    output = training_run("original")
    checkpoint = output / "checkpoint-2"
    training_run("original", resume=checkpoint)
    assert json.loads((output / "trainer_state.json").read_text())["global_step"] == 4


def test_completed_budget_cannot_be_resumed(training_run):
    output = training_run("completed")
    with pytest.raises(ValueError, match="budget must follow"):
        training_run("resumed", resume=output / "checkpoint-4")


def test_adapter_settings_cannot_silently_override_the_checked_in_values(training_run):
    output = training_run("original")
    with pytest.raises(ValueError, match="Saved adapter LoRA settings differ"):
        training_run("candidate", initialize_adapter=output, rank=4)


def test_source_identity_includes_uncommitted_source_files(tmp_path, monkeypatch):
    source = tmp_path / "model/train/src"
    source.mkdir(parents=True)
    for filename in ("main.py", "pyproject.toml", "uv.lock"):
        (source.parent / filename).write_text("")
    file = source / "settings.py"
    file.write_text("steps = 1\n")
    monkeypatch.setattr(checkpoints, "__file__", str(source / "checkpoints.py"))
    monkeypatch.setattr(
        checkpoints.subprocess, "check_output", lambda *args, **kwargs: "a" * 40
    )
    before = checkpoints.Checkpoints({}, SETTINGS)
    file.write_text("steps = 2\n")
    after = checkpoints.Checkpoints({}, SETTINGS)
    assert before.run["code_commit"] == after.run["code_commit"]
    assert before.run["source_sha256"] != after.run["source_sha256"]
