import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from model.train.src.settings import SETTINGS, Settings


@pytest.mark.parametrize(
    "field",
    [
        "epochs",
        "rank",
        "alpha",
        "max_length",
        "batch_size",
        "learning_rate",
        "gradient_accumulation_steps",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_training_limits_must_be_positive(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": -1},
        {"dtype": "unknown"},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"base": ""},
        {"data": ""},
        {"target_modules": ""},
        {"target_modules": ("q_proj", "v_proj")},
        {"target_modules": ["q_proj", "v_proj"]},
        {"base_revision": "main"},
        {"base_revision": None},
        {"unknown_setting": True},
    ],
)
def test_invalid_or_unknown_settings_are_rejected(changes):
    with pytest.raises(ValidationError):
        Settings(**changes)


def test_candidate_output_is_separate_from_published_weights():
    assert (
        SETTINGS.output == Path(__file__).resolve().parents[1] / "artifacts/candidate"
    )


@pytest.mark.parametrize(
    "module",
    [
        "self_attn.q_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "linear_attn.in_proj_qkv",
        "linear_attn.out_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
)
def test_lora_targets_attention_and_feed_forward_projections(module):
    assert re.fullmatch(
        SETTINGS.target_modules, f"model.language_model.layers.0.{module}"
    )


@pytest.mark.parametrize(
    "module",
    [
        "model.visual.blocks.0.mlp.up_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.input_layernorm",
        "lm_head",
    ],
)
def test_lora_excludes_other_modules(module):
    assert re.fullmatch(SETTINGS.target_modules, module) is None


def test_default_data_is_repository_local(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    data = Path(Settings().data)
    assert data == Path(__file__).resolve().parents[1] / "data"
    assert (data / "train.jsonl").is_file()
    assert (data / "validation.jsonl").is_file()


def test_checked_in_values_do_not_read_environment(monkeypatch):
    monkeypatch.setenv("LEARNING_RATE", "0.9")
    assert Settings().learning_rate == SETTINGS.learning_rate
    with pytest.raises(ValidationError, match="frozen"):
        SETTINGS.learning_rate = 0.9
