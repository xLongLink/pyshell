from pathlib import Path

import pytest
from pydantic import ValidationError

from model.train.src.settings import SETTINGS, Settings


@pytest.mark.parametrize(
    "field",
    [
        "steps",
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
        {"data_revision": "main"},
        {"base_revision": None},
        {"data_revision": None},
        {"completion_only_loss": False},
        {"resume": Path("checkpoint")},
        {"initialize_adapter": Path("adapter")},
        {"save_steps": 25},
    ],
)
def test_invalid_or_unknown_settings_are_rejected(changes):
    with pytest.raises(ValidationError):
        Settings(**changes)


def test_candidate_output_is_separate_from_published_weights():
    assert (
        SETTINGS.output == Path(__file__).resolve().parents[1] / "artifacts/candidate"
    )


def test_checked_in_values_do_not_read_environment(monkeypatch):
    monkeypatch.setenv("LEARNING_RATE", "0.9")
    assert Settings().learning_rate == SETTINGS.learning_rate
    with pytest.raises(ValidationError, match="frozen"):
        SETTINGS.learning_rate = 0.9
