import pytest
from pydantic import TypeAdapter, ValidationError
from src.ui.types import Action, TextValue, Identifier


@pytest.mark.parametrize("value", ["a", "Sales_2026-Q4", "a" * 64])
def test_identifier_accepts_supported_characters_and_lengths(value):
    adapter = TypeAdapter(Identifier)
    assert adapter.validate_python(value) == value


@pytest.mark.parametrize("value", ["", "a" * 65, "sales total", "sales/total"])
def test_identifier_rejects_invalid_characters_and_lengths(value):
    adapter = TypeAdapter(Identifier)
    with pytest.raises(ValidationError, match="String should match pattern"):
        adapter.validate_python(value)


def test_text_value_accepts_the_length_limit():
    value = "a" * 10000
    assert TypeAdapter(TextValue).validate_python(value) == value


def test_text_value_rejects_text_above_the_length_limit():
    value = "a" * 10001
    with pytest.raises(ValidationError, match="at most 10000 characters"):
        TypeAdapter(TextValue).validate_python(value)


@pytest.mark.parametrize("value", ["delete", "unfocus", "unpin"])
def test_action_rejects_unknown_commands(value):
    with pytest.raises(ValidationError, match="Input should be"):
        TypeAdapter(Action).validate_python(value)
