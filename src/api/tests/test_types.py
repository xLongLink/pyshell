import pytest
from pydantic import TypeAdapter, ValidationError
from api.ui.types import Action, Number, TextValue, Identifier


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_number_rejects_nonfinite_values(value):
    with pytest.raises(ValidationError, match="Input should be a finite number"):
        TypeAdapter(Number).validate_python(value)


@pytest.mark.parametrize("value", ["a", "Sales_2026-Q4", "a" * 64])
def test_identifier_accepts_supported_characters_and_lengths(value):
    adapter = TypeAdapter(Identifier)
    assert adapter.validate_python(value) == value


@pytest.mark.parametrize("value", ["", "a" * 65, "sales total", "sales/total"])
def test_identifier_rejects_invalid_characters_and_lengths(value):
    adapter = TypeAdapter(Identifier)
    with pytest.raises(ValidationError, match="String should match pattern"):
        adapter.validate_python(value)


def test_text_value_length_boundary():
    adapter = TypeAdapter(TextValue)
    value = "a" * 10000
    assert adapter.validate_python(value) == value
    with pytest.raises(ValidationError, match="at most 10000 characters"):
        adapter.validate_python(value + "a")


@pytest.mark.parametrize("value", ["delete", "unfocus", "unpin"])
def test_action_rejects_unknown_commands(value):
    with pytest.raises(ValidationError, match="Input should be"):
        TypeAdapter(Action).validate_python(value)
