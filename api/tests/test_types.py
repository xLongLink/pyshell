import pytest
from pydantic import TypeAdapter, ValidationError
from src.ui.types import Action, Number, TextValue, Identifier


@pytest.mark.parametrize("value", ["a", "Sales_2026-Q4", "a" * 64])
def test_identifier_accepts_supported_characters_and_lengths(value):

    # Arrange
    adapter = TypeAdapter(Identifier)

    # Act / Assert
    assert adapter.validate_python(value) == value


@pytest.mark.parametrize("value", ["", "a" * 65, "sales total", "sales/total"])
def test_identifier_rejects_invalid_characters_and_lengths(value):

    # Arrange
    adapter = TypeAdapter(Identifier)

    # Act / Assert
    with pytest.raises(ValidationError, match="String should match pattern"):
        adapter.validate_python(value)


def test_text_value_accepts_the_length_limit():

    # Arrange
    value = "a" * 10000

    # Act / Assert
    assert TypeAdapter(TextValue).validate_python(value) == value


def test_text_value_rejects_text_above_the_length_limit():

    # Arrange
    value = "a" * 10001

    # Act / Assert
    with pytest.raises(ValidationError, match="at most 10000 characters"):
        TypeAdapter(TextValue).validate_python(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_number_rejects_nonfinite_values(value):

    # Act / Assert
    with pytest.raises(ValidationError, match="finite number"):
        TypeAdapter(Number).validate_python(value)


@pytest.mark.parametrize("value", [-42, 0, 1.5])
def test_number_accepts_finite_values(value):

    # Act / Assert
    assert TypeAdapter(Number).validate_python(value) == value


@pytest.mark.parametrize("value", ["focus", "unfocus", "pin", "unpin"])
def test_action_accepts_placement_commands(value):

    # Act / Assert
    assert TypeAdapter(Action).validate_python(value) == value


def test_action_rejects_unknown_commands():

    # Act / Assert
    with pytest.raises(ValidationError, match="Input should be"):
        TypeAdapter(Action).validate_python("delete")
