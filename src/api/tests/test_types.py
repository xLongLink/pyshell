import pytest
from pydantic import TypeAdapter, ValidationError
from api.ui.types import TextValue, Identifier


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
