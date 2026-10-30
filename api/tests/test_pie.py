import pytest
from pydantic import ValidationError
from src.ui.plot.pie import Pie


def test_pie_preserves_labels_zero_and_missing_shares():

    # Arrange
    data = {"labels": ["Sales", "Zero", "Missing"], "values": [42, 0, None]}

    # Act
    trace = Pie.model_validate(data)

    # Assert
    assert trace.model_dump() == {"type": "pie", "name": "", **data}


def test_pie_rejects_negative_shares():

    # Arrange
    data = {"labels": ["Loss"], "values": [-1]}

    # Act / Assert
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        Pie.model_validate(data)


def test_pie_requires_a_label_for_each_value():

    # Arrange
    data = {"labels": ["Sales"], "values": [1, 2]}

    # Act / Assert
    with pytest.raises(ValidationError, match="labels and values must have equal lengths"):
        Pie.model_validate(data)
