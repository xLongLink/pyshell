from src.ui.plot.pie import Pie


def test_pie_preserves_labels_zero_and_missing_shares():

    # Arrange
    data = {"labels": ["Sales", "Zero", "Missing"], "values": [42, 0, None]}

    # Act
    trace = Pie.model_validate(data)

    # Assert
    assert trace.model_dump() == {"type": "pie", "name": "", **data}
