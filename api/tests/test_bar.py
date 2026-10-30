from src.ui.plot.bar import Bar


def test_bar_preserves_categories_negative_values_and_gaps():

    # Arrange
    data = {"name": "Revenue", "x": ["Profit", "Loss", "Missing"], "y": [42, -5, None]}

    # Act
    trace = Bar.model_validate(data)

    # Assert
    assert trace.model_dump() == {"type": "bar", **data}
