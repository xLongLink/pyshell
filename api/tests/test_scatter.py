from src.ui.plot.scatter import Scatter


def test_scatter_preserves_mode_coordinate_order_and_gaps():

    # Arrange
    data = {"mode": "lines+markers", "x": [2, 0, 1], "y": [3, None, -1]}

    # Act
    trace = Scatter.model_validate(data)

    # Assert
    assert trace.model_dump() == {"type": "scatter", "name": "", **data}


def test_scatter_defaults_to_lines():

    # Arrange
    data = {"x": [1], "y": [2]}

    # Act
    trace = Scatter.model_validate(data)

    # Assert
    assert trace.mode == "lines"
