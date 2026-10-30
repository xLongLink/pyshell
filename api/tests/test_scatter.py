from src.ui.plot.scatter import Scatter


def test_scatter_defaults_to_lines():

    # Arrange
    data = {"x": [1], "y": [2]}

    # Act
    trace = Scatter.model_validate(data)

    # Assert
    assert trace.mode == "lines"
