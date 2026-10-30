import pytest
from pydantic import ValidationError
from src.ui.plot.scatter import Scatter


@pytest.mark.parametrize("mode", ["lines", "markers", "lines+markers"])
def test_scatter_preserves_mode_coordinate_order_and_gaps(mode):

    # Arrange
    data = {"mode": mode, "x": [2, 0, 1], "y": [3, None, -1]}

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


def test_scatter_rejects_unsupported_mode():

    # Arrange
    data = {"x": [1], "y": [2], "mode": "dots"}

    # Act / Assert
    with pytest.raises(ValidationError, match="mode"):
        Scatter.model_validate(data)
