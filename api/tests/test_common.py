import pytest
from pydantic import ValidationError
from src.ui.plot.common import CartesianTrace


def test_cartesian_trace_preserves_coordinate_pairs():

    # Arrange
    data = {"x": [2, 0, 2], "y": [-1, None, 4]}

    # Act
    trace = CartesianTrace.model_validate(data)

    # Assert
    assert trace.model_dump() == {"name": "", **data}


def test_cartesian_trace_requires_equal_coordinate_lengths():

    # Arrange
    data = {"x": [1, 2], "y": [3]}

    # Act / Assert
    with pytest.raises(ValidationError, match="x and y must have equal lengths"):
        CartesianTrace.model_validate(data)


def test_plot_models_reject_unknown_fields():

    # Arrange
    data = {"x": [1], "y": [2], "html": "<script>"}

    # Act / Assert
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CartesianTrace.model_validate(data)


def test_plot_models_are_frozen():

    # Arrange
    trace = CartesianTrace(x=[1], y=[2])

    # Act / Assert
    with pytest.raises(ValidationError, match="frozen"):
        trace.name = "Unpublished"
    assert trace.name == ""
