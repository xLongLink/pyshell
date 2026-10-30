from src.ui.plot.common import CartesianTrace


def test_cartesian_trace_preserves_coordinate_pairs():

    # Arrange
    data = {"x": [2, 0, 2], "y": [-1, None, 4]}

    # Act
    trace = CartesianTrace.model_validate(data)

    # Assert
    assert trace.model_dump() == {"name": "", **data}
