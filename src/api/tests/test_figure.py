import pytest
from pydantic import ValidationError
from api.ui.session import UI


@pytest.mark.parametrize("field", ["x", "y", "values"])
def test_nonfinite_coordinates_are_rejected(field):
    trace = {"type": "pie", "labels": ["A"], "values": [1]} if field == "values" else {"type": "scatter", "x": [1], "y": [1]}
    trace[field] = [float("inf")]
    with pytest.raises(ValidationError):
        UI().plot([trace])


@pytest.mark.parametrize(("traces", "points"), [(20, 10), (1, 200)])
def test_figure_limits_accept_boundary_values(traces, points):
    # Arrange
    data = [{"type": "scatter", "x": list(range(points)), "y": [1] * points} for _ in range(traces)]

    # Act
    chart = UI().plot(data)

    # Assert
    assert len(chart.data) == traces


@pytest.mark.parametrize(("traces", "points"), [(21, 1), (1, 201), (2, 101)])
def test_figure_limits_reject_excess_values(traces, points):
    # Arrange
    data = [{"type": "scatter", "x": list(range(points)), "y": [1] * points} for _ in range(traces)]

    # Act
    # Assert
    with pytest.raises(ValidationError):
        UI().plot(data)


def test_scatter_modes_and_trace_order_survive_publication():
    # Arrange
    ui = UI()
    data = [{"type": "scatter", "mode": mode, "x": [2.5, 0, 2.5], "y": [3, None, -1]} for mode in ("lines", "markers", "lines+markers")]

    # Act
    ui.plot(data)

    # Assert
    published = ui.messages[1]["updateComponents"]["components"][0]["data"]
    assert [trace["mode"] for trace in published] == ["lines", "markers", "lines+markers"]
    assert all(trace["x"] == [2.5, 0, 2.5] and trace["y"] == [3, None, -1] for trace in published)
