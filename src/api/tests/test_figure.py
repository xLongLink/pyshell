import pytest
from pydantic import ValidationError
from api.ui.session import UI


@pytest.fixture
def standalone_traces(monkeypatch):
    monkeypatch.setattr("api.ui.plot.choropleth._geography_info", lambda name: {"level": "country", "locations": {"CHE": "Switzerland"}})
    return {
        "pie": {"type": "pie", "labels": ["A"], "values": [1]},
        "choropleth": {"type": "choropleth", "locations": ["CHE"], "values": [1]},
        "scatter": {"type": "scatter", "x": [1], "y": [1]},
    }


@pytest.mark.parametrize(
    ("kinds", "message"),
    [
        (("pie", "scatter"), "A pie figure requires exactly one pie trace"),
        (("scatter", "pie"), "A pie figure requires exactly one pie trace"),
        (("choropleth", "scatter"), "A map figure requires exactly one choropleth trace"),
        (("scatter", "choropleth"), "A map figure requires exactly one choropleth trace"),
        (("pie", "choropleth"), "A map figure requires exactly one choropleth trace"),
        (("choropleth", "pie"), "A map figure requires exactly one choropleth trace"),
    ],
)
def test_standalone_traces_reject_mixed_figures(standalone_traces, kinds, message):
    with pytest.raises(ValidationError) as error:
        UI().plot([standalone_traces[kind] for kind in kinds])
    assert error.value.errors()[0]["msg"] == f"Value error, {message}"


@pytest.mark.parametrize(("traces", "points"), [(20, 10), (1, 200)])
def test_figure_limits_accept_boundary_values(traces, points):
    # Arrange
    data = [{"type": "scatter", "x": list(range(points)), "y": [1] * points} for _ in range(traces)]

    # Act
    chart = UI().plot(data)

    # Assert
    assert len(chart.data) == traces


@pytest.mark.parametrize(("traces", "points"), [(21, 1), (2, 101)])
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
    published = ui.state.snapshot()["events"][0]["messages"][1]["updateComponents"]["components"][0]["data"]
    assert [trace["mode"] for trace in published] == ["lines", "markers", "lines+markers"]
    assert all(trace["x"] == [2.5, 0, 2.5] and trace["y"] == [3, None, -1] for trace in published)
