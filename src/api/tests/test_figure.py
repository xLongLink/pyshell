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
    ],
)
def test_standalone_traces_reject_mixed_figures(standalone_traces, kinds, message):
    with pytest.raises(ValidationError) as error:
        UI().plot([standalone_traces[kind] for kind in kinds])
    assert error.value.errors()[0]["msg"] == f"Value error, {message}"


@pytest.mark.parametrize(("traces", "points", "valid"), [(20, 10, True), (1, 200, True), (21, 1, False), (2, 101, False)])
def test_figure_limits_enforce_boundaries(traces, points, valid):
    data = [{"type": "scatter", "x": list(range(points)), "y": [1] * points} for _ in range(traces)]
    if not valid:
        with pytest.raises(ValidationError):
            UI().plot(data)
        return
    chart = UI().plot(data)
    assert len(chart.data) == traces


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


@pytest.mark.parametrize("coordinates", [([1], ["A"]), (["A"], [1]), ([1, "A"], [2])])
def test_mixed_cartesian_coordinate_types_are_rejected_before_publication(coordinates):
    ui = UI()
    data = [{"type": kind, "x": values, "y": [1] * len(values)} for kind, values in zip(("bar", "scatter"), coordinates)]

    with pytest.raises(ValidationError, match="All x coordinates must be numeric or all categorical"):
        ui.plot(data)

    assert ui.state.snapshot()["events"] == []
