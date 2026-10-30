import json
import pytest
from pathlib import Path
from pydantic import ValidationError
from src.ui.session import UI

CASES = json.loads((Path(__file__).parent / "fixtures" / "figures.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_figure_contract(case):
    ui = UI()
    if case["valid"]:
        plot = ui.plot(**case["figure"])
        assert plot.component == "Figure"
        assert len(ui.messages) == 2
    else:
        with pytest.raises(ValidationError):
            ui.plot(**case["figure"])
        assert ui.messages == []
        assert ui.surfaces == set()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["x", "y", "values"])
def test_nonfinite_coordinates_are_rejected(field, value):
    trace = {"type": "pie", "labels": ["A"], "values": [1]} if field == "values" else {"type": "scatter", "x": [1], "y": [1]}
    trace[field] = [value]
    with pytest.raises(ValidationError):
        UI().plot([trace])


@pytest.mark.parametrize(
    ("traces", "points", "valid"), [(1, 200, True), (2, 100, True), (2, 101, False), (20, 1, True), (21, 1, False), (1, 201, False)]
)
def test_figure_limits(traces, points, valid):
    data = [{"type": "scatter", "x": list(range(points)), "y": [1] * points} for _ in range(traces)]
    if valid:
        assert len(UI().plot(data).data) == traces
    else:
        with pytest.raises(ValidationError):
            UI().plot(data)


def test_scatter_modes_and_trace_order_survive_publication():
    ui = UI()
    data = [{"type": "scatter", "mode": mode, "x": [2, 0, 1], "y": [3, None, 1]} for mode in ("lines", "markers", "lines+markers")]
    ui.plot(data)
    published = ui.messages[1]["updateComponents"]["components"][0]["data"]
    assert [trace["mode"] for trace in published] == ["lines", "markers", "lines+markers"]
    assert all(trace["x"] == [2, 0, 1] and trace["y"] == [3, None, 1] for trace in published)
