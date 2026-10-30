import json
import pytest
from pathlib import Path
from pydantic import ValidationError
from src.ui.session import UI

CASES = json.loads((Path(__file__).parent / "fixtures" / "figures.json").read_text())
LIMIT_CASES = json.loads((Path(__file__).parent / "fixtures" / "figure_limits.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_figure_contract(case, geography_provider):
    ui = UI()
    if case["valid"]:
        plot = ui.plot(**case["figure"])
        assert plot.component == "Figure"
        assert len(ui.messages) == 2
        if "layout" in case["figure"]:
            assert ui.messages[1]["updateComponents"]["components"][0]["layout"] == case["figure"]["layout"]

    # Rejected contract cases must not create protocol or session state.
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


@pytest.mark.parametrize("case", LIMIT_CASES, ids=[f"{case['traces']}-traces-{case['points']}-points" for case in LIMIT_CASES])
def test_figure_limits(case):
    traces, points = case["traces"], case["points"]
    data = [{"type": "scatter", "x": list(range(points)), "y": [1] * points} for _ in range(traces)]
    if case["valid"]:
        assert len(UI().plot(data).data) == traces
    else:
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
