import json
import pytest
from copy import deepcopy
from operator import setitem
from pydantic import ValidationError
from api.ui.session import UI
from api.ui.plot.figure import Figure
from src.api.tests.ui_helpers import RecordingUIState


@pytest.fixture(params=["bar", "scatter", "pie", "choropleth"])
def trace(request, country_geography):
    kind = request.param
    if kind == "pie":
        return {"type": kind, "labels": ["A", "B"], "values": [1, None]}
    if kind == "choropleth":
        return {"type": kind, "locations": ["CHE", "FRA"], "values": [1, None]}
    return {"type": kind, "x": [1, 2.5], "y": [1, None]}


def test_published_attributes_and_sequences_cannot_be_mutated(trace):
    ui = UI(RecordingUIState())
    chart = ui.plot([trace])
    before = chart.model_dump(mode="json")
    snapshot = ui.state.snapshot()
    sequences = [chart.data, *(getattr(chart.data[0], name) for name, value in trace.items() if isinstance(value, list))]
    for sequence in sequences:
        with pytest.raises(TypeError):
            setitem(sequence, 0, sequence[0])
    with pytest.raises(ValidationError, match="frozen"):
        chart.data[0].name = "Unpublished"
    assert chart.model_dump(mode="json") == before
    assert ui.state.snapshot() == snapshot
    assert len(ui.state.messages) == 2


def test_layout_and_nested_title_cannot_be_mutated():
    chart = Figure.model_validate({"data": [{"type": "bar", "x": ["A"], "y": [1]}], "layout": {"title": {"text": "Original"}}})
    before = chart.model_dump(mode="json")
    with pytest.raises(ValidationError, match="frozen"):
        chart.layout = {"title": {"text": "Unpublished"}}
    with pytest.raises(ValidationError, match="frozen"):
        chart.layout.title.text = "Unpublished"
    assert chart.model_dump(mode="json") == before


def test_original_lists_cannot_change_published_or_subsequently_updated_content(trace):
    ui = UI(RecordingUIState())
    data = [trace]
    chart = ui.plot(data)
    before = chart.model_dump(mode="json")
    snapshot = ui.state.snapshot()
    for value in trace.values():
        if isinstance(value, list):
            value.clear()
    trace["name"] = "Unpublished"
    data.clear()
    assert chart.model_dump(mode="json") == before
    assert ui.state.snapshot() == snapshot
    chart.update(layout={"title": {"text": "Updated"}})
    assert chart.model_dump(mode="json")["data"] == before["data"]
    assert ui.state.messages[-1]["updateComponents"]["components"][0]["data"] == before["data"]


def test_list_inputs_round_trip_as_arrays_and_explicit_updates_remain_immutable(trace):
    ui = UI(RecordingUIState())
    chart = ui.plot([trace])
    assert Figure.model_validate(chart.model_dump()).model_dump(mode="json") == chart.model_dump(mode="json")
    assert json.loads(chart.model_dump_json()) == chart.model_dump(mode="json")

    replacement = trace.copy()
    field = "y" if "y" in replacement else "values"
    replacement[field] = [3, None]
    chart.update(data=[replacement])
    published = deepcopy(ui.state.messages[-1]["updateComponents"]["components"][0])
    assert isinstance(published["data"], list)
    assert published["data"][0][field] == [3, None]
    assert getattr(chart.data[0], field) == (3, None)
    replacement[field][0] = 99
    assert chart.model_dump(mode="json") == published
    assert ui.state.messages[-1]["updateComponents"]["components"][0] == published


@pytest.mark.parametrize(
    ("trace", "container"), [("bar", tuple), ("bar", set), ("bar", iter), ("pie", tuple), ("choropleth", tuple)], indirect=["trace"]
)
def test_nonlist_inputs_are_rejected_atomically_for_every_sequence(trace, container):
    ui = UI(RecordingUIState())
    chart = ui.plot([trace])
    before = chart.model_dump(mode="json")
    snapshot = ui.state.snapshot()
    candidates = [{"data": container([trace])}] if container is not set else [{"data": set()}]
    for name, value in trace.items():
        if isinstance(value, list):
            candidates.append({"data": [{**trace, name: container(value)}]})
    for attributes in candidates:
        with pytest.raises(ValidationError, match="valid list"):
            chart.update(**attributes)
        assert chart.model_dump(mode="json") == before
        assert ui.state.snapshot() == snapshot
    assert len(ui.state.messages) == 2


@pytest.mark.parametrize(
    ("trace", "value"), [("bar", True), ("bar", float("inf")), ("pie", True), ("choropleth", True)], indirect=["trace"]
)
def test_immutable_sequences_preserve_strict_scalar_validation(trace, value):
    for name, coordinates in trace.items():
        if isinstance(coordinates, list):
            attributes = {"data": [{**trace, name: [value, coordinates[1]]}]}
            with pytest.raises(ValidationError) as error:
                Figure.model_validate(attributes)
            assert any(name in item["loc"] for item in error.value.errors())


@pytest.mark.parametrize("length", [0, 201])
@pytest.mark.parametrize("trace", ["bar", "pie", "choropleth"], indirect=True)
def test_every_trace_sequence_retains_its_length_bounds(trace, length):
    for name, coordinates in trace.items():
        if isinstance(coordinates, list):
            with pytest.raises(ValidationError) as error:
                Figure.model_validate({"data": [{**trace, name: [coordinates[0]] * length}]})
            assert any(item["loc"][-1] == name and item["type"] in {"too_short", "too_long"} for item in error.value.errors())
