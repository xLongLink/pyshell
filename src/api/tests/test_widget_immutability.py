import json
import pytest
from copy import deepcopy
from operator import setitem
from pydantic import ValidationError
from api.ui.session import UI
from api.ui.plot.figure import Figure
from src.api.tests.ui_helpers import RecordingUIState


@pytest.fixture(params=["bar", "scatter", "pie", "choropleth"])
def trace(request, monkeypatch):
    monkeypatch.setattr(
        "api.ui.plot.choropleth._geography_info", lambda name: {"level": "country", "locations": {"CHE": "Switzerland", "FRA": "France"}}
    )
    kind = request.param
    if kind == "pie":
        return {"type": kind, "labels": ["A", "B"], "values": [1, None]}
    if kind == "choropleth":
        return {"type": kind, "locations": ["CHE", "FRA"], "values": [1, None]}
    return {"type": kind, "x": [1, 2.5], "y": [1, None]}


def test_published_attributes_and_sequences_cannot_be_mutated(trace):
    ui = UI(RecordingUIState())
    chart = ui.plot([trace], layout={"title": {"text": "Original"}})
    before = chart.model_dump(mode="json")
    snapshot = ui.state.snapshot()
    sequences = [chart.data, *(getattr(chart.data[0], name) for name, value in trace.items() if isinstance(value, list))]
    for sequence in sequences:
        with pytest.raises(TypeError):
            setitem(sequence, 0, sequence[0])
    with pytest.raises(ValidationError, match="frozen"):
        chart.data[0].name = "Unpublished"
    with pytest.raises(ValidationError, match="frozen"):
        chart.layout = {"title": {"text": "Unpublished"}}
    with pytest.raises(ValidationError, match="frozen"):
        chart.layout.title.text = "Unpublished"
    assert chart.model_dump(mode="json") == before
    assert ui.state.snapshot() == snapshot
    assert len(ui.state.messages) == 2


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
    attributes = {"data": [trace]}
    for chart in (Figure.model_validate(attributes), Figure.model_validate_json(json.dumps(attributes))):
        assert isinstance(chart.data, tuple)
        for name, value in trace.items():
            if isinstance(value, list):
                assert getattr(chart.data[0], name) == tuple(value)
        assert Figure.model_validate(chart.model_dump()).model_dump(mode="json") == chart.model_dump(mode="json")
        assert json.loads(chart.model_dump_json()) == chart.model_dump(mode="json")

    ui = UI(RecordingUIState())
    chart = ui.plot([trace])
    replacement = deepcopy(trace)
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
    ("trace", "container"),
    [("bar", tuple), ("bar", set), ("bar", iter), ("scatter", tuple), ("pie", tuple), ("choropleth", tuple)],
    indirect=["trace"],
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
    ("trace", "value"),
    [("bar", value) for value in [True, float("inf"), float("nan"), {}, []]] + [(kind, True) for kind in ["scatter", "pie", "choropleth"]],
    indirect=["trace"],
)
@pytest.mark.parametrize("json_input", [False, True])
def test_immutable_sequences_preserve_strict_scalar_validation(trace, value, json_input):
    validate = Figure.model_validate_json if json_input else Figure.model_validate
    for name, coordinates in trace.items():
        if isinstance(coordinates, list):
            attributes = {"data": [{**trace, name: [value, coordinates[1]]}]}
            candidate = json.dumps(attributes) if json_input else attributes
            with pytest.raises(ValidationError) as error:
                validate(candidate)
            assert any(name in item["loc"] for item in error.value.errors())


@pytest.mark.parametrize("length", [0, 201])
def test_every_trace_sequence_retains_its_length_bounds(trace, length):
    for name, coordinates in trace.items():
        if isinstance(coordinates, list):
            with pytest.raises(ValidationError) as error:
                Figure.model_validate({"data": [{**trace, name: [coordinates[0]] * length}]})
            assert any(item["loc"][-1] == name and item["type"] in {"too_short", "too_long"} for item in error.value.errors())


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_sequence_schema_contracts_remain_bounded_arrays(mode):
    schema = Figure.model_json_schema(mode=mode)
    for definition, names, maximum in [
        (schema, ["data"], 20),
        (schema["$defs"]["Scatter"], ["x", "y"], 200),
        (schema["$defs"]["Bar"], ["x", "y"], 200),
        (schema["$defs"]["Pie"], ["labels", "values"], 200),
        (schema["$defs"]["Choropleth"], ["locations", "values"], 200),
    ]:
        for name in names:
            field = definition["properties"][name]
            assert (field["type"], field["minItems"], field["maxItems"]) == ("array", 1, maximum)
