import pytest
from pydantic import ValidationError
from api.ui.session import UI
from api.ui.plot.figure import Figure
from src.api.tests.ui_helpers import RecordingUIState


@pytest.fixture
def ui():
    return UI(RecordingUIState())


@pytest.mark.parametrize(
    "trace",
    [
        {"type": "bar", "name": "Revenue", "x": ["Sales", "Missing"], "y": [-42, None]},
        {"type": "scatter", "mode": "lines", "x": [1, 2], "y": [42, None]},
        {"type": "pie", "name": "Revenue", "labels": ["Sales", "Zero", "Missing"], "values": [42, 0, None]},
        {"type": "choropleth", "locations": ["CHE", "FRA"], "values": [42, None]},
    ],
    ids=["bar", "scatter", "pie", "choropleth"],
)
def test_widget_creation_and_updates_publish_protocol_messages(ui, trace, monkeypatch):
    monkeypatch.setattr(
        "api.ui.plot.choropleth._geography_info", lambda name: {"level": "country", "locations": {"CHE": "Switzerland", "FRA": "France"}}
    )
    chart = ui.plot([trace], id="root", surface_id="sales")
    assert chart.data[0].model_dump(mode="json", include=set(trace)) == trace
    creation = {"version": "v0.9", "createSurface": {"surfaceId": "sales"}}

    def content(widget):
        return {"version": "v0.9", "updateComponents": {"surfaceId": "sales", "components": [widget.model_dump(mode="json")]}}

    assert ui.state.messages == [creation, content(chart)]
    ui.state.messages.clear()
    assert isinstance(chart, Figure)
    assert chart.component == "Figure"
    assert chart.id == "root"
    field = "values" if "values" in trace else "y"
    updated_trace = {**trace, field: [43, *trace[field][1:]]}
    chart.update(data=[updated_trace], layout={"title": {"text": "Updated"}, "showlegend": False})
    assert ui.state.messages == [content(chart)]


def test_surface_replacement_and_removal_publish_protocol_messages(ui):
    trace = {"type": "bar", "x": ["A"], "y": [1]}
    ui.plot([trace], surface_id="sales")
    ui.state.messages.clear()
    creation = {"version": "v0.9", "createSurface": {"surfaceId": "sales"}}
    deletion = {"version": "v0.9", "deleteSurface": {"surfaceId": "sales"}}
    replacement = ui.plot([trace], surface_id="sales")
    content = {"version": "v0.9", "updateComponents": {"surfaceId": "sales", "components": [replacement.model_dump(mode="json")]}}
    assert ui.state.messages == [deletion, creation, content]
    assert ui.state.snapshot()["events"] == [{"type": "a2ui", "messages": [creation, content]}]
    ui.state.messages.clear()
    replacement.remove()
    assert ui.state.messages == [deletion]
    assert ui.state.snapshot()["placements"] == {}


@pytest.mark.parametrize(
    ("operation", "attributes", "message"),
    [
        pytest.param("update", {"data": [{"type": "bar", "x": ["A" * 2000] * 200, "y": [1] * 200}]}, "256 KiB", id="oversized-update"),
        pytest.param("plot", {"data": [{"type": "bar", "x": ["A" * 2000] * 200, "y": [1] * 200}]}, "256 KiB", id="oversized-replacement"),
        pytest.param(
            "plot",
            {"data": [{"type": "bar", "x": ["A"], "y": [1]}], "component": "Unknown"},
            "Input should be 'Figure'",
            id="invalid-replacement",
        ),
    ],
)
def test_rejected_publication_preserves_the_existing_widget(ui, operation, attributes, message):
    chart = ui.plot([{"type": "bar", "x": ["A"], "y": [1]}], layout={"title": {"text": "Original"}}, surface_id="sales")
    messages = list(ui.state.messages)
    snapshot = ui.state.snapshot()
    with pytest.raises(ValueError, match=message):
        if operation == "plot":
            ui.plot(**attributes, surface_id="sales")
        else:
            chart.update(**attributes)
    assert ui.state.messages == messages
    assert ui.state.snapshot() == snapshot
    assert chart.data[0].x == ("A",)
    assert chart.layout.title.text == "Original"
    chart.update(layout={"title": {"text": "Still usable"}})
    assert chart.layout.title.text == "Still usable"


@pytest.mark.parametrize(
    "attributes",
    [
        {},
        {"data": [{"type": "bar", "x": ["A"], "y": [1]}], "unknown": True},
        {"data": [{"type": "bar", "x": ["A"], "y": [1]}], "id": "other"},
        {"data": [{"type": "scatter", "x": [1, 2], "y": [1]}]},
    ],
)
def test_invalid_plot_arguments_do_not_publish(ui, attributes):
    with pytest.raises(ValidationError):
        ui.plot(**attributes)
    assert ui.state.messages == []
    assert ui.state.snapshot()["placements"] == {}
    assert ui.state.snapshot()["events"] == []
    assert ui.plot([{"type": "bar", "x": ["A"], "y": [1]}]).surface_id == "surface_1"


def test_update_keeps_the_handle_and_surface_identity(ui):
    chart = ui.plot(
        [{"type": "scatter", "x": ["Jan"], "y": [42]}], layout={"title": {"text": "Before"}, "showlegend": False}, surface_id="sales"
    )
    chart.update(data=[{"type": "scatter", "x": ["Jan"], "y": [45]}])
    assert chart.layout.title.text == "Before"
    assert chart.layout.showlegend is False
    ui.state.messages.clear()
    chart.update(layout={"title": {"text": "After"}}, data=[{"type": "scatter", "x": ["Jan"], "y": [50]}])
    assert chart.layout.title.text == "After"
    assert chart.data[0].y == (50,)
    assert chart.surface_id == "sales"

    # Updates publish once without replacing the original handle.
    assert len(ui.state.messages) == 1
    assert ui.state.messages[0]["updateComponents"]["surfaceId"] == "sales"
    assert ui.state.messages[0]["updateComponents"]["components"][0]["layout"]["title"]["text"] == "After"
    assert chart.layout.showlegend is True
    chart.update(layout={"title": {"text": "Final"}})
    assert chart.data[0].y == (50,)
    ui.state.messages.clear()
    chart.focus()
    chart.pin()
    assert [message["action"] for message in ui.state.messages] == ["focus", "pin"]


@pytest.mark.parametrize(
    ("attributes", "exhaust_budget", "message"),
    [
        ({"data": [{"type": "pie", "labels": ["A"], "values": [-1]}]}, False, None),
        ({"id": "root"}, False, None),
        ({"component": "Figure"}, False, None),
        ({"unknown": True}, False, None),
        ({"layout": {"yaxis": {"title": {"text": "Amount"}}}}, False, None),
        ({"layout": {"title": {"text": "Not published"}}, "data": [{"type": "scatter", "x": [1, 2], "y": [1]}]}, False, None),
        pytest.param({"layout": {"title": {"text": "Not published"}}}, True, "100 messages", id="publication-budget"),
    ],
)
def test_rejected_update_preserves_attributes_and_published_state(ui, attributes, exhaust_budget, message):
    chart = ui.plot([{"type": "pie", "labels": ["A"], "values": [42]}], layout={"title": {"text": "Original"}})
    if exhaust_budget:
        for _ in range(98):
            chart.pin()
    ui.state.messages.clear()
    snapshot = ui.state.snapshot()
    with pytest.raises(ValueError, match=message):
        chart.update(**attributes)

    # Rejection is atomic and leaves the handle usable for a later valid update.
    assert chart.layout.title.text == "Original"
    assert chart.data[0].values == (42,)
    assert chart.id == "root"
    assert ui.state.messages == []
    assert ui.state.snapshot() == snapshot
    if exhaust_budget:
        ui.reset_output()
    chart.update(layout={"title": {"text": "Valid update"}})
    assert chart.layout.title.text == "Valid update"


@pytest.mark.parametrize("invalidate", ["remove", "replace", "close"])
def test_invalidated_handles_cannot_mutate_published_state(ui, invalidate):
    original = ui.plot([{"type": "bar", "x": ["A"], "y": [1]}], surface_id="notice")

    successor = None
    if invalidate == "close":
        ui.close()
    else:
        if invalidate == "remove":
            original.remove()
        successor = ui.plot([{"type": "scatter", "x": ["A"], "y": [2]}], surface_id="notice")

    previous = list(ui.state.messages)
    snapshot = ui.state.snapshot()
    for operation in (original.pin, original.focus, original.remove, lambda: original.update(layout={"title": {"text": "Stale"}})):
        with pytest.raises(RuntimeError, match="removed, replaced, or reset"):
            operation()
    assert ui.state.messages == previous
    assert ui.state.snapshot() == snapshot
    if successor is None:
        return
    assert successor.data[0].y == (2,)
    successor.update(data=[{"type": "scatter", "x": ["A"], "y": [3]}])
    assert successor.data[0].y == (3,)
    assert ui.state.snapshot()["events"][0]["messages"][-1]["updateComponents"]["components"][0]["data"][0]["y"] == [3]
