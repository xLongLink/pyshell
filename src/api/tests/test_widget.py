import pytest
from pydantic import ValidationError
from api.ui.session import UI, CATALOG
from a2ui.core.catalog import Catalog, ModelComponentApi
from api.ui.plot.figure import Figure
from a2ui.core.processing import MessageProcessor, MessageProcessorOptions
from a2ui.core.validation import STRICT_VALIDATION


@pytest.mark.parametrize(
    "trace",
    [
        {"type": "bar", "x": ["A"], "y": [42]},
        {"type": "scatter", "x": [0, 1], "y": [42, None]},
        {"type": "pie", "labels": ["A", "B"], "values": [42, None]},
        {"type": "choropleth", "locations": ["CHE"], "values": [42]},
    ],
)
def test_widget_lifecycle_round_trips_through_the_protocol(monkeypatch, trace):
    monkeypatch.setattr("api.ui.plot.choropleth.geography_info", lambda geography: {"locations": {"CHE": "Switzerland"}})
    processor = MessageProcessor(
        [Catalog(CATALOG, "v0.9", [ModelComponentApi(Figure)])], options=MessageProcessorOptions(validation_config=STRICT_VALIDATION)
    )
    ui = UI()
    ui.on_publish = lambda event: processor.process_messages(event["messages"])
    chart = ui.plot([trace], surface_id="sales")
    surface = processor.model.get_surface("sales")
    assert surface.components_model.get("root").component_tree == chart.model_dump(mode="json")
    chart.update(layout={"title": {"text": "Updated"}})
    assert processor.model.get_surface("sales") is surface
    assert surface.components_model.get("root").component_tree == chart.model_dump(mode="json")
    replacement = ui.plot([trace], surface_id="sales")
    assert processor.model.get_surface("sales") is not surface
    assert processor.model.get_surface("sales").components_model.get("root").component_tree == replacement.model_dump(mode="json")
    replacement.remove()
    assert processor.model.get_surface("sales") is None
    assert ui.surfaces == set()


@pytest.mark.parametrize("replace", [False, True])
def test_surface_size_limit_preserves_the_existing_widget(replace):
    ui = UI()
    chart = ui.plot([{"type": "bar", "x": ["A"], "y": [1]}], surface_id="sales")
    messages = list(ui.messages)
    data = [{"type": "bar", "x": ["A" * 2000] * 200, "y": [1] * 200}]
    publish = (lambda: ui.plot(data, surface_id="sales")) if replace else lambda: chart.update(data=data)
    with pytest.raises(ValueError, match="256 KiB"):
        publish()
    assert ui.messages == messages
    assert chart.data[0].x == ["A"]
    chart.update(layout={"title": {"text": "Still usable"}})
    assert chart.layout.title.text == "Still usable"


def test_plot_returns_typed_widgets_with_common_methods():
    ui = UI()
    widget = ui.plot(data=[{"type": "bar", "x": ["A"], "y": [1]}])
    assert isinstance(widget, Figure)
    assert widget.component == "Figure"
    assert widget.surface_id == "surface_1"

    # All typed widgets use the same ordered placement commands.
    ui.messages = []
    widget.focus()
    widget.pin()
    assert [message["action"] for message in ui.messages] == ["focus", "pin"]
    widget.remove()
    assert ui.messages[-1] == {"version": "v0.9", "deleteSurface": {"surfaceId": "surface_1"}}


def test_standalone_figure_requires_the_root_component_id():
    data = [{"type": "bar", "x": ["A"], "y": [1]}]
    assert Figure(data=data).id == "root"
    assert Figure(id="root", data=data).id == "root"
    with pytest.raises(ValidationError, match="Input should be 'root'"):
        Figure(id="other", data=data)


@pytest.mark.parametrize(
    "attributes",
    [
        {},
        {"data": [{"type": "bar", "x": ["A"], "y": [1]}], "unknown": True},
        {"data": [{"type": "bar", "x": ["A"], "y": [1]}], "id": "other"},
    ],
)
def test_invalid_plot_arguments_do_not_publish(attributes):
    ui = UI()
    events = []
    ui.on_publish = events.append
    with pytest.raises(ValidationError):
        ui.plot(**attributes)
    assert ui.messages == []
    assert ui.surfaces == set()
    assert events == []
    assert ui.plot([{"type": "bar", "x": ["A"], "y": [1]}]).surface_id == "surface_1"


def test_update_keeps_the_handle_and_surface_identity():
    ui = UI()
    chart = ui.plot(
        [{"type": "scatter", "x": ["Jan"], "y": [42]}], layout={"title": {"text": "Before"}, "showlegend": False}, surface_id="sales"
    )
    chart.update(data=[{"type": "scatter", "x": ["Jan"], "y": [45]}])
    assert chart.layout.title.text == "Before"
    assert chart.layout.showlegend is False
    ui.messages = []
    chart.update(layout={"title": {"text": "After"}}, data=[{"type": "scatter", "x": ["Jan"], "y": [50]}])
    assert chart.layout.title.text == "After"
    assert chart.data[0].y == [50]
    assert chart.surface_id == "sales"

    # Updates publish once without replacing the original handle.
    assert len(ui.messages) == 1
    assert ui.messages[0]["updateComponents"]["surfaceId"] == "sales"
    assert ui.messages[0]["updateComponents"]["components"][0]["layout"]["title"]["text"] == "After"
    assert chart.layout.showlegend is True
    chart.update(layout={"title": {"text": "Final"}})
    assert chart.data[0].y == [50]
    chart.pin()
    assert ui.messages[-1]["action"] == "pin"


@pytest.mark.parametrize(
    "attributes",
    [
        {"data": [{"type": "pie", "labels": ["A"], "values": [-1]}]},
        {"id": "other"},
        {"id": "root"},
        {"component": "Other"},
        {"component": "Figure"},
        {"unknown": True},
        {"layout": {"yaxis": {"title": {"text": "Amount"}}}},
        {"layout": {"title": {"text": "Not published"}}, "data": [{"type": "scatter", "x": [1, 2], "y": [1]}]},
    ],
)
def test_rejected_update_preserves_attributes_and_published_state(attributes):
    ui = UI()
    chart = ui.plot([{"type": "pie", "labels": ["A"], "values": [42]}], layout={"title": {"text": "Original"}})
    ui.messages = []
    with pytest.raises((ValueError, ValidationError)):
        chart.update(**attributes)

    # Rejection is atomic and leaves the handle usable for a later valid update.
    assert chart.layout.title.text == "Original"
    assert chart.data[0].values == [42]
    assert chart.id == "root"
    assert ui.messages == []
    chart.update(layout={"title": {"text": "Valid update"}})
    assert chart.layout.title.text == "Valid update"


def test_widget_attributes_cannot_be_assigned_without_publishing():
    ui = UI()
    widget = ui.plot([{"type": "bar", "x": ["A"], "y": [1]}], layout={"title": {"text": "Original"}})
    with pytest.raises(ValidationError, match="frozen"):
        widget.layout = {"title": {"text": "Unpublished"}}

    # Nested models must not bypass publication through direct assignment.
    with pytest.raises(ValidationError, match="frozen"):
        widget.layout.title.text = "Unpublished"
    assert widget.layout.title.text == "Original"
    assert len(ui.messages) == 2


@pytest.mark.parametrize("invalidate", ["remove", "replace", "reset"])
def test_removed_replaced_and_reset_handles_cannot_mutate_new_widgets(invalidate):
    ui = UI()
    original = ui.plot([{"type": "bar", "x": ["A"], "y": [1]}], surface_id="notice")

    # Each invalidation path must reject mutations through the old handle.
    if invalidate == "remove":
        original.remove()
    elif invalidate == "replace":
        ui.plot([{"type": "scatter", "x": ["A"], "y": [2]}], surface_id="notice")
    else:
        ui.close()

    # Stale operations must not append messages to the replacement session.
    previous = list(ui.messages)
    for operation in (original.pin, original.focus, original.remove, lambda: original.update(layout={"title": {"text": "Stale"}})):
        with pytest.raises(RuntimeError, match="removed, replaced, or reset"):
            operation()
    assert ui.messages == previous


@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        ({"data": [{"type": "bar", "x": ["A"], "y": [1]}], "component": "Unknown"}, "Input should be 'Figure'"),
        ({"data": [{"type": "pie", "labels": ["A"], "values": [-1]}]}, "greater than or equal to 0"),
        ({"id": "other", "data": [{"type": "bar", "x": ["A"], "y": [1]}]}, "Input should be 'root'"),
        ({"data": [{"type": "bar", "x": ["A"], "y": [1]}], "layout": {"yaxis": {"title": {"text": "Amount"}}}}, "Extra inputs"),
    ],
)
def test_invalid_replacement_does_not_destroy_the_existing_surface(attributes, message):
    ui = UI()
    original = ui.plot([{"type": "bar", "x": ["A"], "y": [1]}], layout={"title": {"text": "Original"}}, surface_id="notice")
    ui.messages = []
    events = []
    ui.on_publish = events.append
    with pytest.raises(ValueError, match=message):
        ui.plot(**attributes, surface_id="notice")
    assert ui.messages == []
    assert events == []
    assert ui.surfaces == {"notice"}
    assert original.layout.title.text == "Original"
    original.update(layout={"title": {"text": "Still usable"}})
    assert original.layout.title.text == "Still usable"


def test_failed_publication_does_not_change_widget_attributes():
    ui = UI()
    widget = ui.plot([{"type": "bar", "x": ["A"], "y": [1]}], layout={"title": {"text": "Original"}})
    for _ in range(98):
        widget.pin()
    with pytest.raises(ValueError, match="100 messages"):
        widget.update(layout={"title": {"text": "Not published"}})
    assert widget.layout.title.text == "Original"

    # Once the output budget is cleared, the same handle can publish again.
    ui.messages = []
    widget.update(layout={"title": {"text": "Now published"}})
    assert widget.layout.title.text == "Now published"
