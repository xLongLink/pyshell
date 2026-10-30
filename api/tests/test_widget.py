import pytest
from pydantic import ValidationError
from src.ui.widget import Widget
from src.ui.session import UI
from src.ui.plot.figure import Figure


@pytest.mark.parametrize(
    "data",
    [
        [{"type": "bar", "x": ["A"], "y": [1]}],
        [{"type": "scatter", "x": ["A"], "y": [1]}],
        [{"type": "pie", "labels": ["A"], "values": [1]}],
    ],
)
def test_plot_returns_typed_widgets_with_common_methods(data):
    ui = UI()
    widget = ui.plot(data=data)
    assert isinstance(widget, Widget)
    assert widget.component == "Figure"
    assert widget.surface_id == "surface_1"

    # All typed widgets use the same ordered placement commands.
    ui.messages = []
    widget.focus()
    widget.unfocus()
    widget.pin()
    widget.unpin()
    assert [message["action"] for message in ui.messages] == ["focus", "unfocus", "pin", "unpin"]
    widget.remove()
    assert ui.messages[-1] == {"version": "v0.9", "deleteSurface": {"surfaceId": "surface_1"}}


def test_show_publishes_a_typed_handle_once():
    ui = UI()
    figure = Figure(data=[{"type": "bar", "x": ["A"], "y": [1]}])
    assert ui.show(figure, surface_id="sales") is figure
    assert figure.surface_id == "sales"
    previous = list(ui.messages)
    with pytest.raises(RuntimeError, match=r"Use update\(\)"):
        ui.show(figure, surface_id="other")
    assert ui.messages == previous
    assert ui.surfaces == {"sales"}
    figure.pin()
    assert ui.messages[-1] == {"type": "widget", "surface_id": "sales", "action": "pin"}


def test_plot_rejects_duplicate_or_excess_positional_data_before_publication():
    ui = UI()
    data = [{"type": "bar", "x": ["A"], "y": [1]}]
    with pytest.raises(TypeError, match="Multiple values for 'data'"):
        ui.plot(data, data=data)
    with pytest.raises(TypeError, match="at most 1 positional"):
        ui.plot(data, data)
    assert ui.messages == []
    assert ui.surfaces == set()


@pytest.mark.parametrize("attributes", [{}, {"data": [{"type": "bar", "x": ["A"], "y": [1]}], "unknown": True}])
def test_invalid_plot_arguments_do_not_publish(attributes):
    ui = UI()
    with pytest.raises(ValidationError):
        ui.plot(**attributes)
    assert ui.messages == []
    assert ui.surfaces == set()


def test_update_keeps_the_handle_and_surface_identity():
    ui = UI()
    chart = ui.plot([{"type": "scatter", "x": ["Jan"], "y": [42]}], layout={"title": {"text": "Before"}}, surface_id="sales")
    ui.messages = []
    chart.update(layout={"title": {"text": "After"}}, data=[{"type": "scatter", "x": ["Jan"], "y": [50]}])
    assert chart.layout.title.text == "After"
    assert chart.data[0].y == [50]
    assert chart.surface_id == "sales"

    # Updates publish once without replacing the original handle.
    assert len(ui.messages) == 1
    assert ui.messages[0]["updateComponents"]["surfaceId"] == "sales"
    assert ui.messages[0]["updateComponents"]["components"][0]["layout"]["title"]["text"] == "After"
    chart.pin()
    assert ui.messages[-1]["action"] == "pin"


@pytest.mark.parametrize(
    "attributes",
    [
        {"data": [{"type": "pie", "labels": ["A"], "values": [-1]}]},
        {"id": "other"},
        {"component": "Other"},
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
        ({"id": "other", "data": [{"type": "bar", "x": ["A"], "y": [1]}]}, "component ID 'root'"),
        ({"data": [{"type": "bar", "x": ["A"], "y": [1]}], "extra": True}, "Extra inputs"),
        ({"data": [{"type": "bar", "x": ["A"], "y": [1]}], "layout": {"yaxis": {"title": {"text": "Amount"}}}}, "Extra inputs"),
    ],
)
def test_invalid_replacement_does_not_destroy_the_existing_surface(attributes, message):
    ui = UI()
    original = ui.plot([{"type": "bar", "x": ["A"], "y": [1]}], layout={"title": {"text": "Original"}}, surface_id="notice")
    ui.messages = []
    with pytest.raises(ValueError, match=message):
        ui.plot(**attributes, surface_id="notice")
    assert ui.messages == []
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


@pytest.mark.parametrize("name", ["chart", "text", "metric", "table", "column", "row", "card"])
def test_unsupported_widget_constructors_are_not_exposed(name):
    ui = UI()
    assert not hasattr(ui, name)


def test_update_replaces_supplied_fields_and_keeps_omitted_fields():
    ui = UI()
    plot = ui.plot([{"type": "scatter", "x": [1], "y": [2]}], layout={"title": {"text": "Before"}, "showlegend": False})
    plot.update(data=[{"type": "scatter", "x": [1, 2], "y": [3, 4]}])
    assert plot.layout.title.text == "Before"
    assert plot.layout.showlegend is False
    plot.update(layout={"title": {"text": "After"}})
    assert plot.layout.showlegend is True
    assert plot.data[0].y == [3, 4]
