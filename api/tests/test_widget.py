import pytest
from pydantic import ValidationError
from src.ui.widget import Widget
from src.ui.session import UI


@pytest.mark.parametrize(
    ("path", "attributes", "component"),
    [
        ("chart.bar", {"points": [{"label": "A", "value": 1}]}, "BarChart"),
        ("chart.line", {"points": [{"label": "A", "value": 1}]}, "LineChart"),
        ("chart.pie", {"points": [{"label": "A", "value": 1}]}, "PieChart"),
    ],
)
def test_factories_return_typed_widgets_with_common_methods(path, attributes, component):
    ui = UI()
    factory = ui
    for name in path.split("."):
        factory = getattr(factory, name)
    widget = factory(**attributes)
    assert isinstance(widget, Widget)
    assert widget.component == component
    assert widget.surface_id == "surface_1"
    ui.messages = []
    widget.focus()
    widget.unfocus()
    widget.pin()
    widget.unpin()
    assert [message["action"] for message in ui.messages] == ["focus", "unfocus", "pin", "unpin"]
    widget.remove()
    assert ui.messages[-1] == {"version": "v0.9", "deleteSurface": {"surfaceId": "surface_1"}}


def test_update_keeps_the_handle_and_surface_identity():
    ui = UI()
    chart = ui.chart.line(points=[{"label": "Jan", "value": 42}], title="Before", surface_id="sales")
    ui.messages = []
    chart.update(title="After", points=[{"label": "Jan", "value": 50}])
    assert chart.title == "After"
    assert chart.points[0].value == 50
    assert chart.surface_id == "sales"
    assert len(ui.messages) == 1
    assert ui.messages[0]["updateComponents"]["surfaceId"] == "sales"
    assert ui.messages[0]["updateComponents"]["components"][0]["title"] == "After"
    chart.pin()
    assert ui.messages[-1]["action"] == "pin"


@pytest.mark.parametrize(
    "attributes", [{"points": [{"label": "A", "value": -1}]}, {"id": "other"}, {"component": "LineChart"}, {"unknown": True}]
)
def test_rejected_update_preserves_attributes_and_published_state(attributes):
    ui = UI()
    chart = ui.chart.pie(points=[{"label": "A", "value": 42}], title="Original")
    ui.messages = []
    with pytest.raises((ValueError, ValidationError)):
        chart.update(**attributes)
    assert chart.title == "Original"
    assert chart.points[0].value == 42
    assert chart.id == "root"
    assert ui.messages == []
    chart.update(title="Valid update")
    assert chart.title == "Valid update"


def test_widget_attributes_cannot_be_assigned_without_publishing():
    ui = UI()
    widget = ui.chart.bar(points=[{"label": "A", "value": 1}], title="Original")
    with pytest.raises(ValidationError, match="frozen"):
        widget.title = "Unpublished"
    assert widget.title == "Original"


@pytest.mark.parametrize("invalidate", ["remove", "replace", "reset"])
def test_removed_replaced_and_reset_handles_cannot_mutate_new_widgets(invalidate):
    ui = UI()
    original = ui.chart.bar(points=[{"label": "A", "value": 1}], surface_id="notice")
    if invalidate == "remove":
        original.remove()
    elif invalidate == "replace":
        ui.chart.line(points=[{"label": "A", "value": 2}], surface_id="notice")
    else:
        ui.close()
    previous = list(ui.messages)
    for operation in (original.pin, original.focus, original.remove, lambda: original.update(title="Stale")):
        with pytest.raises(RuntimeError, match="removed, replaced, or reset"):
            operation()
    assert ui.messages == previous


def test_invalid_replacement_does_not_destroy_the_existing_surface():
    ui = UI()
    original = ui.chart.bar(points=[{"label": "A", "value": 1}], title="Original", surface_id="notice")
    ui.messages = []
    with pytest.raises(ValidationError):
        ui.show([{"id": "root", "component": "PieChart", "points": [{"label": "A", "value": -1}]}], surface_id="notice")
    assert ui.messages == []
    original.update(title="Still usable")
    assert original.title == "Still usable"


def test_failed_publication_does_not_change_widget_attributes():
    ui = UI()
    widget = ui.chart.bar(points=[{"label": "A", "value": 1}], title="Original")
    for _ in range(98):
        widget.pin()
    with pytest.raises(ValueError, match="100 messages"):
        widget.update(title="Not published")
    assert widget.title == "Original"
    ui.messages = []
    widget.update(title="Now published")
    assert widget.title == "Now published"


@pytest.mark.parametrize("name", ["text", "metric", "table", "column", "row", "card"])
def test_non_chart_widget_constructors_are_not_exposed(name):
    ui = UI()
    assert not hasattr(ui, name)
