import pytest
from pydantic import ValidationError
from src.ui.widget import Widget
from src.ui.session import UI
from a2ui.core.exceptions import A2uiError


@pytest.mark.parametrize(
    ("path", "attributes", "component"),
    [
        ("text", {"text": "Hello"}, "Text"),
        ("metric", {"label": "Revenue", "value": "$42"}, "Metric"),
        ("table", {"columns": ["Name"], "rows": [["Apples"]]}, "Table"),
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
    widget = ui.text("Original")
    with pytest.raises(ValidationError, match="frozen"):
        widget.text = "Unpublished"
    assert widget.text == "Original"


@pytest.mark.parametrize("invalidate", ["remove", "replace", "reset"])
def test_removed_replaced_and_reset_handles_cannot_mutate_new_widgets(invalidate):
    ui = UI()
    original = ui.text("Original", surface_id="notice")
    if invalidate == "remove":
        original.remove()
    elif invalidate == "replace":
        ui.text("Replacement", surface_id="notice")
    else:
        ui.close()
    previous = list(ui.messages)
    for operation in (original.pin, original.focus, original.remove, lambda: original.update(text="Stale")):
        with pytest.raises(RuntimeError, match="removed, replaced, or reset"):
            operation()
    assert ui.messages == previous


def test_invalid_replacement_does_not_destroy_the_existing_surface():
    ui = UI()
    original = ui.text("Original", surface_id="notice")
    ui.messages = []
    with pytest.raises(A2uiError):
        ui.show([{"id": "root", "component": "Column", "children": ["missing"]}], surface_id="notice")
    assert ui.messages == []
    original.update(text="Still usable")
    assert original.text == "Still usable"


def test_failed_publication_does_not_change_widget_attributes():
    ui = UI()
    widget = ui.text("Original")
    for _ in range(98):
        widget.pin()
    with pytest.raises(ValueError, match="100 messages"):
        widget.update(text="Not published")
    assert widget.text == "Original"
    ui.messages = []
    widget.update(text="Now published")
    assert widget.text == "Now published"


@pytest.mark.parametrize(("depth", "valid"), [(16, True), (17, False)])
def test_sdk_validates_the_component_depth_limit(depth, valid):
    components = [
        {"id": "root" if index == 0 else f"node_{index}", "component": "Column", "children": [f"node_{index + 1}"]}
        for index in range(depth - 1)
    ]
    components.append({"id": f"node_{depth - 1}", "component": "Text", "text": "Leaf"})
    ui = UI()
    if valid:
        ui.show(components)
        assert len(ui.messages) == 2
    else:
        with pytest.raises(A2uiError):
            ui.show(components)
        assert ui.messages == []
