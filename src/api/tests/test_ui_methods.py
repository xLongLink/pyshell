import pydoc
import pytest
import inspect
from pydantic import ValidationError
from api.ui.session import UI


@pytest.mark.parametrize(
    ("method", "arguments", "options", "trace"),
    [
        (
            "scatter",
            [[2024, 2025], [9, None]],
            {"mode": "lines+markers"},
            {"type": "scatter", "x": [2024, 2025], "y": [9, None], "mode": "lines+markers"},
        ),
        (
            "area",
            [[2024, 2025], [9, None]],
            {"curve": "monotone"},
            {"type": "area", "x": [2024, 2025], "y": [9, None], "curve": "monotone"},
        ),
        ("bar", [["A", "B"], [9, None]], {}, {"type": "bar", "x": ["A", "B"], "y": [9, None]}),
        ("pie", [["A", "B"], [9, None]], {}, {"type": "pie", "labels": ["A", "B"], "values": [9, None]}),
        ("choropleth", [["CHE", "FRA"], [9, None]], {}, {"type": "choropleth", "locations": ["CHE", "FRA"], "values": [9, None]}),
    ],
)
def test_chart_methods_publish_the_same_models_as_plot(country_geography, method, arguments, options, trace):
    ui = UI()
    chart = getattr(ui, method)(*arguments, **options, name="Series", title="Counts", surface_id="counts")
    expected = UI().plot([{**trace, "name": "Series"}], layout={"title": {"text": "Counts"}}, surface_id="counts")
    assert chart.model_dump() == expected.model_dump()
    assert chart.surface_id == "counts"
    chart.update(layout={"title": {"text": "Updated"}})
    assert chart.layout.title.text == "Updated"
    chart.pin()
    assert ui.state.snapshot()["placements"]["counts"]["pinned"] is True
    chart.focus()
    assert ui.state.snapshot()["placements"]["counts"]["pinned"] is False
    chart.remove()
    assert ui.state.snapshot()["events"] == []


@pytest.mark.parametrize("method", ["scatter", "area", "bar", "pie", "choropleth"])
def test_chart_methods_validate_before_replacing_a_surface(country_geography, method):
    ui = UI()
    chart = ui.bar(["A"], [1], surface_id="counts")
    snapshot = ui.state.snapshot()
    with pytest.raises(ValidationError):
        getattr(ui, method)(["CHE"], [1, 2], surface_id="counts")
    assert ui.state.snapshot() == snapshot
    chart.update(layout={"title": {"text": "Still active"}})


def test_chart_methods_preserve_named_surface_replacement():
    ui = UI()
    previous = ui.bar(["A"], [1], surface_id="counts")
    replacement = ui.pie(["B"], [2], surface_id="counts")
    assert replacement.surface_id == previous.surface_id
    with pytest.raises(RuntimeError, match="replaced"):
        previous.remove()
    assert len(ui.state.snapshot()["events"]) == 1


@pytest.mark.parametrize("method", ["scatter", "area", "bar", "pie", "choropleth"])
def test_chart_method_help_is_focused_and_examples_run(country_geography, method):
    ui = UI()
    function = getattr(ui, method)
    documentation = pydoc.render_doc(function, renderer=pydoc.plaintext)
    assert "Args:" in documentation
    assert "Returns:" in documentation
    assert "model_dump" not in documentation
    assert "BaseModel" not in documentation
    example = inspect.getdoc(function).split("Example:\n", 1)[1].split("\n\n", 1)[0]
    namespace = {"ui": ui}
    exec(inspect.cleandoc(example), namespace)
    assert namespace["chart"].surface_id == "surface_1"
