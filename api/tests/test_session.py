import pytest


def execute(client, code):
    return client.post("/", json={"code": code}).json()


def test_widgets_publish_a2ui_messages_and_preserve_python_state(api_client):
    result = execute(
        api_client, "amount = 42\nui.plot([{'type': 'bar', 'x': ['Revenue'], 'y': [amount]}], surface_id='revenue')\nprint(amount)\ndone()"
    )
    assert result["exception"] is None
    assert result["done"] is True
    assert result["stdout"] == "42\n"
    assert result["ui"] == [
        {"version": "v0.9", "createSurface": {"surfaceId": "revenue", "catalogId": "urn:pyshell:catalog:v3"}},
        {
            "version": "v0.9",
            "updateComponents": {
                "surfaceId": "revenue",
                "components": [
                    {
                        "id": "root",
                        "component": "Figure",
                        "data": [{"type": "bar", "name": "", "x": ["Revenue"], "y": [42]}],
                        "layout": {"title": {"text": ""}, "xaxis": {"title": {"text": ""}}, "showlegend": True},
                    }
                ],
            },
        },
    ]
    assert execute(api_client, "print(amount)")["ui"] == []

    # Replacing a named surface deletes its old protocol state first.
    replacement = execute(api_client, "ui.plot([{'type': 'scatter', 'x': ['Revenue'], 'y': [50]}], surface_id='revenue')")["ui"]
    assert replacement[0] == {"version": "v0.9", "deleteSurface": {"surfaceId": "revenue"}}
    assert replacement[1]["createSurface"]["surfaceId"] == "revenue"
    assert execute(api_client, "ui.remove('revenue')")["ui"] == [replacement[0]]


def test_a_typed_figure_can_be_published(api_client):
    result = execute(
        api_client, "from src.ui.plot.figure import Figure\nui.show(Figure(data=[{'type': 'scatter', 'x': ['Jan'], 'y': [42]}]))"
    )
    assert result["exception"] is None
    assert result["ui"][1]["updateComponents"]["components"][0]["component"] == "Figure"


# Unsupported components and malformed figures must fail before publication.
@pytest.mark.parametrize(
    "attributes",
    [
        {"component": "Image", "url": "https://example.com"},
        {"component": "Text", "text": "Hello"},
        {"component": "Metric", "label": "Revenue", "value": "$42"},
        {"component": "Column", "children": ["root"]},
        {"component": "Row", "children": ["root"]},
        {"component": "Card", "children": ["root"]},
        {"component": "Table", "columns": ["Name"], "rows": [["One"]]},
        {"component": "BarChart", "points": [{"label": "A", "value": 1}], "html": "<script>"},
        {"data": [{"type": "bar", "x": ["A"], "y": [1]}], "html": "<script>"},
        {"id": "other", "data": [{"type": "bar", "x": ["A"], "y": [1]}]},
    ],
)
def test_invalid_ui_does_not_publish_partial_surface(api_client, attributes):
    result = execute(api_client, f"ui.plot(**{attributes!r})")
    assert result["exception"]
    assert result["ui"] == []
    assert execute(api_client, "print(ui.surfaces)")["stdout"] == "set()\n"


def test_python_errors_preserve_widgets_but_reset_clears_display_state(api_client):
    result = execute(api_client, "ui.plot([{'type': 'bar', 'x': ['A'], 'y': [1]}], surface_id='notice')\n1 / 0")
    assert result["exception"]
    assert result["ui"][0]["createSurface"]["surfaceId"] == "notice"
    assert execute(api_client, "pass")["ui"] == []
    api_client.post("/reset")
    fresh = execute(api_client, "ui.plot([{'type': 'bar', 'x': ['A'], 'y': [2]}], surface_id='notice')")["ui"]
    assert "createSurface" in fresh[0]


def test_display_limits_reject_excess_surfaces_and_nonfinite_numbers(api_client):
    result = execute(api_client, "for number in range(21):\n    ui.plot([{'type': 'bar', 'x': ['A'], 'y': [number]}])")
    assert "20 surfaces" in result["exception"]
    assert sum("createSurface" in message for message in result["ui"]) == 20
    api_client.post("/reset")
    invalid = execute(api_client, "ui.plot([{'type': 'bar', 'x': ['Missing'], 'y': [float('nan')]}])")
    assert invalid["exception"]
    assert invalid["ui"] == []


def test_pending_message_limit_keeps_last_successful_surface(api_client):
    result = execute(
        api_client, "for number in range(40):\n    ui.plot([{'type': 'bar', 'x': ['A'], 'y': [number]}], surface_id='counter')"
    )
    assert "100 messages" in result["exception"]
    assert len(result["ui"]) <= 100
    assert execute(api_client, "ui.remove('counter')")["exception"] is None


def test_automatic_surface_ids_do_not_replace_named_results(api_client):
    result = execute(
        api_client,
        "ui.plot([{'type': 'bar', 'x': ['A'], 'y': [1]}], surface_id='surface_1')\nui.plot([{'type': 'bar', 'x': ['B'], 'y': [2]}])",
    )
    assert result["exception"] is None
    created = [message["createSurface"]["surfaceId"] for message in result["ui"] if "createSurface" in message]
    assert created == ["surface_1", "surface_2"]
    assert not any("deleteSurface" in message for message in result["ui"])


@pytest.mark.parametrize("kind", ["bar", "scatter", "pie"])
def test_trace_types_preserve_missing_values(api_client, kind):
    trace = {
        "type": kind,
        **({"labels": ["Sales", "Missing"], "values": [42, None]} if kind == "pie" else {"x": ["Sales", "Missing"], "y": [42, None]}),
    }
    result = execute(api_client, f"ui.plot([{trace!r}])")

    # Missing values survive validation and serialization for every trace type.
    assert result["exception"] is None
    chart = result["ui"][1]["updateComponents"]["components"][0]
    assert chart["component"] == "Figure"
    assert chart["data"][0]["type"] == kind
    assert chart["data"][0]["values" if kind == "pie" else "y"] == [42, None]


def test_chart_type_can_change_without_duplicating_surface(api_client):
    execute(api_client, "ui.plot([{'type': 'bar', 'x': ['Jan'], 'y': [10]}], surface_id='trend')")
    result = execute(api_client, "ui.plot([{'type': 'scatter', 'x': ['Jan'], 'y': [10]}], surface_id='trend')")
    assert result["exception"] is None
    assert result["ui"][0] == {"version": "v0.9", "deleteSurface": {"surfaceId": "trend"}}
    assert result["ui"][2]["updateComponents"]["components"][0]["data"][0]["type"] == "scatter"


# Chart validation failures must not leave a partially created surface.
@pytest.mark.parametrize(
    "code",
    [
        "ui.plot([{'type': 'unknown', 'x': ['Sales'], 'y': [42]}])",
        "ui.plot([{'type': 'pie', 'labels': ['Sales'], 'values': [-1]}])",
        "ui.plot([{'type': 'scatter', 'x': ['Sales'], 'y': [float('inf')]}])",
        "ui.plot([{'type': 'pie', 'labels': ['Sales'], 'values': [float('inf')]}])",
    ],
)
def test_invalid_charts_do_not_publish_partial_results(api_client, code):
    result = execute(api_client, code)
    assert result["exception"]
    assert result["ui"] == []


@pytest.mark.parametrize("kind", ["bar", "scatter"])
def test_cartesian_charts_accept_negative_values(api_client, kind):
    result = execute(api_client, f"ui.plot([{{'type': '{kind}', 'x': ['Loss'], 'y': [-42]}}])")
    assert result["exception"] is None
    assert result["ui"][1]["updateComponents"]["components"][0]["data"][0]["y"] == [-42]
