import pytest


def execute(client, code):
    return client.post("/", json={"code": code}).json()


def test_widgets_publish_a2ui_messages_and_preserve_python_state(api_client):
    result = execute(
        api_client, "amount = 42\nui.chart.bar([{'label': 'Revenue', 'value': amount}], surface_id='revenue')\nprint(amount)\ndone()"
    )
    assert result["exception"] is None
    assert result["done"] is True
    assert result["stdout"] == "42\n"
    assert result["ui"] == [
        {"version": "v0.9", "createSurface": {"surfaceId": "revenue", "catalogId": "urn:pyshell:catalog:v2"}},
        {
            "version": "v0.9",
            "updateComponents": {
                "surfaceId": "revenue",
                "components": [{"id": "root", "component": "BarChart", "points": [{"label": "Revenue", "value": 42}], "title": ""}],
            },
        },
    ]
    assert execute(api_client, "print(amount)")["ui"] == []
    replacement = execute(api_client, "ui.chart.line([{'label': 'Revenue', 'value': 50}], surface_id='revenue')")["ui"]
    assert replacement[0] == {"version": "v0.9", "deleteSurface": {"surfaceId": "revenue"}}
    assert replacement[1]["createSurface"]["surfaceId"] == "revenue"
    assert execute(api_client, "ui.remove('revenue')")["ui"] == [replacement[0]]


def test_a_chart_can_be_published_from_its_wire_model(api_client):
    result = execute(api_client, "ui.show([{'id': 'root', 'component': 'LineChart', 'points': [{'label': 'Jan', 'value': 42}]}])")
    assert result["exception"] is None
    assert result["ui"][1]["updateComponents"]["components"][0]["component"] == "LineChart"


@pytest.mark.parametrize(
    "components",
    [
        [{"id": "root", "component": "Image", "url": "https://example.com"}],
        [{"id": "root", "component": "Text", "text": "Hello"}],
        [{"id": "root", "component": "Metric", "label": "Revenue", "value": "$42"}],
        [{"id": "root", "component": "Column", "children": ["root"]}],
        [{"id": "root", "component": "Row", "children": ["root"]}],
        [{"id": "root", "component": "Card", "children": ["root"]}],
        [{"id": "root", "component": "Table", "columns": ["Name"], "rows": [["One"]]}],
        [{"id": "root", "component": "BarChart", "points": [{"label": "A", "value": 1}], "html": "<script>"}],
        [{"id": "other", "component": "BarChart", "points": [{"label": "A", "value": 1}]}],
        [{"id": "root", "component": "BarChart", "points": [{"label": "A", "value": 1}]}] * 2,
    ],
)
def test_invalid_ui_does_not_publish_partial_surface(api_client, components):
    result = execute(api_client, f"ui.show({components!r})")
    assert result["exception"]
    assert result["ui"] == []
    assert execute(api_client, "print(ui.surfaces)")["stdout"] == "set()\n"


def test_python_errors_preserve_widgets_but_reset_clears_display_state(api_client):
    result = execute(api_client, "ui.chart.bar([{'label': 'A', 'value': 1}], surface_id='notice')\n1 / 0")
    assert result["exception"]
    assert result["ui"][0]["createSurface"]["surfaceId"] == "notice"
    assert execute(api_client, "pass")["ui"] == []
    api_client.post("/reset")
    fresh = execute(api_client, "ui.chart.bar([{'label': 'A', 'value': 2}], surface_id='notice')")["ui"]
    assert "createSurface" in fresh[0]


def test_display_limits_reject_excess_surfaces_and_nonfinite_numbers(api_client):
    result = execute(api_client, "for number in range(21):\n    ui.chart.bar([{'label': 'A', 'value': number}])")
    assert "20 surfaces" in result["exception"]
    assert sum("createSurface" in message for message in result["ui"]) == 20
    api_client.post("/reset")
    invalid = execute(api_client, "ui.chart.bar([{'label': 'Missing', 'value': float('nan')}])")
    assert invalid["exception"]
    assert invalid["ui"] == []


def test_pending_message_limit_keeps_last_successful_surface(api_client):
    result = execute(api_client, "for number in range(40):\n    ui.chart.bar([{'label': 'A', 'value': number}], surface_id='counter')")
    assert "100 messages" in result["exception"]
    assert len(result["ui"]) <= 100
    assert execute(api_client, "ui.remove('counter')")["exception"] is None


def test_automatic_surface_ids_do_not_replace_named_results(api_client):
    result = execute(
        api_client, "ui.chart.bar([{'label': 'A', 'value': 1}], surface_id='surface_1')\nui.chart.bar([{'label': 'B', 'value': 2}])"
    )
    assert result["exception"] is None
    created = [message["createSurface"]["surfaceId"] for message in result["ui"] if "createSurface" in message]
    assert created == ["surface_1", "surface_2"]
    assert not any("deleteSurface" in message for message in result["ui"])


@pytest.mark.parametrize(("kind", "component"), [("bar", "BarChart"), ("line", "LineChart"), ("pie", "PieChart")])
def test_chart_types_publish_the_matching_component(api_client, kind, component):
    result = execute(api_client, f"ui.chart.{kind}([{{'label': 'Sales', 'value': 42}}, {{'label': 'Missing', 'value': None}}])")
    assert result["exception"] is None
    chart = result["ui"][1]["updateComponents"]["components"][0]
    assert chart["component"] == component
    assert chart["points"] == [{"label": "Sales", "value": 42}, {"label": "Missing", "value": None}]


def test_chart_type_can_change_without_duplicating_surface(api_client):
    execute(api_client, "ui.chart.bar([{'label': 'Jan', 'value': 10}], surface_id='trend')")
    result = execute(api_client, "ui.chart.line([{'label': 'Jan', 'value': 10}], surface_id='trend')")
    assert result["exception"] is None
    assert result["ui"][0] == {"version": "v0.9", "deleteSurface": {"surfaceId": "trend"}}
    assert result["ui"][2]["updateComponents"]["components"][0]["component"] == "LineChart"


@pytest.mark.parametrize(
    "code",
    [
        "ui.chart.unknown([{'label': 'Sales', 'value': 42}])",
        "ui.chart.pie([{'label': 'Sales', 'value': -1}])",
        "ui.chart.line([{'label': 'Sales', 'value': float('inf')}])",
        "ui.chart.pie([{'label': 'Sales', 'value': float('inf')}])",
    ],
)
def test_invalid_charts_do_not_publish_partial_results(api_client, code):
    result = execute(api_client, code)
    assert result["exception"]
    assert result["ui"] == []


@pytest.mark.parametrize("kind", ["bar", "line"])
def test_cartesian_charts_accept_negative_values(api_client, kind):
    result = execute(api_client, f"ui.chart.{kind}([{{'label': 'Loss', 'value': -42}}])")
    assert result["exception"] is None
    assert result["ui"][1]["updateComponents"]["components"][0]["points"][0]["value"] == -42
