import pytest
from api.main import create_app
from fastapi.testclient import TestClient
from src.api.tests.helpers import execute


def test_widgets_publish_a2ui_messages_and_preserve_python_state():
    api_client = TestClient(create_app())
    result = execute(
        api_client, "amount = 42\nui.plot([{'type': 'bar', 'x': ['Revenue'], 'y': [amount]}], surface_id='revenue')\nprint(amount)\ndone()"
    )
    assert result["exception"] is None
    assert result["done"] is True
    assert result["stdout"] == "42\n"
    created, updated = result["ui"]
    assert created["createSurface"]["surfaceId"] == "revenue"
    assert updated["updateComponents"]["surfaceId"] == "revenue"
    (component,) = updated["updateComponents"]["components"]
    assert component["id"] == "root"
    assert component["component"] == "Figure"
    (trace,) = component["data"]
    assert (trace["type"], trace["x"], trace["y"]) == ("bar", ["Revenue"], [42])
    assert execute(api_client, "print(amount)")["ui"] == []

    # Replacing a named surface deletes its old protocol state first.
    replacement = execute(api_client, "ui.plot([{'type': 'scatter', 'x': ['Revenue'], 'y': [50]}], surface_id='revenue')")["ui"]
    assert replacement[0] == {"version": "v0.9", "deleteSurface": {"surfaceId": "revenue"}}
    assert replacement[1]["createSurface"]["surfaceId"] == "revenue"
    assert replacement[2]["updateComponents"]["components"][0]["data"][0]["mode"] == "lines"
    assert execute(api_client, "ui.remove('revenue')")["ui"] == [replacement[0]]


def test_invalid_ui_does_not_publish_partial_surface():
    api_client = TestClient(create_app())
    attributes = {"data": [{"type": "bar", "x": ["A"], "y": [1]}], "html": "<script>"}
    result = execute(api_client, f"ui.plot(**{attributes!r})")
    assert result["exception"]
    assert result["ui"] == []
    assert execute(api_client, "print(ui.surfaces)")["stdout"] == "set()\n"


def test_display_limits_reject_excess_surfaces():
    api_client = TestClient(create_app())
    result = execute(api_client, "for number in range(21):\n    ui.plot([{'type': 'bar', 'x': ['A'], 'y': [number]}])")
    assert "20 surfaces" in result["exception"]
    assert sum("createSurface" in message for message in result["ui"]) == 20


def test_pending_message_limit_keeps_last_successful_surface():
    api_client = TestClient(create_app())
    result = execute(
        api_client, "for number in range(40):\n    ui.plot([{'type': 'bar', 'x': ['A'], 'y': [number]}], surface_id='counter')"
    )
    assert "100 messages" in result["exception"]
    assert len(result["ui"]) <= 100
    assert execute(api_client, "ui.remove('counter')")["exception"] is None


def test_automatic_surface_ids_do_not_replace_named_results():
    api_client = TestClient(create_app())
    result = execute(
        api_client,
        "ui.plot([{'type': 'bar', 'x': ['A'], 'y': [1]}], surface_id='surface_1')\nui.plot([{'type': 'bar', 'x': ['B'], 'y': [2]}])",
    )
    assert result["exception"] is None
    created = [message["createSurface"]["surfaceId"] for message in result["ui"] if "createSurface" in message]
    assert created == ["surface_1", "surface_2"]
    assert not any("deleteSurface" in message for message in result["ui"])


MISSING_VALUE_TRACES = [
    {"type": "bar", "name": "Revenue", "x": ["Sales", "Missing"], "y": [-42, None]},
    {"type": "pie", "name": "Revenue", "labels": ["Sales", "Zero", "Missing"], "values": [42, 0, None]},
]


@pytest.mark.parametrize("trace", MISSING_VALUE_TRACES, ids=["bar", "pie"])
def test_trace_types_preserve_missing_values(trace):
    # Arrange
    api_client = TestClient(create_app())

    # Act
    result = execute(api_client, f"ui.plot([{trace!r}])")

    # Assert
    assert result["exception"] is None
    chart = result["ui"][1]["updateComponents"]["components"][0]
    assert chart["component"] == "Figure"
    assert chart["data"] == [trace]
