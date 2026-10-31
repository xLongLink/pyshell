import pytest
from api.main import create_app
from fastapi.testclient import TestClient
from src.api.tests.helpers import execute


def test_widgets_publish_a2ui_messages_and_preserve_python_state():
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
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
    assert replacement[2]["updateComponents"]["components"][0]["data"][0]["mode"] == "lines"
    assert execute(api_client, "ui.remove('revenue')")["ui"] == [replacement[0]]


def test_invalid_ui_does_not_publish_partial_surface():
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
    attributes = {"data": [{"type": "bar", "x": ["A"], "y": [1]}], "html": "<script>"}
    result = execute(api_client, f"ui.plot(**{attributes!r})")
    assert result["exception"]
    assert result["ui"] == []
    assert execute(api_client, "print(ui.surfaces)")["stdout"] == "set()\n"


def test_python_errors_preserve_widgets_but_reset_clears_display_state():
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
    result = execute(api_client, "ui.plot([{'type': 'bar', 'x': ['A'], 'y': [1]}], surface_id='notice')\n1 / 0")
    assert result["exception"]
    assert result["ui"][0]["createSurface"]["surfaceId"] == "notice"
    assert execute(api_client, "pass")["ui"] == []
    api_client.post("/reset")
    fresh = execute(api_client, "ui.plot([{'type': 'bar', 'x': ['A'], 'y': [2]}], surface_id='notice')")["ui"]
    assert "createSurface" in fresh[0]


def test_display_limits_reject_excess_surfaces():
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
    result = execute(api_client, "for number in range(21):\n    ui.plot([{'type': 'bar', 'x': ['A'], 'y': [number]}])")
    assert "20 surfaces" in result["exception"]
    assert sum("createSurface" in message for message in result["ui"]) == 20


def test_pending_message_limit_keeps_last_successful_surface():
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
    result = execute(
        api_client, "for number in range(40):\n    ui.plot([{'type': 'bar', 'x': ['A'], 'y': [number]}], surface_id='counter')"
    )
    assert "100 messages" in result["exception"]
    assert len(result["ui"]) <= 100
    assert execute(api_client, "ui.remove('counter')")["exception"] is None


def test_automatic_surface_ids_do_not_replace_named_results():
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
    result = execute(
        api_client,
        "ui.plot([{'type': 'bar', 'x': ['A'], 'y': [1]}], surface_id='surface_1')\nui.plot([{'type': 'bar', 'x': ['B'], 'y': [2]}])",
    )
    assert result["exception"] is None
    created = [message["createSurface"]["surfaceId"] for message in result["ui"] if "createSurface" in message]
    assert created == ["surface_1", "surface_2"]
    assert not any("deleteSurface" in message for message in result["ui"])


@pytest.mark.parametrize("kind", ["bar", "pie"])
def test_trace_types_preserve_missing_values(kind):
    api_client = TestClient(create_app(), headers={"authorization": "Bearer test-token"})
    # Arrange
    trace = {
        "type": kind,
        "name": "Revenue",
        **(
            {"labels": ["Sales", "Zero", "Missing"], "values": [42, 0, None]}
            if kind == "pie"
            else {"x": ["Sales", "Missing"], "y": [-42, None]}
        ),
    }

    # Act
    result = execute(api_client, f"ui.plot([{trace!r}])")

    # Assert
    assert result["exception"] is None
    chart = result["ui"][1]["updateComponents"]["components"][0]
    assert chart["component"] == "Figure"
    assert chart["data"][0]["type"] == kind
    assert chart["data"][0]["name"] == "Revenue"
    assert chart["data"][0]["labels" if kind == "pie" else "x"] == (["Sales", "Zero", "Missing"] if kind == "pie" else ["Sales", "Missing"])
    assert chart["data"][0]["values" if kind == "pie" else "y"] == ([42, 0, None] if kind == "pie" else [-42, None])
