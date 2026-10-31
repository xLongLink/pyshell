import os
import json
import pytest
from pydantic import ValidationError
from api.ui.plot import geography
from urllib.error import URLError
from api.ui.session import UI
from api.ui.plot.choropleth import Choropleth

INVALID_TRACES = [
    pytest.param({"locations": ["CH-ZH"], "values": [float("inf")]}, id="nonfinite-value"),
    pytest.param({"locations": ["CH-ZH", "CH-BE"], "values": [1]}, id="mismatched-lengths"),
    pytest.param({"locations": ["CH-ZH", "CH-ZH"], "values": [1, 2]}, id="duplicate-locations"),
    pytest.param({"level": "region", "locations": ["CH-ZH"], "values": [1]}, id="legacy-level-field"),
]


@pytest.mark.parametrize(("name", "level", "location"), [("world", "country", "CHE"), ("CHE", "region", "CH-ZH")])
def test_geography_determines_level_without_a_trace_field(monkeypatch, name, level, location):
    monkeypatch.setattr("api.ui.plot.choropleth._geography_info", lambda geography: {"level": level, "locations": {location: "Location"}})
    ui = UI()
    chart = ui.plot([{"type": "choropleth", "geography": name, "locations": [location], "values": [1]}])
    assert chart.data[0].geography == name
    assert "level" not in chart.data[0].model_dump()
    messages = ui.state.snapshot()["events"][0]["messages"]
    assert messages[0]["createSurface"] == {"surfaceId": chart.surface_id}
    assert "level" not in messages[1]["updateComponents"]["components"][0]["data"][0]


@pytest.mark.parametrize(("name", "level"), [("world", "region"), ("CHE", "country")])
@pytest.mark.parametrize("operation", ["update", "replace"])
def test_wrong_boundary_level_preserves_an_existing_plot(monkeypatch, name, level, operation):
    monkeypatch.setattr("api.ui.plot.choropleth._geography_info", lambda geography: {"level": level, "locations": {"CHE": "Switzerland"}})
    ui = UI()
    chart = ui.plot([{"type": "bar", "x": ["A"], "y": [1]}], surface_id="map")
    snapshot = ui.state.snapshot()
    trace = {"type": "choropleth", "geography": name, "locations": ["CHE"], "values": [1]}
    publish = chart.update if operation == "update" else ui.plot
    attributes = {} if operation == "update" else {"surface_id": "map"}
    with pytest.raises(ValidationError, match="Unsupported level for this geography"):
        publish(data=[trace], **attributes)
    assert chart.data[0].type == "bar"
    assert ui.state.snapshot() == snapshot
    chart.update(layout={"title": {"text": "Still valid"}})


def test_location_membership_refreshes_when_cached_file_changes(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path))
    path = tmp_path / "world-v2.json"
    data = {"name": "World", "level": "country", "features": [{"id": "CHE", "properties": {"name": "Switzerland"}}]}
    path.write_text(json.dumps(data))
    revision = path.stat().st_mtime_ns
    assert Choropleth(locations=["CHE"], values=[1]).locations == ("CHE",)
    with pytest.raises(ValidationError, match="Unknown location for this geography"):
        Choropleth(locations=["FRA"], values=[1])

    data["features"] = [{"id": "FRA", "properties": {"name": "France"}}]
    path.write_text(json.dumps(data))
    os.utime(path, ns=(revision + 1_000_000_000, revision + 1_000_000_000))

    assert Choropleth(locations=["FRA"], values=[1]).locations == ("FRA",)
    assert geography.geography_info()["locations"] == {"FRA": "France"}
    with pytest.raises(ValidationError, match="Unknown location for this geography"):
        Choropleth(locations=["CHE"], values=[1])


@pytest.mark.parametrize("trace", INVALID_TRACES)
def test_validation_checks_input_structure_before_geography_lookup(monkeypatch, trace):
    # Arrange
    def unexpected_lookup(name):
        pytest.fail("Invalid input must be rejected before geography lookup")

    monkeypatch.setattr("api.ui.plot.choropleth._geography_info", unexpected_lookup)

    # Act
    # Assert
    with pytest.raises(ValidationError) as error:
        UI().plot([{"type": "choropleth", "geography": "CHE", **trace}])
    if "level" in trace:
        assert [(item["loc"], item["type"]) for item in error.value.errors()] == [(("data", 0, "choropleth", "level"), "extra_forbidden")]


def test_failed_first_lookup_preserves_an_existing_plot(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path))
    ui = UI()
    chart = ui.plot([{"type": "bar", "x": ["A"], "y": [1]}])
    snapshot = ui.state.snapshot()

    def unavailable(url):
        raise URLError("offline")

    monkeypatch.setattr(geography, "download", unavailable)
    with pytest.raises(ValidationError, match="first use requires internet"):
        chart.update(data=[{"type": "choropleth", "geography": "CHE", "locations": ["CH-ZH"], "values": [1]}])
    assert chart.data[0].type == "bar"
    assert ui.state.snapshot() == snapshot
