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
    pytest.param({"locations": ["CH-ZH"], "values": []}, id="mismatched-lengths"),
    pytest.param({"locations": ["CH-ZH", "CH-ZH"], "values": [1, 2]}, id="duplicate-locations"),
    pytest.param({"level": "country", "locations": ["CH-ZH"], "values": [1]}, id="incompatible-level"),
]


def test_location_membership_refreshes_when_cached_file_changes(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path))
    path = tmp_path / "world-v2.json"
    data = {"name": "World", "level": "country", "features": [{"id": "CHE", "properties": {"name": "Switzerland"}}]}
    path.write_text(json.dumps(data))
    revision = path.stat().st_mtime_ns
    assert Choropleth(locations=["CHE"], values=[1]).locations == ["CHE"]
    with pytest.raises(ValidationError, match="Unknown location for this geography"):
        Choropleth(locations=["FRA"], values=[1])

    data["features"] = [{"id": "FRA", "properties": {"name": "France"}}]
    path.write_text(json.dumps(data))
    os.utime(path, ns=(revision + 1_000_000_000, revision + 1_000_000_000))

    assert Choropleth(locations=["FRA"], values=[1]).locations == ["FRA"]
    assert geography.geography_info()["locations"] == {"FRA": "France"}
    with pytest.raises(ValidationError, match="Unknown location for this geography"):
        Choropleth(locations=["CHE"], values=[1])


@pytest.mark.parametrize("trace", INVALID_TRACES)
def test_validation_checks_input_structure_before_fetching(monkeypatch, trace):
    # Arrange
    def unexpected_download(url):
        pytest.fail("Invalid input must be rejected before fetching geography")

    monkeypatch.setattr(geography, "download", unexpected_download)

    # Act
    # Assert
    with pytest.raises(ValidationError):
        UI().plot([{"type": "choropleth", "geography": "CHE", "level": "region", **trace}])


def test_failed_first_lookup_preserves_an_existing_plot(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path))
    ui = UI()
    chart = ui.plot([{"type": "bar", "x": ["A"], "y": [1]}])
    messages = list(ui.messages)

    def unavailable(url):
        raise URLError("offline")

    monkeypatch.setattr(geography, "download", unavailable)
    with pytest.raises(ValidationError, match="first use requires internet"):
        chart.update(data=[{"type": "choropleth", "geography": "CHE", "level": "region", "locations": ["CH-ZH"], "values": [1]}])
    assert chart.data[0].type == "bar"
    assert ui.messages == messages
