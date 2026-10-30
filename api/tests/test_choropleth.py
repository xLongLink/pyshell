import pytest
from pydantic import ValidationError
from src.ui.plot import geography
from urllib.error import URLError
from src.ui.session import UI
from src.ui.plot.geography import geography_info


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_map_values_are_rejected(value):
    with pytest.raises(ValidationError):
        UI().plot([{"type": "choropleth", "locations": ["FRA"], "values": [value]}])


def test_map_publication_and_updates_preserve_validated_state(api_client):
    code = "chart = ui.plot([{'type': 'choropleth', 'locations': ['FRA', 'DEU'], 'values': [0, None]}])"
    result = api_client.post("/", json={"code": code}).json()
    assert result["exception"] is None
    trace = result["ui"][1]["updateComponents"]["components"][0]["data"][0]
    assert trace == {
        "type": "choropleth",
        "name": "",
        "geography": "world",
        "level": "country",
        "locations": ["FRA", "DEU"],
        "values": [0, None],
    }
    invalid = api_client.post(
        "/", json={"code": "chart.update(data=[{'type': 'choropleth', 'locations': ['INVALID'], 'values': [1]}])"}
    ).json()
    assert invalid["exception"]
    assert invalid["ui"] == []
    unchanged = api_client.post("/", json={"code": "print(chart.data[0].locations)"}).json()
    assert unchanged["stdout"] == "['FRA', 'DEU']\n"
    updated = api_client.post(
        "/",
        json={
            "code": "chart.update(data=[{'type': 'choropleth', 'geography': 'FRA', 'level': 'region', 'locations': ['FR-IDF'], 'values': [75]}])"
        },
    ).json()
    assert updated["exception"] is None
    assert updated["ui"][0]["updateComponents"]["components"][0]["data"][0]["geography"] == "FRA"


@pytest.mark.parametrize("geography", ["world", "CHE", "FRA", "USA", "DEU", "GBR"])
def test_provider_locations_can_be_inspected_and_published(geography_provider, geography):
    dataset = geography_info(geography)
    locations = list(dataset["locations"])
    figure = UI().plot(
        [{"type": "choropleth", "geography": geography, "level": dataset["level"], "locations": locations, "values": [1] * len(locations)}]
    )
    assert figure.data[0].locations == locations


@pytest.mark.parametrize("geography", ["ZZZ", "sources", "geographies", "__proto__", "USA.json"])
def test_geography_endpoint_rejects_unknown_assets(api_client, geography):
    response = api_client.get(f"/geographies/{geography}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Unsupported geography"}


def test_validation_checks_input_structure_before_fetching(geography_provider):
    for trace in [
        {"locations": ["CH-ZH"], "values": []},
        {"locations": ["CH-ZH", "CH-ZH"], "values": [1, 2]},
        {"level": "country", "locations": ["CH-ZH"], "values": [1]},
    ]:
        with pytest.raises(ValidationError):
            UI().plot([{"type": "choropleth", "geography": "CHE", "level": "region", **trace}])
    assert geography_provider["calls"] == []


def test_failed_first_lookup_preserves_an_existing_plot(geography_provider, monkeypatch):
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
