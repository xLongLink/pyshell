import pytest
from copy import deepcopy
from pydantic import ValidationError
from api.ui.session import UI
from api.ui.plot.choropleth import Choropleth


@pytest.fixture
def boundaries():
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "CHE",
                "properties": {"name": "Switzerland", "source_code": "CH"},
                "geometry": {"type": "Polygon", "coordinates": [[[7, 46], [8, 46], [8, 47], [7, 46]]]},
            }
        ],
    }


def test_inline_map_publishes_and_updates_without_fetching_geography(monkeypatch, boundaries):
    def unavailable(geography):
        pytest.fail("Inline maps must not fetch geography")

    monkeypatch.setattr("api.ui.plot.choropleth._geography_info", unavailable)
    ui = UI()
    widget = ui.plot([{"type": "choropleth", "geojson": boundaries, "locations": ["CHE"], "values": [42]}])
    trace = widget.model_dump(mode="json")["data"][0]
    assert trace["geojson"]["features"][0]["id"] == "CHE"
    assert trace["geojson"]["features"][0]["properties"] == {"name": "Switzerland"}
    original = deepcopy(trace["geojson"])
    boundaries["features"][0]["geometry"]["coordinates"][0][0][0] = 0
    assert widget.model_dump(mode="json")["data"][0]["geojson"] == original
    widget.update(data=[{**trace, "values": [84]}])
    assert widget.data[0].values == (84,)
    assert widget.model_dump(mode="json")["data"][0]["geojson"] == original


def test_inline_multipolygon_and_missing_names(boundaries):
    feature = boundaries["features"][0]
    feature.pop("properties")
    feature["geometry"] = {"type": "MultiPolygon", "coordinates": [feature["geometry"]["coordinates"]]}
    trace = Choropleth(geojson=boundaries, locations=["CHE"], values=[None])
    assert trace.geojson.features[0].geometry.type == "MultiPolygon"
    assert trace.geojson.features[0].properties.name == ""


@pytest.mark.parametrize("position", [[181, 46], [7, 91], [float("nan"), 46], [7, float("inf")], [True, 46], ["7", 46], [7, 46, 100]])
def test_inline_map_rejects_invalid_positions(boundaries, position):
    boundaries["features"][0]["geometry"]["coordinates"][0][1] = position
    with pytest.raises(ValidationError):
        Choropleth(geojson=boundaries, locations=["CHE"], values=[1])


@pytest.mark.parametrize("case", ["empty", "duplicate", "open", "short", "point", "missing_id", "unknown_location"])
def test_inline_map_rejects_invalid_boundaries(boundaries, case):
    feature = boundaries["features"][0]
    if case == "empty":
        boundaries["features"] = []
    elif case == "duplicate":
        boundaries["features"].append(deepcopy(feature))
    elif case == "open":
        feature["geometry"]["coordinates"][0][-1] = [7, 47]
    elif case == "short":
        feature["geometry"]["coordinates"][0].pop()
    elif case == "point":
        feature["geometry"] = {"type": "Point", "coordinates": [7, 46]}
    elif case == "missing_id":
        feature.pop("id")
    with pytest.raises(ValidationError):
        Choropleth(geojson=boundaries, locations=["OTHER" if case == "unknown_location" else "CHE"], values=[1])


def test_inline_map_bounds_total_positions(boundaries):
    feature = boundaries["features"][0]
    feature["geometry"]["coordinates"] = [[[7, 46]] * 4000]
    second = deepcopy(feature)
    second["id"] = "FRA"
    second["geometry"]["coordinates"] = [[[7, 46]] * 4]
    boundaries["features"].append(second)
    with pytest.raises(ValueError, match="4,000 positions"):
        Choropleth(geojson=boundaries, locations=["CHE"], values=[1])


def test_inline_map_publication_keeps_surface_size_limit(boundaries):
    feature = boundaries["features"][0]
    boundaries["features"] = [{**feature, "id": str(index), "properties": {"name": "x" * 10000}} for index in range(30)]
    with pytest.raises(ValueError, match="256 KiB"):
        UI().plot([{"type": "choropleth", "geojson": boundaries, "locations": ["0"], "values": [1]}])
