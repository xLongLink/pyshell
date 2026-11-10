import pytest
from pydantic import ValidationError
from api.ui.session import UI
from api.ui.plot.map import Map


def layer(identifier="buildings", count=1):
    return {
        "id": identifier,
        "name": "Renovated before 2025",
        "popup_fields": ["renovation_year", "name"],
        "data": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": str(index),
                    "geometry": {"type": "Point", "coordinates": [8.54, 47.37]},
                    "properties": {"renovation_year": 2024, "name": "Building", "occupied": True, "height": None},
                }
                for index in range(count)
            ],
        },
    }


def test_map_publish_navigation_data_update_and_lifecycle():
    ui = UI()
    chart = ui.map([layer()], title="Buildings", surface_id="buildings")
    assert chart.surface_id == "buildings"
    assert chart.data[0].layers[0].data.features[0].properties["renovation_year"] == 2024
    layers = chart.model_dump(mode="json")["data"][0]["layers"]
    chart.set_view(country="CHE")
    assert chart.data[0].view.country == "CHE"
    chart.set_view(country="CHE")
    assert chart.data[0].view.revision == 2
    chart.set_view(center=[8.54, 47.37], zoom=14)
    assert chart.data[0].view.center == (8.54, 47.37)
    assert chart.data[0].layers[0].id == "buildings"
    assert chart.model_dump(mode="json")["data"][0]["layers"] == layers
    assert ui.state.snapshot()["events"][0]["messages"][1]["updateComponents"]["components"][0]["data"][0]["layers"] == layers
    chart.update(data=[{"type": "map", "layers": [layer(count=0)]}])
    assert chart.data[0].view is None
    assert not chart.data[0].layers[0].data.features
    chart.pin()
    assert ui.state.snapshot()["placements"]["buildings"]["pinned"]
    chart.focus()
    chart.remove()
    with pytest.raises(RuntimeError, match="removed"):
        chart.set_view(country="CHE")


@pytest.mark.parametrize(
    "view",
    [
        {},
        {"zoom": 10},
        {"center": [181, 0]},
        {"center": [0, 0], "zoom": 23},
        {"center": [0, 0], "country": "CHE"},
        {"country": "Switzerland"},
        {"country": "CHE", "zoom": 5},
        {"bounds": [[5, 48], [11, 45]]},
    ],
)
def test_invalid_navigation_is_atomic(view):
    ui = UI()
    chart = ui.map([layer()], view={"country": "CHE"})
    snapshot = ui.state.snapshot()
    with pytest.raises(ValidationError):
        chart.set_view(**view)
    assert chart.data[0].view.country == "CHE"
    assert ui.state.snapshot() == snapshot


def test_map_accepts_polygons_multipolygons_and_empty_results():
    ring = [[8, 47], [8.1, 47], [8.1, 47.1], [8, 47]]
    data = layer(count=2)
    data["data"]["features"][0]["geometry"] = {"type": "Polygon", "coordinates": [ring]}
    data["data"]["features"][1]["geometry"] = {"type": "MultiPolygon", "coordinates": [[ring]]}
    ui = UI()
    chart = ui.map([data], view={"bounds": [[5, 45], [11, 48]]})
    assert chart.data[0].layers[0].data.features[1].geometry.type == "MultiPolygon"
    chart.update(data=[{"type": "map", "layers": []}])
    assert chart.data[0].layers == ()


@pytest.mark.parametrize(
    "case", ["duplicate_layers", "duplicate_features", "features", "coordinates", "properties", "color", "ring", "positions"]
)
def test_map_rejects_invalid_data(case):
    layers = [layer()]
    feature = layers[0]["data"]["features"][0]
    if case == "duplicate_layers":
        layers.append(layer())
    elif case == "duplicate_features":
        layers[0]["data"]["features"].append(feature)
    elif case == "features":
        layers = [layer("one", 101), layer("two", 100)]
    elif case == "coordinates":
        feature["geometry"]["coordinates"] = [8, float("nan")]
    elif case == "properties":
        feature["properties"] = {"nested": {"value": "not scalar"}}
    elif case == "color":
        layers[0]["color"] = "url(https://example.com)"
    elif case == "ring":
        feature["geometry"] = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]}
    else:
        ring = [[0, 0], [1, 0], [1, 1], [0, 0]]
        layers = [layer("one", 101)]
        for item in layers[0]["data"]["features"]:
            item["geometry"] = {"type": "Polygon", "coordinates": [ring] * 10}
    with pytest.raises(ValidationError):
        Map.model_validate({"type": "map", "layers": layers})


def test_maps_cannot_mix_with_other_traces_and_nonmaps_cannot_navigate():
    ui = UI()
    with pytest.raises(ValidationError, match="exactly one"):
        ui.plot([{"type": "map"}, {"type": "bar", "x": [1], "y": [1]}])
    chart = ui.bar([1], [1])
    with pytest.raises(ValueError, match="interactive map"):
        chart.set_view(country="CHE")


def test_oversized_map_cannot_replace_an_existing_surface():
    ui = UI()
    chart = ui.map(surface_id="map")
    oversized = layer(count=30)
    for feature in oversized["data"]["features"]:
        feature["properties"]["name"] = "x" * 10000
    with pytest.raises(ValueError, match="256 KiB"):
        ui.map([oversized], surface_id="map")
    chart.set_view(country="CHE")


def test_map_data_and_published_properties_are_immutable():
    ui = UI()
    source = layer()
    chart = ui.map([source])
    snapshot = ui.state.snapshot()
    source["data"]["features"][0]["properties"]["renovation_year"] = 2026
    feature = chart.data[0].layers[0].data.features[0]
    assert feature.properties["renovation_year"] == 2024
    with pytest.raises(TypeError):
        feature.properties["renovation_year"] = 2026
    with pytest.raises(TypeError):
        feature.geometry.coordinates[0] = 0
    chart.update(layout={"title": {"text": "Updated title"}})
    assert chart.data[0].layers[0].data.features[0].properties["renovation_year"] == 2024
    assert (
        snapshot["events"][0]["messages"][1]["updateComponents"]["components"][0]["data"][0]["layers"][0]["data"]["features"][0][
            "properties"
        ]["renovation_year"]
        == 2024
    )
