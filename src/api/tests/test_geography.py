import json
import pytest
from io import BytesIO
from types import SimpleNamespace
from api.main import create_app
from api.ui.plot import geography
from urllib.error import URLError, HTTPError
from urllib.request import Request
from fastapi.testclient import TestClient
from api.ui.plot.choropleth import Choropleth


@pytest.mark.parametrize(
    ("metadata_list", "boundary_id", "license_metadata"),
    [(False, None, {}), (True, "CHE-current", {"boundaryLicense": "CC BY 4.0", "licenseSource": "https://[invalid"})],
)
def test_geography_omits_boundary_id_and_preserves_source_metadata(monkeypatch, tmp_path, metadata_list, boundary_id, license_metadata):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path))
    metadata_url = geography.METADATA_URL.format(geography="CHE")
    download_url = "https://raw.githubusercontent.com/wmgeolab/geoBoundaries/main/CHE.geojson"
    metadata = {
        "boundaryISO": "CHE",
        "boundaryType": "ADM1",
        "boundaryName": "Switzerland",
        "boundarySource": "Swiss boundaries",
        "boundaryYearRepresented": 2024,
        "admUnitCount": "1",
        "simplifiedGeometryGeoJSON": download_url,
        **license_metadata,
    }
    if boundary_id is not None:
        metadata["boundaryID"] = boundary_id
    geometry = {"type": "Polygon", "coordinates": [[[7, 46], [8, 46], [8, 47], [7, 46]]]}
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"shapeISO": "CH-BE", "shapeName": "Bern", "shapeGroup": "CHE", "shapeType": "ADM1"},
                "geometry": geometry,
            }
        ],
    }
    responses = {metadata_url: [metadata] if metadata_list else metadata, download_url: data}
    monkeypatch.setattr(geography, "download", responses.__getitem__)

    with TestClient(create_app()) as client:
        response = client.get("/geographies/CHE")

    assert response.status_code == 200
    assert response.json() == {
        "type": "FeatureCollection",
        "geography": "CHE",
        "name": "Switzerland",
        "level": "region",
        "source": {"source": "Swiss boundaries / geoBoundaries gbOpen", "url": metadata_url, "year": "2024"},
        "features": [{"type": "Feature", "id": "CH-BE", "properties": {"name": "Bern"}, "geometry": geometry}],
    }
    assert json.loads((tmp_path / "CHE-v2.json").read_text()) == response.json()
    assert geography.geography_info("CHE") == {"name": "Switzerland", "level": "region", "locations": {"CH-BE": "Bern"}}


def test_world_ignores_administrative_identifiers_and_omits_license_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path))

    def unexpected_home():
        pytest.fail("Explicit cache directory should not require home-directory resolution")

    monkeypatch.setattr(geography.Path, "home", unexpected_home)
    geometry = {"type": "Polygon", "coordinates": [[[7, 46], [8, 46], [8, 47], [7, 46]]]}
    data = {
        "type": "FeatureCollection",
        "features": [{"properties": {"ISO_A3_EH": "CHE", "NAME_EN": "Switzerland", "shapeISO": 42}, "geometry": geometry}],
    }
    monkeypatch.setattr(geography, "download", lambda url: data)

    result = json.loads(geography.geography_path("world").read_text())

    assert result["source"] == {"source": "Natural Earth, 1:110m Admin 0 Countries", "url": geography.WORLD_URL}
    assert result["features"] == [{"type": "Feature", "id": "CHE", "properties": {"name": "Switzerland"}, "geometry": geometry}]


def test_administrative_maps_still_reject_invalid_identifiers():
    data = {"type": "FeatureCollection", "features": [{"properties": {"shapeISO": 42}}]}

    with pytest.raises(ValueError, match="Invalid geographic identifier"):
        geography.normalize("CHE", "Switzerland", {"count": 1}, data)


@pytest.mark.parametrize(
    ("metadata", "error", "message"),
    [
        ({"boundaryISO": "FRA", "boundaryType": "ADM1"}, ValueError, "Unexpected geography metadata"),
        ({"boundaryISO": "CHE", "boundaryType": "ADM0"}, ValueError, "Unexpected geography metadata"),
        (None, ValueError, "Unexpected geography metadata"),
        ([], geography.UnsupportedGeographyError, "Unsupported geography"),
        ([{"boundaryISO": "CHE", "boundaryType": "ADM1"}] * 2, ValueError, "Ambiguous geography metadata"),
    ],
)
def test_source_rejects_unexpected_or_ambiguous_metadata(monkeypatch, metadata, error, message):
    monkeypatch.setattr(geography, "download", lambda url: metadata)

    with pytest.raises(error, match=message):
        geography._source("CHE")


@pytest.mark.parametrize(
    ("codes", "expected"),
    [((" ch-be ", "ch-zh"), ["CH-BE", "CH-ZH"]), ((None, ""), ["CHE-0", "CHE-1"]), (("CH-BE", " ch-be "), ["CHE-0", "CHE-1"])],
)
def test_administrative_identifiers_are_normalized_and_duplicates_use_shape_ids(codes, expected):
    geometry = {"type": "Polygon", "coordinates": [[[7, 46], [8, 46], [8, 47], [7, 46]]]}
    data = {
        "type": "FeatureCollection",
        "features": [
            {"properties": {"shapeISO": code, "shapeID": str(index), "shapeGroup": "CHE", "shapeType": "ADM1"}, "geometry": geometry}
            for index, code in enumerate(codes)
        ],
    }

    result = geography.normalize("CHE", "Switzerland", {"count": 2}, data)

    assert [feature["id"] for feature in result["features"]] == expected


@pytest.mark.parametrize(("geometry_type", "label"), [("Polygon", "ZÃ¼rich"), ("MultiPolygon", "Â"), ("MultiPolygon", "Switzerland")])
def test_world_polygon_normalization_preserves_labels(geometry_type, label):
    coordinates = [[[7, 46], [8, 46], [8, 47], [7, 46]]]
    geometry = {"type": geometry_type, "coordinates": coordinates if geometry_type == "Polygon" else [coordinates]}
    data = {"type": "FeatureCollection", "features": [{"properties": {"ISO_A3_EH": "CHE", "NAME_EN": label}, "geometry": geometry}]}

    result = geography.normalize("world", "World", {}, data)

    assert result["features"] == [{"type": "Feature", "id": "CHE", "properties": {"name": label}, "geometry": geometry}]


@pytest.mark.parametrize("failure", [None, "create", "write", "replace"])
def test_cache_write_closes_files_and_cleans_up_on_failure(monkeypatch, tmp_path, failure):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path))
    geometry = {"type": "Polygon", "coordinates": [[[7, 46], [8, 46], [8, 47], [7, 46]]]}
    data = {"type": "FeatureCollection", "features": [{"properties": {"ISO_A3_EH": "CHE", "NAME_EN": "Switzerland"}, "geometry": geometry}]}
    monkeypatch.setattr(geography, "download", lambda url: data)
    create_temporary = geography.NamedTemporaryFile
    replace = geography.Path.replace
    outputs = []

    def temporary_file(**kwargs):
        if failure == "create":
            raise OSError("Cannot create temporary file")
        output = create_temporary(**kwargs)
        outputs.append(output)
        return output

    def failed_write(value, output, **kwargs):
        output.write('{"partial":')
        raise OSError("Cannot write temporary file")

    def replace_file(temporary, destination):
        assert outputs[0].closed
        assert not destination.exists()
        if failure == "replace":
            raise OSError("Cannot replace cache file")
        return replace(temporary, destination)

    monkeypatch.setattr(geography, "NamedTemporaryFile", temporary_file)
    monkeypatch.setattr(geography.Path, "replace", replace_file)
    if failure == "write":
        monkeypatch.setattr(geography.json, "dump", failed_write)

    if failure is None:
        path = geography.geography_path("world")
        assert json.loads(path.read_text())["features"][0]["id"] == "CHE"
        assert list(tmp_path.iterdir()) == [path]
    else:
        with pytest.raises(OSError, match="Cannot"):
            geography.geography_path("world")
        assert not list(tmp_path.iterdir())
    assert all(output.closed for output in outputs)


def test_cached_geography_with_boundary_id_remains_readable(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path))
    cached = {
        "type": "FeatureCollection",
        "geography": "CHE",
        "name": "Switzerland",
        "level": "region",
        "source": {"source": "Swiss boundaries / geoBoundaries gbOpen", "license": "CC BY 4.0", "boundaryID": "CHE-legacy"},
        "features": [{"type": "Feature", "id": "CH-BE", "properties": {"name": "Bern"}}],
    }
    path = tmp_path / "CHE-v2.json"
    path.write_text(json.dumps(cached))

    def unexpected_download(url):
        pytest.fail("Existing geography cache should not be downloaded again")

    monkeypatch.setattr(geography, "download", unexpected_download)
    with TestClient(create_app()) as client:
        response = client.get("/geographies/CHE")

    assert response.status_code == 200
    assert response.json() == cached
    assert geography.geography_path("CHE") == path
    assert geography.geography_info("CHE") == {"name": "Switzerland", "level": "region", "locations": {"CH-BE": "Bern"}}
    assert Choropleth(geography="CHE", locations=["CH-BE"], values=[1]).locations == ("CH-BE",)


def test_public_geography_info_does_not_expose_cached_mutable_data(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path))
    (tmp_path / "world-v2.json").write_text(
        json.dumps({"name": "World", "level": "country", "features": [{"id": "CHE", "properties": {"name": "Switzerland"}}]})
    )

    info = geography.geography_info()
    info["name"] = "Changed"
    info["locations"]["CHE"] = "Changed"
    info["locations"]["FRA"] = "France"

    assert geography.geography_info() == {"name": "World", "level": "country", "locations": {"CHE": "Switzerland"}}
    assert Choropleth(locations=["CHE"], values=[1]).locations == ("CHE",)
    with pytest.raises(ValueError, match="Unknown location for this geography"):
        Choropleth(locations=["FRA"], values=[1])


@pytest.mark.parametrize(
    ("failure", "status", "detail"),
    [
        pytest.param(lambda url: URLError("private upstream address"), 503, "Map data is unavailable", id="unavailable"),
        pytest.param(lambda url: HTTPError(url, 404, "not found", None, None), 404, "Unsupported geography", id="unsupported"),
    ],
)
def test_failed_download_does_not_cache_or_expose_upstream_details(monkeypatch, tmp_path, failure, status, detail):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path))

    def unavailable(url):
        raise failure(url)

    monkeypatch.setattr(geography, "download", unavailable)
    with TestClient(create_app()) as client:
        response = client.get("/geographies/CHE")
    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert not list(tmp_path.iterdir())


def test_download_uses_lfs_url_and_bounded_reads(monkeypatch):
    # Arrange
    calls = []
    read_sizes = []
    download_limit = geography.MAX_DOWNLOAD_BYTES

    class Response(BytesIO):
        def read(self, size=-1):
            read_sizes.append(size)
            return super().read(size)

    def open_url(url, *, timeout):
        calls.append((url, timeout))
        return Response(b"{}")

    monkeypatch.setattr(geography, "build_opener", lambda handler: SimpleNamespace(open=open_url))
    url = "https://github.com/wmgeolab/geoBoundaries/raw/revision/releaseData/gbOpen/CHE/ADM1/example.geojson"

    # Act
    # Assert
    assert geography.download(url) == {}
    assert calls[0][0].startswith("https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/revision/")
    assert calls[0][1] == 30
    monkeypatch.setattr(geography, "MAX_DOWNLOAD_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds"):
        geography.download(url)
    assert read_sizes == [download_limit + 1, 2]


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "http://github.com/file", "https://127.0.0.1/private", "https://github.com:444/file"]
)
def test_provider_downloads_and_redirects_cannot_target_other_origins(url):
    with pytest.raises(ValueError, match="provider URL"):
        geography.download(url)
    with pytest.raises(ValueError, match="provider URL"):
        geography._ProviderRedirect().redirect_request(Request(geography.WORLD_URL), None, 302, "Found", {}, url)
