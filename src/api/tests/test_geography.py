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


@pytest.mark.parametrize("metadata_list", [False, True])
@pytest.mark.parametrize("boundary_id", [None, "CHE-current"])
def test_geography_omits_boundary_id_and_preserves_source_metadata(monkeypatch, tmp_path, metadata_list, boundary_id):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path))
    metadata_url = geography.METADATA_URL.format(geography="CHE")
    download_url = "https://raw.githubusercontent.com/wmgeolab/geoBoundaries/main/CHE.geojson"
    metadata = {
        "boundaryISO": "CHE",
        "boundaryType": "ADM1",
        "boundaryName": "Switzerland",
        "boundarySource": "Swiss boundaries",
        "boundaryLicense": "CC BY 4.0",
        "licenseSource": "creativecommons.org/licenses/by/4.0/",
        "boundaryYearRepresented": 2024,
        "admUnitCount": "1",
        "simplifiedGeometryGeoJSON": download_url,
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
        "source": {
            "source": "Swiss boundaries / geoBoundaries gbOpen",
            "url": metadata_url,
            "license": "CC BY 4.0",
            "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
            "year": "2024",
        },
        "features": [{"type": "Feature", "id": "CH-BE", "properties": {"name": "Bern"}, "geometry": geometry}],
    }
    assert json.loads((tmp_path / "CHE-v2.json").read_text()) == response.json()
    assert geography.geography_info("CHE") == {"name": "Switzerland", "level": "region", "locations": {"CH-BE": "Bern"}}


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
    assert Choropleth(geography="CHE", locations=["CH-BE"], values=[1]).locations == ["CH-BE"]


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
    assert Choropleth(locations=["CHE"], values=[1]).locations == ["CHE"]
    with pytest.raises(ValueError, match="Unknown location for this geography"):
        Choropleth(locations=["FRA"], values=[1])


def test_failed_download_does_not_cache_or_expose_upstream_details(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path))

    def unavailable(url):
        raise URLError("private upstream address")

    monkeypatch.setattr(geography, "download", unavailable)
    with TestClient(create_app()) as client:
        response = client.get("/geographies/CHE")
    assert response.status_code == 503
    assert response.json() == {"detail": "Map data is unavailable"}
    assert not list(tmp_path.iterdir())


def test_missing_provider_metadata_is_an_unsupported_geography(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path))

    def missing(url):
        raise HTTPError(url, 404, "not found", None, None)

    monkeypatch.setattr(geography, "download", missing)
    with TestClient(create_app()) as client:
        response = client.get("/geographies/CHE")
    assert response.status_code == 404
    assert response.json() == {"detail": "Unsupported geography"}
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
