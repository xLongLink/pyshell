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
