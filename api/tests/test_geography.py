import pytest
from io import BytesIO
from main import create_app
from types import SimpleNamespace
from src.ui.plot import geography
from urllib.error import URLError, HTTPError
from urllib.request import Request
from fastapi.testclient import TestClient


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
    calls = []

    def open_url(url, *, timeout):
        calls.append((url, timeout))
        return BytesIO(b"{}")

    monkeypatch.setattr(geography, "build_opener", lambda handler: SimpleNamespace(open=open_url))
    url = "https://github.com/wmgeolab/geoBoundaries/raw/revision/releaseData/gbOpen/CHE/ADM1/example.geojson"
    assert geography.download(url) == {}
    assert calls[0][0].startswith("https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/revision/")
    assert calls[0][1] == 30
    monkeypatch.setattr(geography, "MAX_DOWNLOAD_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds"):
        geography.download(url)


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "http://github.com/file", "https://127.0.0.1/private", "https://github.com:444/file"]
)
def test_provider_downloads_and_redirects_cannot_target_other_origins(url):
    with pytest.raises(ValueError, match="provider URL"):
        geography.download(url)
    with pytest.raises(ValueError, match="provider URL"):
        geography._ProviderRedirect().redirect_request(Request(geography.WORLD_URL), None, 302, "Found", {}, url)
