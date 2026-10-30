import json
import pytest
from io import BytesIO
from copy import deepcopy
from types import SimpleNamespace
from src.ui.plot import geography
from urllib.error import URLError, HTTPError
from urllib.request import Request
from concurrent.futures import ThreadPoolExecutor


def metadata(provider, country="CHE"):
    return provider["responses"][geography.METADATA_URL.format(geography=country)]


def payload(provider, country="CHE"):
    return provider["responses"][metadata(provider, country)["simplifiedGeometryGeoJSON"]]


@pytest.mark.parametrize(
    ("country", "codes"), [("world", ["FRA", "DEU", "ESP"]), ("CHE", ["CH-ZH", "CH-BE"]), ("FRA", ["FR-IDF", "FR-BRE", "FR-COR"])]
)
def test_metadata_and_all_boundaries_are_downloaded_once_and_cached_together(api_client, geography_provider, monkeypatch, country, codes):
    response = api_client.get(f"/geographies/{country}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/geo+json"
    assert response.headers["cache-control"] == "public, max-age=3600"
    assert response.headers["etag"]
    data = response.json()
    assert [feature["id"] for feature in data["features"]] == codes
    assert data["source"]["license"]
    assert "downloadUrl" not in data["source"]
    assert "count" not in data["source"]
    if country != "world":
        assert data["source"]["boundaryID"] == metadata(geography_provider, country)["boundaryID"]
    expected = (
        [geography.WORLD_URL]
        if country == "world"
        else [geography.METADATA_URL.format(geography=country), metadata(geography_provider, country)["simplifiedGeometryGeoJSON"]]
    )
    assert geography_provider["calls"] == expected

    def offline(url):
        raise URLError("offline")

    monkeypatch.setattr(geography, "download", offline)
    assert api_client.get(f"/geographies/{country}").content == response.content
    assert list(geography.geography_info(country)["locations"]) == codes
    assert len(list(geography_provider["cache"].glob("*.json"))) == 1
    assert not list(geography_provider["cache"].glob("*.tmp"))


def test_failed_download_can_be_retried_without_cached_error(api_client, geography_provider, monkeypatch):
    def fail(url):
        raise URLError("private upstream address")

    monkeypatch.setattr(geography, "download", fail)
    response = api_client.get("/geographies/CHE")
    assert response.status_code == 503
    assert response.json() == {"detail": "Map data is unavailable"}
    assert not list(geography_provider["cache"].glob("*"))
    monkeypatch.setattr(geography, "download", geography_provider["download"])
    assert api_client.get("/geographies/CHE").status_code == 200


def test_existing_cache_with_download_bookkeeping_remains_readable(api_client, geography_provider, monkeypatch):
    path = geography.geography_path("CHE")
    data = json.loads(path.read_text())
    data["source"].update(downloadUrl=metadata(geography_provider)["simplifiedGeometryGeoJSON"], count=2)
    path.write_text(json.dumps(data))

    def offline(url):
        raise URLError("offline")

    monkeypatch.setattr(geography, "download", offline)
    response = api_client.get("/geographies/CHE")
    assert response.status_code == 200
    assert response.json() == data
    assert geography.geography_info("CHE")["locations"] == {"CH-ZH": "Zürich", "CH-BE": "Bern"}


@pytest.mark.parametrize("invalid", ["root", "missing", "wrong-country", "duplicate", "geometry", "coordinates"])
def test_invalid_provider_geometry_is_not_cached(api_client, geography_provider, invalid):
    data = payload(geography_provider)
    if invalid == "root":
        geography_provider["responses"][metadata(geography_provider)["simplifiedGeometryGeoJSON"]] = []
    elif invalid == "missing":
        data["features"].pop()
    elif invalid == "wrong-country":
        data["features"][0]["properties"]["shapeGroup"] = "USA"
    elif invalid == "duplicate":
        data["features"].append(deepcopy(data["features"][0]))
    elif invalid == "geometry":
        data["features"][0]["geometry"]["type"] = "Point"
    else:
        data["features"][0]["geometry"]["coordinates"][0][1] = ["bad", 1]
    assert api_client.get("/geographies/CHE").status_code == 503
    assert not list(geography_provider["cache"].glob("*"))


@pytest.mark.parametrize("invalid", ["country", "name", "count", "url", "ambiguous"])
def test_invalid_provider_metadata_is_not_cached(api_client, geography_provider, invalid):
    data = metadata(geography_provider)
    if invalid == "country":
        data["boundaryISO"] = "USA"
    elif invalid == "name":
        data.pop("boundaryName")
    elif invalid == "count":
        data["admUnitCount"] = "unknown"
    elif invalid == "url":
        data["simplifiedGeometryGeoJSON"] = "http://127.0.0.1/private"
    else:
        geography_provider["responses"][geography.METADATA_URL.format(geography="CHE")] = [data, data]
    assert api_client.get("/geographies/CHE").status_code == 503
    assert not list(geography_provider["cache"].glob("*"))


def test_metadata_lists_select_exact_administrative_level(api_client, geography_provider):
    data = metadata(geography_provider, "IND")
    wrong_level = {**data, "boundaryType": "ADM0", "admUnitCount": "1"}
    geography_provider["responses"][geography.METADATA_URL.format(geography="IND")] = [wrong_level, data]
    response = api_client.get("/geographies/IND")
    assert response.status_code == 200
    assert response.json()["source"]["boundaryID"] == data["boundaryID"]


@pytest.mark.parametrize("missing_iso", [True, False])
def test_missing_or_duplicate_iso_codes_use_source_identifiers(geography_provider, missing_iso):
    data = payload(geography_provider)
    for feature in data["features"]:
        feature["properties"]["shapeISO"] = "" if missing_iso else "CH-DUP"
    info = geography.geography_info("CHE")
    assert info["locations"] == {"CHE-shape-0": "Zürich", "CHE-shape-1": "Bern"}


def test_unknown_country_can_become_available_without_a_deployment(api_client, geography_provider):
    assert api_client.get("/geographies/ZZZ").status_code == 404
    info = deepcopy(metadata(geography_provider))
    info.update(boundaryISO="ZZZ", boundaryName="New country", boundaryID="ZZZ-ADM1-v1")
    info["simplifiedGeometryGeoJSON"] = "https://raw.githubusercontent.com/wmgeolab/geoBoundaries/tests/ZZZ.geojson"
    data = deepcopy(payload(geography_provider))
    for index, feature in enumerate(data["features"]):
        feature["properties"].update(shapeGroup="ZZZ", shapeISO=f"ZZ-{index}")
    geography_provider["responses"][geography.METADATA_URL.format(geography="ZZZ")] = info
    geography_provider["responses"][info["simplifiedGeometryGeoJSON"]] = data
    assert api_client.get("/geographies/ZZZ").status_code == 200


def test_metadata_not_found_is_an_unsupported_geography(api_client, geography_provider, monkeypatch):
    def missing(url):
        raise HTTPError(url, 404, "not found", None, None)

    monkeypatch.setattr(geography, "download", missing)
    response = api_client.get("/geographies/CHE")
    assert response.status_code == 404
    assert response.json() == {"detail": "Unsupported geography"}
    assert not list(geography_provider["cache"].glob("*"))


def test_metadata_version_and_lookup_refresh_when_cache_is_removed(geography_provider):
    first = geography.geography_info("CHE")
    path = geography.geography_path("CHE")
    metadata(geography_provider)["boundaryID"] = "CHE-ADM1-v2"
    payload(geography_provider)["features"][1]["properties"].update(shapeISO="CH-NEW", shapeName="New division")
    assert geography.geography_info("CHE") == first
    path.unlink()
    assert geography.geography_info("CHE")["locations"]["CH-NEW"] == "New division"
    assert json.loads(path.read_text())["source"]["boundaryID"] == "CHE-ADM1-v2"


def test_lookup_callers_cannot_mutate_cached_location_codes(geography_provider):
    info = geography.geography_info("CHE")
    info["locations"].clear()
    assert geography.geography_info("CHE")["locations"] == {"CH-ZH": "Zürich", "CH-BE": "Bern"}


def test_simultaneous_requests_share_one_complete_cached_file(geography_provider):
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(geography.geography_path, ["CHE"] * 4))
    assert len(set(paths)) == 1
    assert len(geography_provider["calls"]) == 2
    assert json.loads(paths[0].read_text())["geography"] == "CHE"
    assert not list(geography_provider["cache"].glob("*.tmp"))


def test_default_cache_uses_xdg_directory(geography_provider, monkeypatch, tmp_path):
    monkeypatch.delenv("PYSHELL_GEOGRAPHY_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert geography.geography_path("CHE").parent == tmp_path / "xdg/pyshell/geographies"


def test_license_links_are_derived_and_invalid_source_links_fall_back_to_metadata(api_client, geography_provider):
    metadata(geography_provider)["licenseSource"] = "Data https"
    response = api_client.get("/geographies/CHE")
    assert response.status_code == 200
    assert response.json()["source"]["licenseUrl"] == geography.METADATA_URL.format(geography="CHE")


def test_geography_reads_do_not_execute_python_or_publish_output(api_client):
    api_client.post("/", json={"code": "answer = 42"})
    assert api_client.get("/geographies/CHE").status_code == 200
    result = api_client.post("/", json={"code": "print(answer)"}).json()
    assert result["stdout"] == "42\n"
    assert result["ui"] == []


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
