import os
import re
import json
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from functools import lru_cache
from threading import Lock
from collections import Counter, defaultdict
from http.client import HTTPMessage, HTTPResponse
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener

METADATA_URL = "https://www.geoboundaries.org/api/current/gbOpen/{geography}/ADM1/"
WORLD_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/9380cca83db5f9aef52d5e762765100745f84b27/geojson/ne_110m_admin_0_countries.geojson"
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
_locks = defaultdict(Lock)


class UnsupportedGeographyError(ValueError):
    """The requested country or administrative level is unavailable."""


def _download_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname
        not in {"www.geoboundaries.org", "geoboundaries.org", "github.com", "raw.githubusercontent.com", "media.githubusercontent.com"}
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Unexpected boundary provider URL")


class _ProviderRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, request: Request, response: HTTPResponse, code: int, message: str, headers: HTTPMessage, url: str
    ) -> Request | None:
        _download_url(url)
        return super().redirect_request(request, response, code, message, headers, url)


def download(url: str) -> dict | list:
    """Read provider metadata or geometry with size and network limits."""
    _download_url(url)
    url = url.replace("https://github.com/wmgeolab/geoBoundaries/raw/", "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/")
    with build_opener(_ProviderRedirect()).open(url, timeout=30) as response:
        content = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Geography download exceeds 32 MiB")
    return json.loads(content)


def _source(geography: str) -> tuple[str, dict]:
    if geography == "world":
        return "World", {
            "source": "Natural Earth, 1:110m Admin 0 Countries",
            "url": WORLD_URL,
            "downloadUrl": WORLD_URL,
            "license": "Public domain",
            "licenseUrl": "https://www.naturalearthdata.com/about/terms-of-use/",
        }
    url = METADATA_URL.format(geography=geography)
    try:
        metadata = download(url)
    except HTTPError as error:
        if error.code == 404:
            raise UnsupportedGeographyError("Unsupported geography") from error
        raise
    if isinstance(metadata, list):
        matches = [
            item
            for item in metadata
            if isinstance(item, dict) and item.get("boundaryISO") == geography and item.get("boundaryType") == "ADM1"
        ]
        if not matches:
            raise UnsupportedGeographyError("Unsupported geography")
        if len(matches) != 1:
            raise ValueError("Ambiguous geography metadata")
        metadata = matches[0]
    if not isinstance(metadata, dict) or metadata.get("boundaryISO") != geography or metadata.get("boundaryType") != "ADM1":
        raise ValueError("Unexpected geography metadata")
    fields = ("boundaryName", "boundarySource", "boundaryLicense", "simplifiedGeometryGeoJSON")
    if any(not isinstance(metadata.get(field), str) or not metadata[field].strip() for field in fields):
        raise ValueError("Incomplete geography metadata")
    _download_url(metadata["simplifiedGeometryGeoJSON"])
    license_url = metadata.get("licenseSource")
    if isinstance(license_url, str) and license_url.strip():
        license_url = license_url.strip()
        if "://" not in license_url:
            license_url = "https://" + license_url
        parsed = urlsplit(license_url)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or "." not in parsed.hostname
            or any(char.isspace() for char in parsed.netloc)
        ):
            license_url = url
    else:
        license_url = url
    try:
        count = int(metadata["admUnitCount"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid administrative unit count") from error
    if count < 1:
        raise ValueError("Invalid administrative unit count")
    return metadata["boundaryName"], {
        "source": metadata["boundarySource"] + " / geoBoundaries gbOpen",
        "url": url,
        "downloadUrl": metadata["simplifiedGeometryGeoJSON"],
        "license": metadata["boundaryLicense"],
        "licenseUrl": license_url,
        "year": str(metadata.get("boundaryYearRepresented", "")),
        "count": count,
    }


def _validate_geometry(geometry: dict) -> None:
    polygons = [geometry.get("coordinates")] if geometry["type"] == "Polygon" else geometry.get("coordinates")
    if not isinstance(polygons, list) or not polygons:
        raise ValueError("Missing geographic coordinates")
    for polygon in polygons:
        if not isinstance(polygon, list) or not polygon:
            raise ValueError("Invalid geographic polygon")
        for ring in polygon:
            if not isinstance(ring, list) or len(ring) < 4 or ring[0] != ring[-1]:
                raise ValueError("Invalid geographic ring")
            for position in ring:
                if (
                    not isinstance(position, list)
                    or len(position) != 2
                    or any(type(value) not in (int, float) or not isfinite(value) for value in position)
                ):
                    raise ValueError("Invalid geographic position")


def normalize(geography: str, name: str, source: dict, data: dict | list) -> dict:
    """Derive geographic identifiers and labels from the downloaded features."""
    if (
        not isinstance(data, dict)
        or data.get("type") != "FeatureCollection"
        or not isinstance(data.get("features"), list)
        or not data["features"]
    ):
        raise ValueError("Invalid geography response")
    if any(not isinstance(feature, dict) or not isinstance(feature.get("properties"), dict) for feature in data["features"]):
        raise ValueError("Invalid geographic properties")
    if any(not isinstance(feature["properties"].get("shapeISO") or "", str) for feature in data["features"]):
        raise ValueError("Invalid geographic identifier")
    codes = [(feature["properties"].get("shapeISO") or "").strip().upper() for feature in data["features"]]
    counts = Counter(codes)
    features = []
    seen = set()
    for feature, code in zip(data["features"], codes):
        properties = feature["properties"]
        if geography == "world":
            code = properties.get("ISO_A3_EH")
            if not isinstance(code, str) or not re.fullmatch(r"[A-Z]{3}", code):
                continue
            label = properties.get("NAME_EN") or properties.get("NAME") or code
        else:
            if properties.get("shapeGroup") != geography or properties.get("shapeType") != "ADM1":
                raise ValueError("Unexpected administrative boundaries")
            if not re.fullmatch(r"[A-Z]{2,3}-[A-Z0-9-]+", code) or counts[code] != 1:
                identifier = properties.get("shapeID")
                if not isinstance(identifier, str) or not identifier:
                    raise ValueError("Missing geographic identifier")
                code = f"{geography}-{identifier}"
            if geography == "FRA" and code == "FR-20R":
                code = "FR-COR"
            label = properties.get("shapeName") or code
        if not isinstance(label, str):
            raise ValueError("Invalid geographic name")
        if re.search("[ÃÂ]", label):
            try:
                label = label.encode("latin1").decode("utf8")
            except UnicodeError:
                pass
        geometry = feature.get("geometry")
        if code in seen or not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Unexpected geographic feature")
        _validate_geometry(geometry)
        seen.add(code)
        features.append({"type": "Feature", "id": code, "properties": {"name": label}, "geometry": geometry})
    if not features or (geography != "world" and len(features) != source["count"]):
        raise ValueError("Provider returned incomplete boundaries")
    return {
        "type": "FeatureCollection",
        "geography": geography,
        "name": name,
        "level": "country" if geography == "world" else "region",
        "source": {key: value for key, value in source.items() if key not in {"downloadUrl", "count"}},
        "features": features,
    }


def geography_path(geography: str) -> Path:
    """Download a geography once and atomically cache it outside the source tree."""
    if not re.fullmatch(r"world|[A-Z]{3}", geography):
        raise UnsupportedGeographyError("Unsupported geography")
    default_cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "pyshell/geographies"
    directory = Path(os.environ.get("PYSHELL_GEOGRAPHY_CACHE") or default_cache)
    path = directory / f"{geography}-v2.json"
    with _locks[geography]:
        if path.is_file():
            return path
        name, source = _source(geography)
        data = normalize(geography, name, source, download(source["downloadUrl"]))
        directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                json.dump(data, output, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return path


@lru_cache(maxsize=256)
def _lookup(path: Path, _revision: int) -> dict:
    data = json.loads(path.read_text())
    return {
        "name": data["name"],
        "level": data["level"],
        "locations": {feature["id"]: feature["properties"]["name"] for feature in data["features"]},
    }


def _geography_info(geography: str) -> dict:
    path = geography_path(geography)
    return _lookup(path, path.stat().st_mtime_ns)


def geography_info(geography: str = "world") -> dict:
    """Return the selected map's name, level, and available location codes."""
    info = _geography_info(geography)
    return {**info, "locations": dict(info["locations"])}
