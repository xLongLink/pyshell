import os
import httpx
import pytest
from main import create_app
from contextlib import asynccontextmanager
from src.ui.plot import geography
from unittest.mock import patch
from web.src.shell import RemoteShell
from fastapi.testclient import TestClient
from pydantic_ai.models.function import FunctionModel

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
    from web.src import settings as settings_module


@pytest.fixture
def agent_settings():
    return settings_module.settings


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path / "geographies"))

    def download(url):
        data_url = "https://raw.githubusercontent.com/wmgeolab/geoBoundaries/tests/CHE.geojson"
        if url == geography.METADATA_URL.format(geography="ZZZ"):
            return []
        if url == geography.METADATA_URL.format(geography="CHE"):
            return {
                "boundaryISO": "CHE",
                "boundaryType": "ADM1",
                "boundaryName": "Switzerland",
                "boundaryID": "CHE-ADM1-test",
                "boundarySource": "Test provider",
                "boundaryLicense": "Public domain",
                "licenseSource": "example.com/license",
                "boundaryYearRepresented": "2025",
                "admUnitCount": "2",
                "simplifiedGeometryGeoJSON": data_url,
            }
        assert url == data_url
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"shapeGroup": "CHE", "shapeType": "ADM1", "shapeISO": code, "shapeName": name},
                    "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
                }
                for code, name in {"CH-ZH": "Zürich", "CH-BE": "Bern"}.items()
            ],
        }

    monkeypatch.setattr(geography, "download", download)
    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def remote_shell(api_client, monkeypatch):
    shell = RemoteShell("http://testserver")

    async def open_transport():
        shell.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=api_client.app), base_url=shell.url)

    monkeypatch.setattr(shell, "open", open_transport)
    return shell


class CompletedFunctionModel(FunctionModel):
    @asynccontextmanager
    async def request_stream(self, *args, **kwargs):
        async with super().request_stream(*args, **kwargs) as response:
            response.finish_reason = "stop"
            yield response


@pytest.fixture
def model_for():
    def build(*responses, requests=None):
        responses = iter(responses)

        async def stream(messages, info):
            if requests is not None:
                requests.append((list(messages), info))

            # Responses may fail after yielding partial code.
            for chunk in next(responses):
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk

        return CompletedFunctionModel(stream_function=stream)

    return build
