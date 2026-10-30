import pytest
from main import create_app
from fastapi.testclient import TestClient
from api.tests.geography_provider import install_geography_provider


@pytest.fixture
def geography_provider(monkeypatch, tmp_path):
    return install_geography_provider(monkeypatch, tmp_path)


@pytest.fixture
def api_client(geography_provider):
    with TestClient(create_app()) as client:
        yield client
