import pytest
from main import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def api_client():
    with TestClient(create_app()) as client:
        yield client
