import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from script import RemoteShell


@pytest.fixture
def api_client():
    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def remote_shell(api_client, monkeypatch):
    def post(url, **kwargs):
        kwargs.pop("timeout", None)
        kwargs.pop("trust_env", None)
        return api_client.post(url, **kwargs)

    monkeypatch.setattr("script.httpx.post", post)
    return RemoteShell("http://testserver")
