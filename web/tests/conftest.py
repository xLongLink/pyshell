import os
import httpx
import pytest
from main import create_app
from unittest.mock import patch
from web.src.shell import RemoteShell
from fastapi.testclient import TestClient
from pydantic_ai.messages import TextPart, ModelResponse
from pydantic_ai.models.function import FunctionModel
from api.tests.geography_provider import install_geography_provider

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
    from web.src import settings as settings_module


@pytest.fixture
def agent_settings():
    return settings_module.settings


@pytest.fixture
def api_client(monkeypatch, tmp_path):
    install_geography_provider(monkeypatch, tmp_path)
    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def remote_shell(api_client, monkeypatch):
    shell = RemoteShell("http://testserver")

    async def open_transport():
        shell.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=api_client.app), base_url=shell.url)

    monkeypatch.setattr(shell, "open", open_transport)
    return shell


@pytest.fixture
def model_for():
    def build(*responses, requests=None, finish_reason="stop"):
        responses = iter(responses)

        async def respond(messages, info):
            if requests is not None:
                requests.append((list(messages), info))

            content = []
            for chunk in next(responses):
                if isinstance(chunk, BaseException):
                    raise chunk
                content.append(chunk)
            return ModelResponse(parts=[TextPart("".join(content))], finish_reason=finish_reason)

        return FunctionModel(function=respond)

    return build
