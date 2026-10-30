import os
import httpx
import pytest
from api.main import create_app
from importlib import import_module
from unittest.mock import patch
from web.src.shell import RemoteShell
from fastapi.testclient import TestClient
from pydantic_ai.messages import TextPart, ModelResponse
from pydantic_ai.models.function import FunctionModel
from api.tests.geography_provider import install_geography_provider

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
    import_module("web.src.settings")


@pytest.fixture
def execution_app(monkeypatch, tmp_path):
    install_geography_provider(monkeypatch, tmp_path)
    return create_app()


@pytest.fixture
def api_client(execution_app):
    with TestClient(execution_app) as client:
        yield client


@pytest.fixture
def remote_shell(execution_app):
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=execution_app), base_url="http://testserver")
    return RemoteShell(client)


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
