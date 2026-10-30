import os
import httpx
import pytest
from main import create_app
from contextlib import asynccontextmanager
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
def api_client():
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
