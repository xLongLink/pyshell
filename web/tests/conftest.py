import pytest
from main import create_app
from contextlib import asynccontextmanager
from web.src.script import RemoteShell
from fastapi.testclient import TestClient
from pydantic_ai.models.function import FunctionModel


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

    monkeypatch.setattr("web.src.script.httpx.post", post)
    return RemoteShell("http://testserver")


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
