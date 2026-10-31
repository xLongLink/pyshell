import httpx
import asyncio
import importlib
from types import SimpleNamespace
from fastapi import FastAPI
from pydantic import SecretStr
from sample.src.shell import RemoteShell
from pydantic_ai.messages import TextPart, ModelResponse
from sample.src.agents.script import Agent
from pydantic_ai.models.function import FunctionModel


def test_application_pairs_configured_execution_model_with_low_reasoning(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    main = importlib.import_module("sample.main")
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            openai_api_key=SecretStr("test-key"), openai_model="test-execution-model", openai_realtime_model="test-realtime-model"
        ),
    )

    async def run():
        app = FastAPI()
        async with main.lifespan(app):
            model = app.state.workspace.agent.agent.model
            assert model.model_name == "test-execution-model"
            assert model.settings == {"openai_reasoning_effort": "low"}
            assert app.state.realtime_model.model_name == "test-realtime-model"
        assert app.state.workspace.closing

    asyncio.run(run())


def test_execution_agent_preserves_injected_model_settings():
    received = []
    settings = {"temperature": 0.25, "max_tokens": 64}

    def respond(_messages, info):
        received.append(info.model_settings)
        return ModelResponse(parts=[TextPart("done()")], finish_reason="stop")

    def handle(request):
        if request.url.path == "/libraries":
            return httpx.Response(200, json=[])
        return httpx.Response(
            200, text=('{"type":"result","stdout":"","stderr":"","exception":null,"done":true}\n{"type":"execution_finished"}\n')
        )

    async def run():
        model = FunctionModel(respond, settings=settings)
        async with httpx.AsyncClient(base_url="http://execution", transport=httpx.MockTransport(handle)) as client:
            agent = Agent(model, shell=RemoteShell(client))
            assert await agent.arun("Finish") == "Task completed without printed output."
        assert received == [settings]

    asyncio.run(run())
