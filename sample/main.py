import os
import httpx2
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import FastAPI, Request, WebSocket, HTTPException
from pathlib import Path
from contextlib import suppress, asynccontextmanager
from collections.abc import AsyncGenerator
from sample.src.shell import RemoteShell
from fastapi.staticfiles import StaticFiles
from sample.src.settings import Settings
from sample.src.workspace import Workspace, Subscription
from starlette.websockets import WebSocketDisconnect
from sample.src.agents.voice import VoiceSession, voice_command_adapter
from sample.src.agents.script import Agent, LoggedModel
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from src.api.src.api.websocket import run_websocket
from pydantic_ai.providers.openai import OpenAIProvider
from starlette.middleware.trustedhost import TrustedHostMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Own provider and workspace resources for the application lifetime."""
    load_dotenv(Path(__file__).parent / ".env", override=False)
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    settings = Settings()
    async with AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value(), max_retries=0) as openai:
        provider = OpenAIProvider(openai_client=openai)
        async with (
            httpx2.AsyncClient(base_url="http://127.0.0.1:8000", timeout=httpx2.Timeout(180, connect=10), trust_env=False) as client,
            LoggedModel(
                OpenAIResponsesModel(
                    "gpt-6.1-sol",
                    provider=provider,
                    settings=OpenAIResponsesModelSettings(
                        openai_reasoning_effort="low", extra_body={"service_tier": settings.openai_worker_service_tier}
                    ),
                )
            ) as model,
        ):
            workspace = Workspace(Agent(model, shell=RemoteShell(client)))
            app.state.workspace = workspace
            app.state.openai = openai
            app.state.settings = settings
            app.state.voices = set()

            # Finish accepted work before closing either transport.
            try:
                yield
            finally:
                await workspace.close()


# Browser commands are accepted only through local hostnames and origins.
app = FastAPI(title="PyShell agent API", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.post("/reset")
async def reset_workspace(request: Request) -> dict[str, str]:
    """Reset the shared workspace before a browser page starts using it."""
    if request.headers.get("origin") not in {"http://127.0.0.1:8010", "http://localhost:8010"}:
        raise HTTPException(status_code=403, detail="Reset requires a local browser origin.")
    workspace: Workspace = request.app.state.workspace
    if workspace.busy or workspace.resetting or workspace.closing:
        raise HTTPException(status_code=409, detail="The workspace is busy. Wait for work to finish, then reload.")
    if workspace.agent.shell.uncertain:
        raise HTTPException(status_code=409, detail="Execution state is uncertain. Restart the execution container and web server.")
    workspace.resetting = True
    try:
        for subscription in tuple(workspace.listeners):
            subscription.close()
        await asyncio.gather(*(voice.stop() for voice in tuple(request.app.state.voices)))
        await workspace.agent.reset()
        workspace.recent_tasks.clear()
        workspace.task = None
    except Exception as error:
        raise HTTPException(status_code=502, detail="Workspace reset failed. Restart the execution container and web server.") from error
    finally:
        workspace.resetting = False
    return {"status": "reset"}


@app.websocket("/ws")
async def workspace_socket(socket: WebSocket) -> None:
    """Subscribe to shared state and own only this browser's voice call."""
    if socket.headers.get("origin") not in {"http://127.0.0.1:8010", "http://localhost:8010"}:
        await socket.close(code=1008)
        return
    await socket.accept()
    workspace: Workspace = socket.app.state.workspace
    if workspace.resetting:
        await socket.close(code=1013)
        return
    subscription = Subscription(workspace)
    settings: Settings = socket.app.state.settings
    voice = VoiceSession(
        socket.app.state.openai,
        workspace,
        subscription.publish,
        model=settings.openai_live_model,
        backend_model=settings.openai_voice_backend_model,
    )
    socket.app.state.voices.add(voice)

    async def send() -> None:
        while not subscription.closed.is_set():
            await socket.send_json(await subscription.queue.get())

    async def receive() -> None:
        while not subscription.closed.is_set():
            text = await socket.receive_text()
            if subscription.closed.is_set():
                return

            # Reject invalid commands without interrupting the subscription.
            try:
                command = voice_command_adapter.validate_json(text)
                await voice.handle(command)
            except (ValueError, RuntimeError) as error:
                subscription.publish({"type": "error", "text": str(error)})

    try:
        await run_websocket(send(), receive(), subscription.closed.wait())
    finally:
        overloaded = subscription.closed.is_set()
        subscription.close()
        try:
            if overloaded:
                with suppress(WebSocketDisconnect):
                    async with asyncio.timeout(5):
                        await socket.close(code=1013)
        finally:
            try:
                await voice.stop(notify=False)
            finally:
                socket.app.state.voices.discard(voice)


app.mount("/", StaticFiles(directory=Path(__file__).parent / "assets", html=True), name="assets")
