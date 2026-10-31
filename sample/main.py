import httpx
import asyncio
from httpx import AsyncClient
from typing import Literal
from fastapi import FastAPI, WebSocket
from pathlib import Path
from pydantic import Field, BaseModel, ConfigDict, TypeAdapter, ValidationError
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from sample.src.shell import RemoteShell
from fastapi.staticfiles import StaticFiles
from sample.src.settings import settings
from sample.src.workspace import Workspace
from sample.src.agents.voice import VoiceCommand, VoiceSession
from sample.src.agents.script import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from src.api.src.api.websocket import run_websocket
from pydantic_ai.realtime.openai import OpenAIRealtimeModel
from pydantic_ai.providers.openai import OpenAIProvider
from starlette.middleware.trustedhost import TrustedHostMiddleware


class Command(BaseModel):
    """Validate browser text commands before allocating work."""

    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["run"]
    instruction: str = Field(default="", max_length=8000)


command_adapter = TypeAdapter(Command | VoiceCommand)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Own provider and workspace resources for the application lifetime."""
    provider = OpenAIProvider(api_key=settings.openai_api_key.get_secret_value())
    try:
        async with AsyncClient(base_url="http://127.0.0.1:8000", timeout=httpx.Timeout(180, connect=10), trust_env=False) as client:
            model = OpenAIResponsesModel(settings.openai_model, provider=provider)
            workspace = Workspace(Agent(model, shell=RemoteShell(client)))
            app.state.workspace = workspace
            app.state.realtime_model = OpenAIRealtimeModel(settings.openai_realtime_model, provider=provider)

            # Finish accepted work before closing either transport.
            try:
                yield
            finally:
                await workspace.close()
    finally:
        await provider.client.close()


# Browser commands are accepted only through local hostnames and origins.
app = FastAPI(title="PyShell agent API", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.websocket("/ws")
async def workspace_socket(socket: WebSocket) -> None:
    """Subscribe to shared state and own only this browser's voice call."""
    if socket.headers.get("origin") not in {"http://127.0.0.1:8010", "http://localhost:8010"}:
        await socket.close(code=1008)
        return
    await socket.accept()
    workspace: Workspace = socket.app.state.workspace
    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=256)

    # Subscribe before taking the initial snapshot so no updates are missed.
    workspace.listeners.add(queue)
    queue.put_nowait(workspace.snapshot())

    async def emit(event: dict[str, object]) -> None:
        await queue.put(event)

    voice = VoiceSession(socket.app.state.realtime_model, workspace, emit)

    async def send() -> None:
        while True:
            await socket.send_json(await queue.get())

    async def receive() -> None:
        while True:
            text = await socket.receive_text()

            # Reject invalid commands without interrupting the subscription.
            try:
                command = command_adapter.validate_json(text)
                if isinstance(command, VoiceCommand):
                    await voice.handle(command)

                # Text tasks belong to the workspace, not this connection.
                else:
                    workspace.start(command.instruction)
            except (ValidationError, ValueError, RuntimeError) as error:
                await queue.put({"type": "error", "text": str(error)})

    try:
        await run_websocket(send(), receive())
    finally:
        workspace.listeners.discard(queue)
        await voice.stop(notify=False)


app.mount("/", StaticFiles(directory=Path(__file__).parent / "assets", html=True), name="assets")
