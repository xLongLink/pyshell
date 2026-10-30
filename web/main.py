import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path
from pydantic import TypeAdapter, ValidationError
from contextlib import suppress, asynccontextmanager
from web.src.voice import VoiceCommand, VoiceSession
from web.src.script import Agent, RemoteShell
from collections.abc import AsyncGenerator
from web.src.settings import settings
from web.src.workspace import Command, Workspace
from starlette.staticfiles import StaticFiles
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.realtime.openai import OpenAIRealtimeModel
from pydantic_ai.providers.openai import OpenAIProvider
from starlette.middleware.trustedhost import TrustedHostMiddleware

command_adapter = TypeAdapter(Command | VoiceCommand)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Own provider and workspace resources for the application lifetime."""
    provider = OpenAIProvider(api_key=settings.openai_api_key.get_secret_value())
    model = OpenAIResponsesModel(settings.openai_model, provider=provider)
    shell = RemoteShell("http://127.0.0.1:8000")
    await shell.open()
    workspace = Workspace(Agent(model, settings.openai_service_tier, shell=shell))
    app.state.workspace = workspace
    app.state.realtime_model = OpenAIRealtimeModel(settings.openai_realtime_model, provider=provider)
    try:
        yield
    finally:
        try:
            await workspace.close()
        finally:
            await provider.client.close()


app = FastAPI(title="PyShell agent API", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.websocket("/ws")
async def workspace_socket(socket: WebSocket) -> None:
    """Subscribe to shared state and own only this browser's voice call."""
    if socket.headers.get("origin") not in {
        "http://127.0.0.1:8010",
        "http://localhost:8010",
        "http://127.0.0.1:5180",
        "http://localhost:5180",
    }:
        await socket.close(code=1008)
        return
    await socket.accept()
    workspace: Workspace = socket.app.state.workspace
    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=256)
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
            try:
                command = command_adapter.validate_json(text)
                if isinstance(command, VoiceCommand):
                    await voice.handle(command)
                else:
                    workspace.start(command)
            except (ValidationError, ValueError, RuntimeError) as error:
                await queue.put({"type": "error", "text": str(error)})
                await queue.put({"type": "state", "busy": workspace.busy})

    sender = asyncio.create_task(send())
    receiver = asyncio.create_task(receive())
    try:
        done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass
    finally:
        workspace.listeners.discard(queue)
        for task in (sender, receiver):
            task.cancel()
        for task in (sender, receiver):
            with suppress(asyncio.CancelledError, WebSocketDisconnect):
                await task
        await voice.stop(notify=False)


frontend = StaticFiles(directory=Path(__file__).parent / "src" / "ui" / "dist", html=True, check_dir=False)
app.mount("/", frontend, name="website")
