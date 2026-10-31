import httpx
import asyncio
from httpx import AsyncClient
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from pathlib import Path
from pydantic import TypeAdapter, ValidationError
from contextlib import suppress, asynccontextmanager
from collections.abc import AsyncGenerator
from sample.src.page import render_page
from sample.src.shell import RemoteShell
from sample.src.settings import settings
from starlette.responses import FileResponse, HTMLResponse
from sample.src.workspace import Command, Workspace
from sample.src.agents.voice import VoiceCommand, VoiceSession
from sample.src.agents.script import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.realtime.openai import OpenAIRealtimeModel
from pydantic_ai.providers.openai import OpenAIProvider
from starlette.middleware.trustedhost import TrustedHostMiddleware

command_adapter = TypeAdapter(Command | VoiceCommand)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Own provider and workspace resources for the application lifetime."""
    provider = OpenAIProvider(api_key=settings.openai_api_key.get_secret_value())
    try:
        async with AsyncClient(
            base_url="http://127.0.0.1:8000",
            timeout=httpx.Timeout(180, connect=10),
            trust_env=False,
        ) as client:
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
                    workspace.start(command)
            except (ValidationError, ValueError, RuntimeError) as error:
                await queue.put({"type": "error", "text": str(error)})

    sender = asyncio.create_task(send())
    receiver = asyncio.create_task(receive())

    # Either socket direction ending must release both tasks and the voice call.
    try:
        done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()

    # Browser disconnects are normal; other task failures remain visible.
    except WebSocketDisconnect:
        pass
    finally:
        workspace.listeners.discard(queue)

        # Cancel both directions before awaiting either one.
        for task in (sender, receiver):
            task.cancel()

        # Await cancellation before releasing the socket's voice resources.
        for task in (sender, receiver):
            with suppress(asyncio.CancelledError, WebSocketDisconnect):
                await task
        await voice.stop(notify=False)


@app.get("/", response_class=HTMLResponse)
def website(request: Request) -> str:
    """Serve the sample without a frontend build."""
    return render_page(f"{request.url.scheme}://{request.url.hostname}:8000/")


@app.get("/voice.js")
def voice_script() -> FileResponse:
    """Serve the browser's microphone and voice signaling controller."""
    return FileResponse(Path(__file__).parent / "assets" / "voice.mjs", media_type="text/javascript")
