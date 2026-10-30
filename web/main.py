import asyncio
from typing import Literal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path
from pydantic import Field, BaseModel, ConfigDict, ValidationError
from contextlib import suppress, asynccontextmanager
from web.src.script import Agent, RemoteShell
from collections.abc import AsyncGenerator
from web.src.settings import settings
from concurrent.futures import ThreadPoolExecutor
from starlette.staticfiles import StaticFiles
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from starlette.middleware.trustedhost import TrustedHostMiddleware


class Command(BaseModel):
    """Validate browser commands before starting model or container work."""

    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["run"]
    instruction: str = Field(default="", max_length=8000)


class WidgetCommand(BaseModel):
    """Validate placement operations emitted by Python widgets."""

    # Placement actions address a validated, existing surface.
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["widget"]
    action: Literal["focus", "unfocus", "pin", "unpin"]
    surface_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")


class Workspace:
    """Own the single container's agent and shared browser transcript."""

    def __init__(self, agent: Agent) -> None:
        """Keep one conversation for the hardcoded execution container."""
        self.agent = agent
        self.events: list[dict[str, object]] = []
        self.listeners: set[asyncio.Queue[dict[str, object]]] = set()
        self.task: asyncio.Task[None] | None = None
        self.placements: dict[str, dict[str, bool]] = {}
        self.executor = ThreadPoolExecutor(max_workers=1, initializer=lambda: asyncio.set_event_loop(asyncio.new_event_loop()))

    def publish(self, event: dict[str, object]) -> None:
        """Record events on the event loop and broadcast them to connected tabs."""
        if event["type"] == "widget":
            self.place(WidgetCommand.model_validate(event))
            return
        placement_changed = False

        # Apply surface lifecycle changes before broadcasting their placement.
        if event["type"] == "a2ui":
            for message in event["messages"]:
                if "createSurface" in message:
                    surface_id = message["createSurface"]["surfaceId"]

                    # New surfaces start centered, evicting the oldest if needed.
                    self.placements[surface_id] = {"pinned": False}
                    self._focus(surface_id)
                    placement_changed = True

                # Deleting a surface also removes its authoritative placement.
                elif "deleteSurface" in message:
                    self.placements.pop(message["deleteSurface"]["surfaceId"], None)
                    placement_changed = True

        # Snapshots retain content, not transient state or placement events.
        if event["type"] not in {"state", "placement"}:
            self.events.append(event)
        for listener in self.listeners:
            listener.put_nowait(event)
        if placement_changed:
            self._placement_event()

    def _placement_event(self) -> None:
        self.publish({"type": "placement", "widgets": {key: dict(value) for key, value in self.placements.items()}})

    def _focus(self, surface_id: str) -> None:
        placement = self.placements.pop(surface_id)
        placement["pinned"] = False
        self.placements[surface_id] = placement
        centered = [value for value in self.placements.values() if not value["pinned"]]
        for oldest in centered[:-4]:
            oldest["pinned"] = True

    def place(self, command: WidgetCommand) -> None:
        """Keep one authoritative placement state for Python and all browser tabs."""
        placement = self.placements.get(command.surface_id)
        if placement is None:
            raise ValueError("Unknown or removed widget")

        # Focus and unpin share centered placement; the other actions pin.
        if command.action in {"focus", "unpin"}:
            self._focus(command.surface_id)
        else:
            placement["pinned"] = True

        # Every placement mutation is visible to all connected tabs.
        self._placement_event()

    def start(self, command: Command) -> None:
        """Start one operation without tying its lifetime to a browser socket."""
        if self.task is not None and not self.task.done():
            raise RuntimeError("The shared workspace is busy. Wait for the current operation to finish.")
        if not command.instruction.strip():
            raise ValueError("Enter an instruction.")

        # Mark the workspace busy before scheduling background execution.
        self.publish({"type": "state", "busy": True})
        self.publish({"type": "instruction", "text": command.instruction.strip()})
        self.task = asyncio.create_task(self.execute(command))

    async def execute(self, command: Command) -> None:
        """Serialize agent work and retain feedback even if the browser disconnects."""
        loop = asyncio.get_running_loop()

        def emit(event: dict[str, object]) -> None:
            loop.call_soon_threadsafe(self.publish, event)

        try:
            await loop.run_in_executor(self.executor, self.agent.run, command.instruction.strip(), emit)
            self.publish({"type": "done"})
        except Exception as error:
            self.publish({"type": "error", "text": str(error)})

        # Failed runs must also release the browser's busy state.
        finally:
            self.publish({"type": "state", "busy": False})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize the agent and shared workspace."""
    provider = OpenAIProvider(api_key=settings.openai_api_key.get_secret_value())
    model = OpenAIResponsesModel(settings.openai_model, provider=provider)
    agent = Agent(model, settings.openai_service_tier, shell=RemoteShell("http://127.0.0.1:8000"))
    workspace = Workspace(agent)
    app.state.workspace = workspace

    # Let submitted work finish before closing its provider and executor.
    try:
        yield
    finally:
        if workspace.task is not None:
            await workspace.task

        def close_provider() -> None:
            loop = asyncio.get_event_loop()
            try:
                loop.run_until_complete(provider.client.close())
            finally:
                loop.close()

        # Close the provider on the same thread and loop used for requests.
        try:
            await asyncio.get_running_loop().run_in_executor(workspace.executor, close_provider)
        finally:
            workspace.executor.shutdown(wait=True)


app = FastAPI(title="PyShell agent API", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.websocket("/ws")
async def workspace_socket(socket: WebSocket) -> None:
    """Exchange instructions and execution events with the local website."""
    if socket.headers.get("origin") not in {
        "http://127.0.0.1:8010",
        "http://localhost:8010",
        "http://127.0.0.1:5180",
        "http://localhost:5180",
    }:
        await socket.close(code=1008)
        return
    await socket.accept()

    # Register before taking the snapshot so subsequent events cannot be missed.
    workspace: Workspace = socket.app.state.workspace
    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    workspace.listeners.add(queue)
    queue.put_nowait(
        {
            "type": "snapshot",
            "events": list(workspace.events),
            "busy": workspace.task is not None and not workspace.task.done(),
            "model": workspace.agent.model,
            # Copy mutable placements so later changes cannot alter the snapshot.
            "placements": {key: dict(value) for key, value in workspace.placements.items()},
        }
    )

    async def send() -> None:
        while True:
            await socket.send_json(await queue.get())

    async def receive_command() -> None:
        text = await socket.receive_text()
        try:
            command = Command.model_validate_json(text)
            workspace.start(command)
        except (ValidationError, ValueError, RuntimeError) as error:
            queue.put_nowait({"type": "error", "text": str(error)})

            # Invalid commands do not change the active operation's state.
            queue.put_nowait({"type": "state", "busy": workspace.task is not None and not workspace.task.done()})

    async def receive() -> None:
        while True:
            await receive_command()

    sender = asyncio.create_task(send())
    receiver = asyncio.create_task(receive())

    # End the socket session if either direction disconnects or fails.
    try:
        done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()

    # Normal disconnects need only the socket-local cleanup below.
    except WebSocketDisconnect:
        pass
    finally:
        workspace.listeners.discard(queue)

        # Socket tasks are local; the shared workspace task keeps running.
        for task in (sender, receiver):
            task.cancel()

        # Await both cancellations to avoid leaving pending socket tasks.
        for task in (sender, receiver):
            with suppress(asyncio.CancelledError, WebSocketDisconnect):
                await task


frontend = StaticFiles(directory=Path(__file__).parent / "src" / "ui" / "dist", html=True, check_dir=False)
app.mount("/", frontend, name="website")
