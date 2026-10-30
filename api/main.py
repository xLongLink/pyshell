import json
import asyncio
from queue import Full, Queue
from fastapi import FastAPI
from pydantic import Field, BaseModel, JsonValue
from src.shell import PythonShell
from threading import Lock
from collections.abc import Iterator
from importlib.metadata import distributions
from starlette.responses import StreamingResponse


class Library(BaseModel):
    """An installed Python distribution."""

    name: str
    version: str


class ExecutionRequest(BaseModel):
    """Python source submitted for execution."""

    code: str


class ExecutionResult(BaseModel):
    """Captured output and completion state of an execution."""

    # Match the response fields to the captured execution state.
    ui: list[dict[str, JsonValue]] = Field(default_factory=list)
    done: bool
    stderr: str
    stdout: str
    exception: str | None


def create_app() -> FastAPI:
    """Create an API with a shared, serialized Python session."""
    app = FastAPI(title="PyShell execution API")
    shell = PythonShell()
    lock = Lock()
    executions: set[asyncio.Task[None]] = set()

    @app.get("/libraries", response_model=list[Library])
    def libraries() -> list[Library]:
        return sorted(
            [Library(name=package.metadata["Name"], version=package.version) for package in distributions()],
            key=lambda library: library.name.casefold(),
        )

    @app.get("/health", response_model=dict[str, str])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/", response_model=ExecutionResult)
    def execute(request: ExecutionRequest) -> dict[str, object]:
        with lock:
            return shell.execute(request.code)

    @app.post("/execute/stream")
    async def stream(request: ExecutionRequest) -> StreamingResponse:
        """Stream a submitted execution without coupling it to the HTTP connection."""
        queue: Queue[dict[str, object] | None] = Queue(maxsize=256)
        disconnected = False

        def emit(event: dict[str, object]) -> None:
            nonlocal disconnected
            if disconnected:
                return

            # Stop buffering if the HTTP reader falls behind execution.
            try:
                queue.put(event, timeout=1)
            except Full:
                disconnected = True

        def execute() -> None:
            try:
                with lock:
                    result = shell.execute(request.code, on_event=emit)
                emit({"type": "result", **{key: value for key, value in result.items() if key != "ui"}})

            # Terminate connected readers even when execution fails.
            finally:
                emit({"type": "execution_finished"})
                if not disconnected:
                    queue.put(None)

        task = asyncio.create_task(asyncio.to_thread(execute))
        executions.add(task)
        task.add_done_callback(executions.discard)

        def events() -> Iterator[str]:
            nonlocal disconnected
            try:
                # The sentinel ends the stream after all queued events.
                while True:
                    event = queue.get(timeout=180)
                    if event is None:
                        return
                    yield json.dumps(event, allow_nan=False) + "\n"

                    # A saturated queue has no sentinel; drain it before exiting.
                    if disconnected and queue.empty():
                        return
            finally:
                disconnected = True

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.post("/reset", response_model=dict[str, str])
    def reset() -> dict[str, str]:
        nonlocal shell
        with lock:
            shell.ui.close()
            shell = PythonShell()
        return {"status": "reset"}

    return app
