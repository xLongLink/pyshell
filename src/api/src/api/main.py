import json
import asyncio
from queue import Queue
from fastapi import Depends, FastAPI, Request, WebSocket, HTTPException
from pathlib import Path
from pydantic import BaseModel
from api.shell import PythonShell
from threading import Lock
from api.ui.state import UIState
from api.websocket import run_websocket
from collections.abc import Callable, Iterator, Awaitable
from importlib.metadata import distributions
from starlette.responses import Response, FileResponse, StreamingResponse
from api.ui.plot.geography import UnsupportedGeographyError, geography_path
from starlette.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware


class Library(BaseModel):
    """An installed Python distribution."""

    name: str
    version: str


class ExecutionRequest(BaseModel):
    """Python source submitted for execution."""

    code: str


def create_app() -> FastAPI:
    """Create an API with a shared, serialized Python session."""
    app = FastAPI(title="PyShell execution API")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    state = UIState()
    shell = PythonShell(state)
    lock = Lock()
    executions: set[asyncio.Task[None]] = set()

    def require_server_request(request: Request) -> None:
        if request.headers.get("origin") or request.headers.get("sec-fetch-mode"):
            raise HTTPException(status_code=403, detail="Execution is server-only")

    @app.middleware("http")
    async def frame_policy(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "frame-ancestors http://127.0.0.1:8010 http://localhost:8010; "
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/libraries", response_model=list[Library], dependencies=[Depends(require_server_request)])
    def libraries() -> list[Library]:
        return sorted(
            (Library(name=package.metadata["Name"], version=package.version) for package in distributions()),
            key=lambda library: library.name.casefold(),
        )

    @app.get("/health", response_model=dict[str, str])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/geographies/{geography}")
    def geography(geography: str) -> FileResponse:
        """Fetch or reuse one cached geography without executing Python."""
        try:
            path = geography_path(geography)
        except UnsupportedGeographyError as error:
            raise HTTPException(status_code=404, detail="Unsupported geography") from error
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=503, detail="Map data is unavailable") from error
        return FileResponse(path, media_type="application/geo+json", headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/ui/context", dependencies=[Depends(require_server_request)])
    def ui_context() -> dict[str, object]:
        return state.context()

    @app.websocket("/ui/ws")
    async def widgets(socket: WebSocket) -> None:
        if socket.headers.get("origin") not in {"http://127.0.0.1:8000", "http://localhost:8000"}:
            await socket.close(code=1008)
            return
        await socket.accept()
        queue = state.subscribe()

        async def send() -> None:
            while True:
                await socket.send_json(await queue.get())

        async def receive() -> None:
            await socket.receive_text()
            await socket.close(code=1008)

        try:
            await run_websocket(send(), receive())
        finally:
            state.unsubscribe(queue)

    @app.post("/execute/stream", dependencies=[Depends(require_server_request)])
    async def stream(request: ExecutionRequest) -> StreamingResponse:
        """Stream a submitted execution without coupling it to the HTTP connection."""
        queue: Queue[dict[str, object]] = Queue(maxsize=2)

        def execute() -> None:
            try:
                with lock:
                    result = shell.execute(request.code)
                queue.put({"type": "result", **result})

            # Terminate connected readers even when execution fails.
            finally:
                queue.put({"type": "execution_finished"})

        task = asyncio.create_task(asyncio.to_thread(execute))
        executions.add(task)
        task.add_done_callback(executions.discard)

        def events() -> Iterator[str]:
            while True:
                event = queue.get(timeout=180)

                # Encode one complete event per line.
                yield json.dumps(event, allow_nan=False) + "\n"

                if event["type"] == "execution_finished":
                    return

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.post("/reset", response_model=dict[str, str], dependencies=[Depends(require_server_request)])
    def reset() -> dict[str, str]:
        nonlocal shell
        with lock:
            shell.ui.close()
            state.reset()
            shell = PythonShell(state)
        return {"status": "reset"}

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True, check_dir=False))
    return app
