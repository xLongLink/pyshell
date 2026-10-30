from fastapi import FastAPI
from pydantic import Field, BaseModel, JsonValue
from src.shell import PythonShell
from threading import Lock
from importlib.metadata import distributions


class Library(BaseModel):
    """An installed Python distribution."""

    name: str
    version: str


class ExecutionRequest(BaseModel):
    """Python source submitted for execution."""

    code: str


class ExecutionResult(BaseModel):
    """Captured output and completion state of an execution."""

    done: bool
    stderr: str
    stdout: str
    exception: str | None
    ui: list[dict[str, JsonValue]] = Field(default_factory=list)


def create_app() -> FastAPI:
    """Create an API with a shared, serialized Python session."""
    app = FastAPI(title="PyShell execution API")
    shell = PythonShell()
    lock = Lock()

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

    @app.post("/reset", response_model=dict[str, str])
    def reset() -> dict[str, str]:
        nonlocal shell
        with lock:
            shell.ui.close()
            shell = PythonShell()
        return {"status": "reset"}

    return app
