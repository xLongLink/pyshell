from fastapi import FastAPI
from pydantic import BaseModel
from api.shell import PythonShell
from threading import Lock


class ExecutionRequest(BaseModel):
    """Python source submitted for execution."""

    code: str


class ExecutionResult(BaseModel):
    """Captured output and completion state of an execution."""

    stdout: str
    stderr: str
    exception: str | None
    done: bool


def create_app() -> FastAPI:
    """Create an API with a shared, serialized Python session."""
    app = FastAPI(title="PyShell execution API")
    shell = PythonShell()
    lock = Lock()

    @app.get("/")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/", response_model=ExecutionResult)
    def execute(request: ExecutionRequest) -> dict[str, str | bool | None]:
        with lock:
            return shell.execute(request.code)

    @app.post("/reset")
    def reset() -> dict[str, str]:
        nonlocal shell
        with lock:
            shell = PythonShell()
        return {"status": "reset"}

    return app
