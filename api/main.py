from threading import Lock

from fastapi import FastAPI
from pydantic import BaseModel

from api.shell import PythonShell


class ExecutionRequest(BaseModel):
    code: str


class ExecutionResult(BaseModel):
    stdout: str
    stderr: str
    exception: str | None
    done: bool


def create_app():
    app = FastAPI(title="PyShell execution API")
    shell = PythonShell()
    lock = Lock()

    @app.get("/")
    def health():
        return {"status": "ok"}

    @app.post("/", response_model=ExecutionResult)
    def execute(request: ExecutionRequest):
        with lock:
            return shell.execute(request.code)

    @app.post("/reset")
    def reset():
        nonlocal shell
        with lock:
            shell = PythonShell()
        return {"status": "reset"}

    return app
