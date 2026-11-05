"""Lifecycle of an execution container reserved for benchmark trials."""

import os
import fcntl
import asyncio
from pathlib import Path
from contextlib import contextmanager
from collections.abc import Generator

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def backend_lock() -> Generator[None, None, None]:
    """The Compose project has one workspace, regardless of requested port."""
    (ROOT / "results").mkdir(exist_ok=True)
    with (ROOT / "results" / "backend.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another benchmark owns the execution backend") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class ContainerBackend:
    """Recreate the isolated execution service before each trial."""

    def __init__(self, port: int, log: Path) -> None:
        self.url = f"http://127.0.0.1:{port}"
        self.log = log
        self.env = {**os.environ, "BENCHMARK_PORT": str(port)}
        self.compose = Path(__file__).resolve().parents[1] / "compose.yaml"

    async def command(self, *arguments: str) -> None:
        with self.log.open("ab") as output:
            process = await asyncio.create_subprocess_exec(
                "docker", "compose", "-f", str(self.compose), *arguments, stdout=output, stderr=output, env=self.env
            )
            try:
                async with asyncio.timeout(600):
                    returncode = await process.wait()
            except BaseException:
                process.kill()
                await process.wait()
                raise
        if returncode:
            raise RuntimeError(f"Docker Compose failed ({returncode}); see {self.log}")

    async def build(self) -> None:
        await self.command("build", "pyshell")

    async def fresh(self) -> None:
        await self.command("up", "--detach", "--force-recreate", "--wait", "--wait-timeout", "90", "pyshell")

    async def close(self) -> None:
        await self.command("down", "--remove-orphans")
