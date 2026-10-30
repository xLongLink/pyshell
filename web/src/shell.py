import httpx
import asyncio
from typing import TypedDict
from web.src.events import execution_event
from collections.abc import Callable


class ExecutionResult(TypedDict):
    """Output returned by the execution API."""

    stdout: str
    stderr: str
    exception: str | None
    done: bool
    ui: list[dict[str, object]]


# Requests bypass proxy settings for the local execution server.
class RemoteShell:
    """Access the container's persistent Python session."""

    def __init__(self, url: str = "http://127.0.0.1:8000") -> None:
        """Select the execution API endpoint."""
        self.url = url.rstrip("/")
        self.client: httpx.AsyncClient | None = None
        self.uncertain = False

    async def open(self) -> None:
        """Open the workspace-owned execution transport."""
        self.client = httpx.AsyncClient(base_url=self.url, timeout=httpx.Timeout(180, connect=10), trust_env=False)

    async def close(self) -> None:
        """Release the execution transport."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def inventory(self) -> list[dict[str, str]]:
        """Read installed packages through the asynchronous transport."""
        if self.client is None:
            raise RuntimeError("Execution transport is closed")
        response = await self.client.get("/libraries")
        response.raise_for_status()
        return response.json()

    async def stream(self, code: str, emit: Callable[[dict[str, object]], None]) -> ExecutionResult:
        """Submit once and consume ordered output until completion is confirmed."""
        if self.client is None:
            raise RuntimeError("Execution transport is closed")
        if self.uncertain:
            raise RuntimeError("Execution state is uncertain. Restart the execution container and web server before submitting more work.")

        # Require both a result and the terminal marker before confirming success.
        result = None
        finished = False
        received = 0

        # Once submitted, a broken stream leaves execution state uncertain.
        try:
            async with asyncio.timeout(180), self.client.stream("POST", "/execute/stream", json={"code": code}) as response:
                response.raise_for_status()

                # Bound the aggregate stream, not only individual output events.
                async for line in response.aiter_lines():
                    received += len(line.encode())

                    # Reject excess output before decoding more protocol data.
                    if received > 16_777_216:
                        raise RuntimeError("Execution stream exceeded 16 MiB")

                    # Validate each event before applying completion transitions.
                    event = execution_event.validate_json(line).model_dump()
                    if finished or (result is not None and event["type"] != "execution_finished"):
                        raise RuntimeError("Execution events arrived after the final result")

                    # A finish marker confirms a result, never substitutes for it.
                    if event["type"] == "execution_finished" and result is None:
                        raise RuntimeError("Execution finished without a result")

                    # Remember the terminal marker until the transport reaches EOF.
                    if event["type"] == "execution_finished":
                        finished = True

                    else:
                        if event["type"] == "result":
                            result = {**event, "ui": []}
                            result.pop("type")
                        emit(event)

            # EOF alone does not confirm that execution finished.
            if not finished or result is None:
                raise RuntimeError("Execution stream ended without a confirmed result")

        # Cancellation is just as ambiguous as a transport or validation failure.
        except BaseException as error:
            self.uncertain = True
            if not isinstance(error, Exception):
                raise

            # Tell callers not to retry code whose effects cannot be confirmed.
            raise RuntimeError(
                "Execution was not confirmed; code may already have run. Restart the execution container and web server before submitting more work."
            )
        return result

    def execute(self, code: str) -> ExecutionResult:
        """Submit code once and wait for its result."""
        try:
            response = httpx.post(f"{self.url}/", json={"code": code}, timeout=None, trust_env=False)
            response.raise_for_status()

        # Transport failure does not mean the server skipped execution.
        except httpx.HTTPError:
            raise RuntimeError("Execution request failed; code may already have run. Inspect the session or use /reset before retrying.")
        return response.json()

    def reset(self) -> None:
        """Clear the remote namespace after any active execution finishes."""
        response = httpx.post(f"{self.url}/reset", timeout=None, trust_env=False)
        response.raise_for_status()
        self.uncertain = False

    def libraries(self) -> list[dict[str, str]]:
        """List packages installed in the execution environment."""
        response = httpx.get(f"{self.url}/libraries", timeout=10, trust_env=False)
        response.raise_for_status()
        return response.json()
