import httpx2
import asyncio
from typing import Literal
from pydantic import BaseModel, ConfigDict

EXECUTION_UNCERTAIN = "Execution state is uncertain. Restart the execution container and web server before submitting more work."


class Result(BaseModel):
    """Confirm a completed Python execution."""

    model_config = ConfigDict(extra="forbid", strict=True)

    # A result includes failures as data, alongside captured output.
    type: Literal["result"]
    stdout: str
    stderr: str
    exception: str | None
    done: bool


class _Finished(BaseModel):
    """Mark a clean end to the execution stream."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["execution_finished"]


class RemoteShell:
    """Access the container's persistent Python session."""

    def __init__(self, client: httpx2.AsyncClient) -> None:
        """Use the application-owned execution transport."""
        self.client = client
        self.uncertain = False
        self._inventory: list[dict[str, str]] | None = None

    async def inventory(self) -> list[dict[str, str]]:
        """Cache successful package inventory reads for the application lifetime."""
        if self._inventory is None:
            response = await self.client.get("/libraries")
            response.raise_for_status()
            self._inventory = response.json()
        return self._inventory

    async def context(self) -> dict[str, object]:
        """Read widget metadata without executing Python or retaining widget state."""
        response = await self.client.get("/ui/context", timeout=5)
        response.raise_for_status()
        return response.json()

    async def stream(self, code: str) -> Result:
        """Submit once and consume ordered output until completion is confirmed."""
        if self.uncertain:
            raise RuntimeError(EXECUTION_UNCERTAIN)

        result: Result | None = None
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

                    if finished:
                        raise RuntimeError("Execution events arrived after the final result")
                    if result is None:
                        result = Result.model_validate_json(line)
                    else:
                        _Finished.model_validate_json(line)
                        finished = True

            # EOF alone does not confirm that execution finished.
            if result is None or not finished:
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
