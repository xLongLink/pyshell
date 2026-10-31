import httpx2
import asyncio
from typing import cast
from sample.src.events import Result, Finished

EXECUTION_UNCERTAIN = "Execution state is uncertain. Restart the execution container and web server before submitting more work."


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

        events: list[Result | Finished] = []
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

                    if len(events) == 2:
                        raise RuntimeError("Execution events arrived after the final result")
                    events.append((Result, Finished)[len(events)].model_validate_json(line))

            # EOF alone does not confirm that execution finished.
            if len(events) != 2:
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
        return cast(Result, events[0])
