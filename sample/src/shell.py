import httpx2
import asyncio
from sample.src.events import Result, execution_event


class RemoteShell:
    """Access the container's persistent Python session."""

    def __init__(self, client: httpx2.AsyncClient) -> None:
        """Use the application-owned execution transport."""
        self.client = client
        self.uncertain = False

    async def inventory(self) -> list[dict[str, str]]:
        """Read installed packages through the asynchronous transport."""
        response = await self.client.get("/libraries")
        response.raise_for_status()
        return response.json()

    async def context(self) -> dict[str, object]:
        """Read widget metadata without executing Python or retaining widget state."""
        response = await self.client.get("/ui/context", timeout=5)
        response.raise_for_status()
        return response.json()

    async def stream(self, code: str) -> Result:
        """Submit once and consume ordered output until completion is confirmed."""
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
                    event = execution_event.validate_json(line)
                    if finished or (result is not None and event.type != "execution_finished"):
                        raise RuntimeError("Execution events arrived after the final result")

                    # Remember the terminal marker until the transport reaches EOF.
                    if event.type == "execution_finished":
                        if result is None:
                            raise RuntimeError("Execution finished without a result")
                        finished = True

                    elif event.type == "result":
                        result = event
                    # Widgets are delivered directly to the container-hosted UI.

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
