import asyncio
from uuid import uuid4
from typing import Annotated
from pydantic import Field, TypeAdapter
from collections.abc import Callable
from sample.src.shell import EXECUTION_UNCERTAIN
from sample.src.agents.script import Agent

instruction_adapter = TypeAdapter(Annotated[str, Field(strict=True, max_length=8000)])


class Subscription:
    """Bound one browser's pending events without delaying shared work."""

    def __init__(self, workspace: "Workspace") -> None:
        """Register a connection and its initial readiness event."""
        self.workspace = workspace
        self.queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=256)
        self.closed = asyncio.Event()
        workspace.listeners.add(self)
        self.publish({"type": "snapshot"})

    def publish(self, event: dict[str, object]) -> None:
        """Detach an overloaded browser instead of losing queued events."""
        if self.closed.is_set():
            return
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.close()

    def close(self) -> None:
        """Wake the connection owner and stop receiving broadcasts."""
        self.workspace.listeners.discard(self)
        self.closed.set()


class Workspace:
    """Own shared execution, history, and application browser subscriptions."""

    def __init__(self, agent: Agent) -> None:
        """Keep workspace resources independent of browser connections."""
        self.id = str(uuid4())
        self.agent = agent
        self.listeners: set[Subscription] = set()
        self.task: asyncio.Task[None] | None = None

        self.closing = False
        self.results: set[Callable[[dict[str, object]], None]] = set()
        self.current_instruction: str | None = None
        self.phase = "idle"
        self.recent_tasks: list[dict[str, object]] = []

    @property
    def busy(self) -> bool:
        """Whether an accepted instruction is still running."""
        return self.current_instruction is not None

    async def context(self) -> dict[str, object]:
        """Snapshot work status and history, then fetch current chart metadata."""
        snapshot: dict[str, object] = {
            "status": {
                "busy": self.busy,
                "phase": self.phase,
                "instruction": self.current_instruction,
                "execution_uncertain": self.agent.shell.uncertain,
            },
            "recent_tasks": list(self.recent_tasks),
            "charts": [],
        }

        try:
            snapshot["charts"] = (await self.agent.shell.context())["charts"]
        except Exception:
            snapshot["charts_unavailable"] = True
        return snapshot

    def update_phase(self, phase: str) -> None:
        """Update progress for the active execution."""
        self.phase = phase

    def publish(self, event: dict[str, object]) -> None:
        """Broadcast browser messages without waiting on slow tabs."""
        for listener in tuple(self.listeners):
            listener.publish(event)

    def start(self, instruction: str) -> None:
        """Submit work once, outside the socket's lifetime."""
        instruction = instruction_adapter.validate_python(instruction)
        if self.closing:
            raise RuntimeError("The workspace is shutting down.")
        if self.busy:
            raise RuntimeError("The shared workspace is busy. Wait for the current operation to finish.")

        # Reject unsafe submissions before changing shared run state.
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Enter an instruction.")
        if self.agent.shell.uncertain:
            raise RuntimeError(EXECUTION_UNCERTAIN)
        self.current_instruction = instruction
        self.phase = "starting"

        self.publish({"type": "instruction"})
        self.task = asyncio.create_task(self._execute(instruction))

    async def _execute(self, instruction: str) -> None:
        """Record execution feedback for voice and publish shared status."""
        outcome = "completed"
        try:
            result = await self.agent.arun(instruction, self.update_phase)
        except Exception as error:
            outcome = "failed"
            self.publish({"type": "error", "text": str(error)})

            # A failed task may still have changed the persistent session.
            result = f"Task failed: {error}. Earlier changes may have applied; do not automatically resubmit."
        finally:
            self.phase = "idle"
            self.current_instruction = None
        record = {"instruction": instruction[:2000], "outcome": outcome, "result": result[-4000:]}
        self.recent_tasks.append(record)
        self.recent_tasks = self.recent_tasks[-8:]
        for report in self.results:
            report(record)

    async def close(self) -> None:
        """Stop submissions and finish accepted work."""
        self.closing = True
        if self.task is not None:
            await self.task
