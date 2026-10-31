import json
import asyncio
from uuid import uuid4
from typing import Annotated
from pydantic import Field, TypeAdapter
from collections.abc import Callable
from sample.src.agents.script import Agent

Instruction = Annotated[str, Field(strict=True, max_length=8000)]
instruction_adapter = TypeAdapter(Instruction)


class Subscription:
    """Bound one browser's pending events without delaying shared work."""

    def __init__(self, workspace: "Workspace", *, maxsize: int = 256) -> None:
        """Register a connection and its initial readiness event."""
        self.workspace = workspace
        self.queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=maxsize)
        self.closed = asyncio.Event()
        workspace.listeners.add(self)
        self.publish(workspace.snapshot())

    def publish(self, event: dict[str, object]) -> None:
        """Detach an overloaded browser instead of losing queued events."""
        if self.closed.is_set():
            return
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.close()

    async def emit(self, event: dict[str, object]) -> None:
        """Use the same nonblocking policy for voice signaling."""
        self.publish(event)

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
        self.results: set[Callable[[str], None]] = set()
        self.current_instruction: str | None = None
        self.phase = "idle"
        self.recent_tasks: list[dict[str, object]] = []

    @property
    def busy(self) -> bool:
        """Whether an accepted instruction is still running."""
        return self.current_instruction is not None

    def status(self) -> dict[str, object]:
        """Describe current work without submitting Python or changing state."""
        return {
            "busy": self.busy,
            "phase": self.phase,
            "instruction": self.current_instruction,
            "execution_uncertain": self.agent.shell.uncertain,
        }

    async def context(self) -> dict[str, object]:
        """Summarize recent work and current charts without copying their datasets."""
        recent_tasks = [dict(task) for task in self.recent_tasks]
        try:
            context = await self.agent.shell.context()
            charts = context["charts"]
        except Exception:
            return {"recent_tasks": recent_tasks, "charts": [], "charts_unavailable": True}
        return {"recent_tasks": recent_tasks, "charts": charts}

    def snapshot(self) -> dict[str, object]:
        """Signal readiness without implying state replay."""
        return {"type": "snapshot"}

    def publish(self, event: dict[str, object]) -> None:
        """Apply ordered mutations and broadcast without waiting on slow tabs."""
        if event["type"] in {"code_start", "execution_start"}:
            if self.busy:
                self.phase = "generating_code" if event["type"] == "code_start" else "executing_python"
            return
        if event["type"] not in {"instruction", "error"}:
            return

        for listener in tuple(self.listeners):
            listener.publish(event)

    def start(self, instruction: str) -> asyncio.Task[None]:
        """Submit text or voice work once, outside the socket's lifetime."""
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
            raise RuntimeError("Execution state is uncertain. Restart the execution container and web server before submitting more work.")
        self.current_instruction = instruction
        self.phase = "starting"

        self.publish({"type": "instruction", "text": instruction})
        self.task = asyncio.create_task(self._execute(instruction))
        return self.task

    async def _execute(self, instruction: str) -> None:
        """Record execution feedback for voice and publish shared status."""
        outcome = "completed"
        try:
            result = await self.agent.arun(instruction, self.publish)
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
        if self.results:
            message = json.dumps(record)
            for report in self.results:
                report(message)

    async def close(self) -> None:
        """Stop submissions and finish accepted work."""
        self.closing = True
        if self.task is not None:
            await self.task
