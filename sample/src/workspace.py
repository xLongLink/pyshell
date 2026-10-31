import json
import asyncio
from uuid import uuid4
from typing import Literal
from pydantic import Field, BaseModel, ConfigDict
from collections.abc import Callable
from sample.src.agents.script import Agent


class Command(BaseModel):
    """Validate text instructions before allocating work."""

    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["run"]
    instruction: str = Field(default="", max_length=8000)


class Workspace:
    """Own shared execution, history, and application browser subscriptions."""

    def __init__(self, agent: Agent) -> None:
        """Keep workspace resources independent of browser connections."""
        self.id = str(uuid4())
        self.agent = agent
        self.listeners: set[asyncio.Queue[dict[str, object]]] = set()
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
            "latest_outcome": dict(self.recent_tasks[-1]) if self.recent_tasks else None,
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
        """Copy current state without replaying any submitted operation."""
        return {"type": "snapshot"}

    def publish(self, event: dict[str, object]) -> None:
        """Apply ordered mutations and broadcast without waiting on slow tabs."""
        if event["type"] in {"code_start", "execution_start"}:
            if self.busy:
                self.phase = "generating_code" if event["type"] == "code_start" else "executing_python"
            return
        if event["type"] not in {"instruction", "error"}:
            return

        # Slow tabs resynchronize from state rather than blocking execution.
        for listener in self.listeners:
            if listener.full():
                controls = []

                # Preserve voice signaling while discarding stale replay events.
                while not listener.empty():
                    queued = listener.get_nowait()

                    # Voice answers cannot be reconstructed from workspace state.
                    if queued["type"] == "voice":
                        controls.append(queued)

                # Reserve queue capacity for the replacement snapshot first.
                listener.put_nowait(self.snapshot())

                # A snapshot can consume the subscriber's entire queue.
                for control in controls[-max(1, listener.maxsize - 1) :]:
                    if listener.full():
                        break

                    # Restore voice controls only while capacity remains.
                    listener.put_nowait(control)

            # Healthy subscribers receive each ordered event directly.
            else:
                listener.put_nowait(event)

    def start(self, command: Command) -> asyncio.Task[None]:
        """Submit text or voice work once, outside the socket's lifetime."""
        if self.closing:
            raise RuntimeError("The workspace is shutting down.")
        if self.busy:
            raise RuntimeError("The shared workspace is busy. Wait for the current operation to finish.")

        # Reject unsafe submissions before changing shared run state.
        instruction = command.instruction.strip()
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
        for report in self.results:
            report(json.dumps(record))

    async def close(self) -> None:
        """Stop submissions and finish accepted work."""
        self.closing = True
        if self.task is not None:
            await self.task
