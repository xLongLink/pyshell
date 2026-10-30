import json
import asyncio
from uuid import uuid4
from typing import Literal
from pydantic import Field, BaseModel, ConfigDict
from web.src.events import WidgetCommand
from collections.abc import Callable
from web.src.agents.script import Agent


class Command(BaseModel):
    """Validate text instructions before allocating work."""

    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["run"]
    instruction: str = Field(default="", max_length=8000)


class Workspace:
    """Own shared execution, history, widgets, and browser subscriptions."""

    def __init__(self, agent: Agent) -> None:
        """Keep workspace resources independent of browser connections."""
        self.id = str(uuid4())
        self.agent = agent
        self.events: list[dict[str, object]] = []
        self.listeners: set[asyncio.Queue[dict[str, object]]] = set()
        self.task: asyncio.Task[str] | None = None
        self.placements: dict[str, dict[str, bool]] = {}
        self.surfaces: dict[str, list[dict[str, object]]] = {}
        self.run_id: str | None = None

        # Track bounded replay separately from the current widget state.
        self.sequence = 0
        self.history_bytes = 0
        self.compacted = False
        self.closing = False
        self.busy = False
        self.results: set[Callable[[str], None]] = set()
        self.current_instruction: str | None = None
        self.phase = "idle"
        self.recent_tasks: list[dict[str, object]] = []

    def status(self) -> dict[str, object]:
        """Describe current work without submitting Python or changing state."""
        return {
            "busy": self.busy,
            "phase": self.phase,
            "run_id": self.run_id,
            "instruction": self.current_instruction,
            "execution_uncertain": self.agent.shell.uncertain,
            "latest_outcome": dict(self.recent_tasks[-1]) if self.recent_tasks else None,
        }

    def context(self) -> dict[str, object]:
        """Summarize recent work and current charts without copying their datasets."""
        charts = []
        for identifier, messages in self.surfaces.items():
            for message in messages:
                for component in message.get("updateComponents", {}).get("components", []):
                    if component.get("component") != "Figure":
                        continue
                    charts.append(
                        {
                            "surface_id": identifier,
                            "title": component.get("layout", {}).get("title", {}).get("text", "")[:300],
                            "trace_types": list(dict.fromkeys(trace["type"] for trace in component.get("data", []))),
                            "series": [trace.get("name", "")[:100] for trace in component.get("data", [])],
                            "pinned": self.placements[identifier]["pinned"],
                        }
                    )
        return {"recent_tasks": [dict(task) for task in self.recent_tasks], "charts": charts}

    def snapshot(self) -> dict[str, object]:
        """Copy current state without replaying any submitted operation."""
        events = list(self.events)
        if self.compacted:
            events = [event for event in events if event["type"] != "a2ui"]
            for messages in self.surfaces.values():
                events.append({"type": "a2ui", "messages": messages})

        # Reconnecting tabs receive state without resubmitting work.
        return {
            "type": "snapshot",
            "events": events,
            "busy": self.busy,
            "model": self.agent.model,
            "placements": self._copy_placements(),
            "sequence": self.sequence,
            "run_id": self.run_id,
        }

    def publish(self, event: dict[str, object]) -> None:
        """Apply ordered mutations and broadcast without waiting on slow tabs."""
        if self.busy and event["type"] in {"code_start", "execution_start"}:
            self.phase = "generating_code" if event["type"] == "code_start" else "executing_python"
        if event["type"] == "widget":
            self.place(WidgetCommand.model_validate(event))
            return
        placement_changed = False

        # Maintain current surfaces even after their historical events are evicted.
        if event["type"] == "a2ui":
            for message in event["messages"]:
                if "createSurface" in message:
                    identifier = message["createSurface"]["surfaceId"]

                    # New surfaces start centered and may evict an older widget.
                    self.surfaces[identifier] = [message]
                    self.placements[identifier] = {"pinned": False}
                    self._focus(identifier)
                    placement_changed = True

                # Component updates replace content but retain surface creation.
                elif "updateComponents" in message:
                    identifier = message["updateComponents"]["surfaceId"]
                    self.surfaces[identifier] = [self.surfaces[identifier][0], message]

                # Removed surfaces must disappear from reconnect snapshots too.
                elif "deleteSurface" in message:
                    identifier = message["deleteSurface"]["surfaceId"]
                    self.surfaces.pop(identifier, None)
                    self.placements.pop(identifier, None)
                    placement_changed = True

        # Stamp events before retaining or broadcasting them.
        self.sequence += 1
        event = {**event, "sequence": self.sequence, "run_id": self.run_id}
        if event["type"] not in {"state", "placement"}:
            self.events.append(event)
            self.history_bytes += len(json.dumps(event).encode())

            # Bound replay by both event count and serialized size.
            while self.events and (len(self.events) > 2000 or self.history_bytes > 2_097_152):
                self.history_bytes -= len(json.dumps(self.events.pop(0)).encode())
                self.compacted = True

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
        if placement_changed:
            self._placement_event()

    def _copy_placements(self) -> dict[str, dict[str, bool]]:
        return {key: dict(value) for key, value in self.placements.items()}

    def _placement_event(self) -> None:
        self.publish({"type": "placement", "widgets": self._copy_placements()})

    def _focus(self, surface_id: str) -> None:
        placement = self.placements.pop(surface_id)
        placement["pinned"] = False
        self.placements[surface_id] = placement
        centered = [value for value in self.placements.values() if not value["pinned"]]
        for oldest in centered[:-4]:
            oldest["pinned"] = True

    def place(self, command: WidgetCommand) -> None:
        """Keep Python widget placement authoritative across all tabs."""
        placement = self.placements.get(command.surface_id)
        if placement is None:
            raise ValueError("Unknown or removed widget")

        # Focusing can evict the oldest centered widget to the sidebar.
        if command.action in {"focus", "unpin"}:
            self._focus(command.surface_id)
        else:
            placement["pinned"] = True
        self._placement_event()

    def start(self, command: Command) -> asyncio.Task[str]:
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
        self.run_id = str(uuid4())
        self.busy = True
        self.current_instruction = instruction
        self.phase = "starting"

        # Publish acceptance before scheduling work independently of its caller.
        self.publish({"type": "state", "busy": True})
        self.publish({"type": "instruction", "text": instruction})
        self.task = asyncio.create_task(self.execute(command))
        return self.task

    async def execute(self, command: Command) -> str:
        """Return confirmed execution feedback to voice and publish shared status."""
        outcome = "completed"
        instruction = command.instruction.strip()
        before = dict(self.surfaces)
        placements = self._copy_placements()
        try:
            result = await self.agent.arun(instruction, self.publish)
            self.publish({"type": "done"})
        except Exception as error:
            outcome = "failed"
            self.publish({"type": "error", "text": str(error)})

            # A failed task may still have changed the persistent session.
            result = f"Task failed: {error}. Earlier changes may have applied; do not automatically resubmit."
        finally:
            self.busy = False
            self.phase = "idle"
            self.current_instruction = None
            self.publish({"type": "state", "busy": False})
        record = {
            "run_id": self.run_id,
            "instruction": instruction[:2000],
            "outcome": outcome,
            "result": result[-4000:],
            "changed_charts": [
                key for key, value in self.surfaces.items() if before.get(key) != value or placements.get(key) != self.placements[key]
            ],
            "removed_charts": [key for key in before if key not in self.surfaces],
        }
        self.recent_tasks.append(record)
        self.recent_tasks = self.recent_tasks[-8:]
        for report in self.results:
            report(json.dumps(record))
        return result

    async def close(self) -> None:
        """Finish accepted work before releasing the execution transport."""
        self.closing = True
        if self.task is not None:
            await self.task
        await self.agent.shell.close()
