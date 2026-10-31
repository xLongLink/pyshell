import asyncio
from threading import RLock
from dataclasses import dataclass


@dataclass
class Surface:
    """Retain a widget's current content and placement."""

    creation: dict[str, object]
    latest_update: dict[str, object] | None = None
    pinned: bool = False
    focused_at: int = 0


class UIState:
    """Own widget snapshots independently of executions and browser connections."""

    def __init__(self) -> None:
        """Initialize session state and bounded browser subscriptions."""
        self.lock = RLock()
        self.surfaces: dict[str, Surface] = {}
        self.sequence = 0
        self.listeners: dict[asyncio.Queue, asyncio.AbstractEventLoop] = {}
        self.pending: set[asyncio.Queue] = set()

    def snapshot(self) -> dict[str, object]:
        """Copy the current widgets and their placement."""
        with self.lock:
            return {
                "type": "snapshot",
                "events": [
                    {
                        "type": "a2ui",
                        "messages": [surface.creation] + ([surface.latest_update] if surface.latest_update is not None else []),
                    }
                    for surface in self.surfaces.values()
                ],
                "placements": {
                    key: {"pinned": surface.pinned} for key, surface in sorted(self.surfaces.items(), key=lambda item: item[1].focused_at)
                },
            }

    def subscribe(self) -> asyncio.Queue:
        """Subscribe on the browser event loop before taking its first snapshot."""
        with self.lock:
            queue = asyncio.Queue(maxsize=1)
            queue.put_nowait(self.snapshot())
            self.listeners[queue] = asyncio.get_running_loop()
            return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Detach a browser without affecting Python execution."""
        with self.lock:
            self.listeners.pop(queue, None)
            self.pending.discard(queue)

    def _deliver(self, queue: asyncio.Queue) -> None:
        with self.lock:
            self.pending.discard(queue)
            if queue not in self.listeners:
                return
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(self.snapshot())

    def _broadcast(self) -> None:
        for queue, loop in self.listeners.items():
            if queue not in self.pending:
                self.pending.add(queue)
                loop.call_soon_threadsafe(self._deliver, queue)

    def _focus(self, identifier: str) -> None:
        self.sequence += 1
        surface = self.surfaces[identifier]
        surface.pinned = False
        surface.focused_at = self.sequence
        centered = sorted((item for item in self.surfaces.values() if not item.pinned), key=lambda item: item.focused_at)
        for item in centered[:-4]:
            item.pinned = True

    def publish(self, messages: list[dict[str, object]]) -> None:
        """Apply a validated widget mutation from the execution thread."""
        with self.lock:
            for message in messages:
                if message.get("type") == "widget":
                    identifier = message["surface_id"]
                    if message["action"] == "focus":
                        self._focus(identifier)
                    else:
                        self.surfaces[identifier].pinned = True
                elif "createSurface" in message:
                    identifier = message["createSurface"]["surfaceId"]
                    self.surfaces[identifier] = Surface(message)
                    self._focus(identifier)
                elif "updateComponents" in message:
                    surface = self.surfaces[message["updateComponents"]["surfaceId"]]
                    surface.latest_update = message
                elif "deleteSurface" in message:
                    self.surfaces.pop(message["deleteSurface"]["surfaceId"], None)
            self._broadcast()

    def reset(self) -> None:
        """Clear widgets and resynchronize all connected browsers."""
        with self.lock:
            self.surfaces.clear()
            self.sequence = 0
            self._broadcast()

    def context(self) -> dict[str, object]:
        """Summarize charts without returning their datasets."""
        with self.lock:
            charts = []
            for identifier, surface in self.surfaces.items():
                if surface.latest_update is None:
                    continue
                charts.extend(
                    {
                        "surface_id": identifier,
                        "title": component.get("layout", {}).get("title", {}).get("text", "")[:300],
                        "trace_types": list(dict.fromkeys(trace["type"] for trace in component.get("data", []))),
                        "series": [trace.get("name", "")[:100] for trace in component.get("data", [])],
                        "pinned": surface.pinned,
                    }
                    for component in surface.latest_update["updateComponents"].get("components", [])
                )
            return {"charts": charts}
