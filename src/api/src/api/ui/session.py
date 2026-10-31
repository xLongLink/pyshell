import json
from pydantic import TypeAdapter
from api.ui.state import UIState
from api.ui.types import Action, Identifier
from api.ui.plot.figure import Figure

identifier_adapter = TypeAdapter(Identifier)
action_adapter = TypeAdapter(Action)


def _encoded_json_size(value: object) -> int:
    return len(json.dumps(value, allow_nan=False).encode())


class UI:
    """Own widget handles and publish A2UI messages from persistent Python."""

    def __init__(self, state: UIState | None = None) -> None:
        """Keep protocol state local to one execution session."""
        self.reset_output()
        self._widgets: dict[str, Figure] = {}
        self._sequence = 0
        self._closed = False
        self.state = state if state is not None else UIState()

    def plot(self, data: object = None, *, surface_id: str | None = None, **attributes: object) -> Figure:
        """Validate and publish a figure, accepting positional or keyword data."""
        widget = Figure.model_validate({"data": data, **attributes})

        # Automatic IDs must not replace explicitly named surfaces.
        if surface_id is None:
            while True:
                self._sequence += 1
                surface_id = f"surface_{self._sequence}"
                if surface_id not in self._widgets:
                    break

        # Validate the entire replacement before publishing or changing handles.
        identifier = identifier_adapter.validate_python(surface_id)
        if identifier not in self._widgets and len(self._widgets) >= 20:
            raise ValueError("At most 20 surfaces may be displayed; remove an existing surface first")
        messages = [{"version": "v0.9", "createSurface": {"surfaceId": identifier}}, self._content(identifier, widget)]

        # Bind the handle only after publication succeeds.
        self._publish(messages)
        widget._owner = self
        widget._surface_id = identifier
        self._widgets[identifier] = widget
        return widget

    def reset_output(self) -> None:
        """Start a new execution's publication budget without clearing widgets."""
        self._message_count = 0
        self._byte_count = 2
        self._node_count = 1

    def _content(self, surface_id: str, widget: Figure) -> dict[str, object]:
        component = widget.model_dump(mode="json")
        if _encoded_json_size(component) > 262144:
            raise ValueError("A surface must not exceed 256 KiB")
        return {"version": "v0.9", "updateComponents": {"surfaceId": surface_id, "components": [component]}}

    def _publish(self, messages: list[dict[str, object]]) -> None:
        if self._closed:
            raise RuntimeError("The UI session has been reset or closed")
        message_count = self._message_count + len(messages)
        byte_count = self._byte_count + _encoded_json_size(messages) - 2
        if self._message_count and messages:
            byte_count += 2
        if message_count > 100 or byte_count > 1048576:
            raise ValueError("UI output must not exceed 100 messages or 1 MiB per execution")

        # Bound node count before publishing messages.
        pending = list(messages)
        count = self._node_count
        while pending:
            item = pending.pop()
            count += 1
            if count > 20000:
                raise ValueError("UI output exceeds the display complexity limit")

            # Walk only containers; scalar values contribute a single node.
            if isinstance(item, dict):
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)

        self.state.publish(messages)
        self._message_count = message_count
        self._byte_count = byte_count
        self._node_count = count

    def _active(self, widget: Figure) -> None:
        if self._closed or self._widgets.get(widget.surface_id) is not widget:
            raise RuntimeError("This widget was removed, replaced, or reset")

    def update(self, widget: Figure, attributes: dict[str, object]) -> None:
        """Validate a complete candidate before publishing an in-place update."""
        self._active(widget)
        if {"id", "component"} & attributes.keys():
            raise ValueError("Widget identity and type cannot be changed by update()")
        candidate = Figure.model_validate({**widget.model_dump(mode="json"), **attributes})
        self._publish([self._content(widget.surface_id, candidate)])

        # Commit validated attributes only after the update has been published.
        for name in Figure.model_fields:
            object.__setattr__(widget, name, getattr(candidate, name))

    def action(self, widget: Figure, action: Action) -> None:
        """Preserve placement commands in the same order as content updates."""
        self._active(widget)
        action = action_adapter.validate_python(action)
        self._publish([{"type": "widget", "surface_id": widget.surface_id, "action": action}])

    def remove_widget(self, widget: Figure) -> None:
        """Reject stale handles before removing their surface."""
        self._active(widget)
        self.remove(widget.surface_id)

    def remove(self, surface_id: str) -> None:
        """Remove a named surface and invalidate its widget handle."""
        if surface_id not in self._widgets:
            raise ValueError("Unknown surface ID")
        self._publish([{"version": "v0.9", "deleteSurface": {"surfaceId": surface_id}}])
        del self._widgets[surface_id]

    def close(self) -> None:
        """Invalidate handles retained outside a reset Python namespace."""
        self._closed = True
        self._widgets.clear()
