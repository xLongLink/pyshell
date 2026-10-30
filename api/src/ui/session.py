import json
from pydantic import TypeAdapter
from src.ui.types import Action, Identifier
from src.ui.widget import Widget
from collections.abc import Callable
from a2ui.core.catalog import Catalog, ModelComponentApi
from src.ui.plot.figure import Figure
from a2ui.core.processing import MessageProcessor
from a2ui.core.schema.v0_9 import server_to_client

CATALOG = "urn:pyshell:catalog:v3"
CATALOG_MODEL = Catalog(CATALOG, "v0.9", [ModelComponentApi(Figure)])


class UI:
    """Own widget handles and publish A2UI messages from persistent Python."""

    def __init__(self) -> None:
        """Keep protocol state local to one execution session."""
        self.messages: list[dict[str, object]] = []
        self.on_publish: Callable[[dict[str, object]], None] | None = None
        self._widgets: dict[str, Widget] = {}
        self._sequence = 0
        self._closed = False
        self._processor = MessageProcessor([CATALOG_MODEL])

    def plot(self, *values: object, surface_id: str | None = None, **attributes: object) -> Figure:
        """Validate and publish a figure, accepting positional or keyword data."""
        if len(values) > 1:
            raise TypeError("Figure accepts at most 1 positional arguments")
        if values:
            if "data" in attributes:
                raise TypeError("Multiple values for 'data'")
            attributes["data"] = values[0]
        widget = Figure.model_validate(attributes)
        self.show(widget, surface_id=surface_id)
        return widget

    @property
    def surfaces(self) -> set[str]:
        """The currently published surface IDs."""
        return set(self._widgets)

    def _create(self, surface_id: str) -> dict[str, object]:
        body = server_to_client.CreateSurface(surfaceId=surface_id, catalogId=CATALOG)
        message = server_to_client.CreateSurfaceMessage(createSurface=body)
        return message.model_dump(mode="json", by_alias=True, exclude_none=True)

    def _content(self, surface_id: str, widget: Widget) -> dict[str, object]:
        body = server_to_client.UpdateComponents(surfaceId=surface_id, components=[widget.model_dump(mode="json")])
        message = server_to_client.UpdateComponentsMessage(updateComponents=body)
        return message.model_dump(mode="json", by_alias=True, exclude_none=True)

    def _delete(self, surface_id: str) -> dict[str, object]:
        body = server_to_client.DeleteSurface(surfaceId=surface_id)
        message = server_to_client.DeleteSurfaceMessage(deleteSurface=body)
        return message.model_dump(mode="json", by_alias=True, exclude_none=True)

    def _validate(self, surface_id: str, widget: Widget) -> None:
        if len(json.dumps(widget.model_dump(mode="json"), allow_nan=False).encode()) > 262144:
            raise ValueError("A surface must not exceed 256 KiB")
        processor = MessageProcessor([CATALOG_MODEL])
        processor.process_messages([self._create(surface_id), self._content(surface_id, widget)])

    def _publish(self, messages: list[dict[str, object]]) -> None:
        if self._closed:
            raise RuntimeError("The UI session has been reset or closed")
        combined = self.messages + messages
        if len(combined) > 100 or len(json.dumps(combined, allow_nan=False).encode()) > 1048576:
            raise ValueError("UI output must not exceed 100 messages or 1 MiB per execution")

        # Bound nesting and node count before handing messages to the processor.
        pending = [(combined, 0)]
        count = 0
        while pending:
            item, depth = pending.pop()
            count += 1
            if count > 20000 or depth > 24:
                raise ValueError("UI output exceeds the display complexity limit")

            # Walk only containers; scalar values contribute a single node.
            if isinstance(item, dict):
                pending.extend((value, depth + 1) for value in item.values())
            elif isinstance(item, list):
                pending.extend((value, depth + 1) for value in item)

        # Placement commands are workspace events, not A2UI protocol messages.
        protocol = [message for message in messages if message.get("type") != "widget"]
        if protocol:
            self._processor.process_messages(protocol)
        self.messages.extend(messages)

        # Keep content batches on their original side of placement commands.
        if self.on_publish is None:
            return
        batch = []

        # Flush content before forwarding the placement command.
        for message in messages:
            if message.get("type") == "widget":
                if batch:
                    self.on_publish({"type": "a2ui", "messages": batch})

                    # The placement command begins a new content batch.
                    batch = []
                self.on_publish(message)
            else:
                batch.append(message)

        # A trailing content batch has no placement boundary to flush it.
        if batch:
            self.on_publish({"type": "a2ui", "messages": batch})

    def show(self, component: Widget, *, surface_id: str | None = None) -> Widget:
        """Publish a single figure as the root of its own surface."""
        if component._owner is not None:
            raise RuntimeError("Use update() to change an already published widget")
        widget = component
        if widget.id != "root":
            raise ValueError("A figure requires the component ID 'root'")

        # Automatic IDs must not replace explicitly named surfaces.
        if surface_id is None:
            self._sequence += 1
            surface_id = f"surface_{self._sequence}"
            while surface_id in self._widgets:
                self._sequence += 1
                surface_id = f"surface_{self._sequence}"

        # Validate the entire replacement before publishing or changing handles.
        identifier = TypeAdapter(Identifier).validate_python(surface_id)
        if identifier not in self._widgets and len(self._widgets) >= 20:
            raise ValueError("At most 20 surfaces may be displayed; remove an existing surface first")
        self._validate(identifier, widget)
        messages = [self._create(identifier), self._content(identifier, widget)]
        if identifier in self._widgets:
            messages.insert(0, self._delete(identifier))

        # Bind the handle only after publication succeeds.
        self._publish(messages)
        widget._owner = self
        widget._surface_id = identifier
        self._widgets[identifier] = widget
        return widget

    def _active(self, widget: Widget) -> None:
        if self._closed or self._widgets.get(widget.surface_id) is not widget:
            raise RuntimeError("This widget was removed, replaced, or reset")

    def update(self, widget: Widget, attributes: dict[str, object]) -> None:
        """Validate a complete candidate before publishing an in-place update."""
        self._active(widget)
        if {"id", "component"} & attributes.keys():
            raise ValueError("Widget identity and type cannot be changed by update()")
        candidate = type(widget).model_validate({**widget.model_dump(mode="json"), **attributes})
        self._validate(widget.surface_id, candidate)
        self._publish([self._content(widget.surface_id, candidate)])

        # Commit validated attributes only after the update has been published.
        for name in type(widget).model_fields:
            object.__setattr__(widget, name, getattr(candidate, name))

    def action(self, widget: Widget, action: Action) -> None:
        """Preserve placement commands in the same order as content updates."""
        self._active(widget)
        action = TypeAdapter(Action).validate_python(action)
        self._publish([{"type": "widget", "surface_id": widget.surface_id, "action": action}])

    def remove_widget(self, widget: Widget) -> None:
        """Reject stale handles before removing their surface."""
        self._active(widget)
        self.remove(widget.surface_id)

    def remove(self, surface_id: str) -> None:
        """Remove a named surface and invalidate its widget handle."""
        if surface_id not in self._widgets:
            raise ValueError("Unknown surface ID")
        self._publish([self._delete(surface_id)])
        del self._widgets[surface_id]

    def close(self) -> None:
        """Invalidate handles retained outside a reset Python namespace."""
        self._closed = True
        self._widgets.clear()
