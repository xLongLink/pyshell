import json
from typing import Generic, TypeVar
from pydantic import TypeAdapter
from src.ui.row import Row
from src.ui.card import Card
from src.ui.text import Text
from src.ui.table import Table
from src.ui.types import Action, Identifier
from src.ui.column import Column
from src.ui.common import Layout
from src.ui.metric import Metric
from src.ui.widget import Widget
from src.ui.chart.bar import BarChart
from src.ui.chart.pie import PieChart
from a2ui.core.catalog import Catalog, ModelComponentApi
from src.ui.chart.line import LineChart
from a2ui.core.processing import MessageProcessor, MessageProcessorOptions
from a2ui.core.validation import ValidationConfig
from a2ui.core.schema.v0_9 import server_to_client

CATALOG = "urn:pyshell:catalog:v1"
MODELS = [Text, Metric, Table, Column, Row, Card, BarChart, LineChart, PieChart]
CATALOG_MODEL = Catalog(CATALOG, "v0.9", [ModelComponentApi(model) for model in MODELS])
OPTIONS = MessageProcessorOptions(validation_config=ValidationConfig(max_depth=15))
T = TypeVar("T", bound=Widget)


class Factory(Generic[T]):
    """Create and publish a widget using its own model's attributes."""

    def __init__(self, model: type[T], owner: "UI") -> None:
        """Bind a component model to a Python UI session."""
        self.model = model
        self.owner = owner

    def __call__(self, *values: object, surface_id: str | None = None, **attributes: object) -> T:
        """Validate constructor arguments without repeating component schemas."""
        if len(values) > len(self.model.arguments):
            raise TypeError(f"{self.model.__name__} accepts at most {len(self.model.arguments)} positional arguments")
        for name, value in zip(self.model.arguments, values):
            if name in attributes:
                raise TypeError(f"Multiple values for '{name}'")
            attributes[name] = value
        widget = self.model.model_validate(attributes)
        self.owner.show(widget, surface_id=surface_id)
        return widget


class Charts:
    """Expose chart-specific constructors without a kind argument."""

    def __init__(self, owner: "UI") -> None:
        """Share the widget publishing path across chart models."""
        self.bar = Factory(BarChart, owner)
        self.line = Factory(LineChart, owner)
        self.pie = Factory(PieChart, owner)


class UI:
    """Own widget handles and publish A2UI messages from persistent Python."""

    def __init__(self) -> None:
        """Keep protocol state local to one execution session."""
        self.messages: list[dict[str, object]] = []
        self._widgets: dict[str, list[Widget]] = {}
        self._sequence = 0
        self._closed = False
        self._processor = MessageProcessor([CATALOG_MODEL], options=OPTIONS)
        self.text = Factory(Text, self)
        self.metric = Factory(Metric, self)
        self.table = Factory(Table, self)
        self.chart = Charts(self)

    @property
    def surfaces(self) -> set[str]:
        """The currently published surface IDs."""
        return set(self._widgets)

    def _create(self, surface_id: str) -> dict[str, object]:
        body = server_to_client.CreateSurface(surfaceId=surface_id, catalogId=CATALOG)
        message = server_to_client.CreateSurfaceMessage(createSurface=body)
        return message.model_dump(mode="json", by_alias=True, exclude_none=True)

    def _content(self, surface_id: str, widgets: list[Widget]) -> dict[str, object]:
        body = server_to_client.UpdateComponents(surfaceId=surface_id, components=[widget.model_dump(mode="json") for widget in widgets])
        message = server_to_client.UpdateComponentsMessage(updateComponents=body)
        return message.model_dump(mode="json", by_alias=True, exclude_none=True)

    def _delete(self, surface_id: str) -> dict[str, object]:
        body = server_to_client.DeleteSurface(surfaceId=surface_id)
        message = server_to_client.DeleteSurfaceMessage(deleteSurface=body)
        return message.model_dump(mode="json", by_alias=True, exclude_none=True)

    def _validate(self, surface_id: str, widgets: list[Widget]) -> None:
        if len(json.dumps([widget.model_dump(mode="json") for widget in widgets], allow_nan=False).encode()) > 262144:
            raise ValueError("A surface must not exceed 256 KiB")
        references = [str(child) for widget in widgets if isinstance(widget, Layout) for child in widget.children]
        if len(set(references)) != len(references):
            raise ValueError("A component cannot have multiple parents")
        processor = MessageProcessor([CATALOG_MODEL], options=OPTIONS)
        processor.process_messages([self._create(surface_id), self._content(surface_id, widgets)])

    def _publish(self, messages: list[dict[str, object]]) -> None:
        if self._closed:
            raise RuntimeError("The UI session has been reset or closed")
        combined = self.messages + messages
        if len(combined) > 100 or len(json.dumps(combined, allow_nan=False).encode()) > 1048576:
            raise ValueError("UI output must not exceed 100 messages or 1 MiB per execution")
        pending = [(combined, 0)]
        count = 0
        while pending:
            item, depth = pending.pop()
            count += 1
            if count > 20000 or depth > 24:
                raise ValueError("UI output exceeds the display complexity limit")
            if isinstance(item, dict):
                pending.extend((value, depth + 1) for value in item.values())
            elif isinstance(item, list):
                pending.extend((value, depth + 1) for value in item)
        protocol = [message for message in messages if message.get("type") != "widget"]
        if protocol:
            self._processor.process_messages(protocol)
        self.messages.extend(messages)

    def show(self, component: Widget | list[dict[str, object]], *, surface_id: str | None = None) -> Widget:
        """Publish a typed widget or a complete flat component tree."""
        if isinstance(component, Widget):
            if component._owner is not None:
                raise RuntimeError("Use update() to change an already published widget")
            widgets = [component]
        else:
            if not isinstance(component, list) or not 1 <= len(component) <= 100:
                raise ValueError("A surface requires between 1 and 100 components")
            if len(json.dumps(component, allow_nan=False).encode()) > 262144:
                raise ValueError("A surface must not exceed 256 KiB")
            widgets = []
            for attributes in component:
                if not isinstance(attributes, dict) or attributes.get("component") not in CATALOG_MODEL.components:
                    raise ValueError("Unknown UI component")
                model = CATALOG_MODEL.components[attributes["component"]]
                widgets.append(model.type_adapter.validate_python(attributes))
        if len({widget.id for widget in widgets}) != len(widgets):
            raise ValueError("Components require unique IDs")
        root = next((widget for widget in widgets if widget.id == "root"), None)
        if root is None:
            raise ValueError("A surface requires a 'root' component")
        if surface_id is None:
            self._sequence += 1
            surface_id = f"surface_{self._sequence}"
            while surface_id in self._widgets:
                self._sequence += 1
                surface_id = f"surface_{self._sequence}"
        identifier = TypeAdapter(Identifier).validate_python(surface_id)
        if identifier not in self._widgets and len(self._widgets) >= 20:
            raise ValueError("At most 20 surfaces may be displayed; remove an existing surface first")
        self._validate(identifier, widgets)
        messages = [self._create(identifier), self._content(identifier, widgets)]
        if identifier in self._widgets:
            messages.insert(0, self._delete(identifier))
        self._publish(messages)
        root._owner = self
        root._surface_id = identifier
        self._widgets[identifier] = widgets
        return root

    def _active(self, widget: Widget) -> list[Widget]:
        widgets = self._widgets.get(widget.surface_id, [])
        if self._closed or not any(node is widget for node in widgets):
            raise RuntimeError("This widget was removed, replaced, or reset")
        return widgets

    def update(self, widget: Widget, attributes: dict[str, object]) -> None:
        """Validate a complete candidate before publishing an in-place update."""
        widgets = self._active(widget)
        if {"id", "component"} & attributes.keys():
            raise ValueError("Widget identity and type cannot be changed by update()")
        candidate = type(widget).model_validate({**widget.model_dump(mode="json"), **attributes})
        updated = [candidate if node is widget else node for node in widgets]
        self._validate(widget.surface_id, updated)
        self._publish([self._content(widget.surface_id, updated)])
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
