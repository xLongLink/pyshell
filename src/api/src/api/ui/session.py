import json
from pydantic import TypeAdapter
from api.ui.state import UIState
from api.ui.types import Action, Identifier
from collections.abc import Sequence
from api.ui.plot.figure import Trace, Figure
from api.ui.plot.layout import Layout

identifier_adapter = TypeAdapter(Identifier)
action_adapter = TypeAdapter(Action)


def _encoded_json_size(value: object) -> int:
    return len(json.dumps(value, allow_nan=False))


class UI:
    """Publish charts from persistent Python using a custom API, not Plotly.

    Trace models available in the workspace: Scatter, Area, Bar, Pie, Choropleth.
    Layout configures presentation; Figure is the returned chart handle.
    Use help(Choropleth) or help(Layout) for usage and types. For exact fields
    and constraints, print(Choropleth.model_json_schema()). Do not assume
    Plotly fields are supported.

    Example bar chart:
        chart = ui.plot(data=[{"type": "bar", "x": ["A", "B"], "y": [10, 20]}],
                        layout={"title": {"text": "Counts"}})

    Example country map with built-in boundaries (no GeoJSON download needed):
        chart = ui.plot(data=[{"type": "choropleth", "geography": "world",
                               "locations": ["FRA", "DEU"], "values": [10, 20]}])
        print(chart.surface_id)

    Use geography_info("world") to discover supported country codes and names;
    geography_info("FRA") discovers administrative regions within France.
    Read that output before choosing locations. Map coverage varies by provider.
    Reuse chart.update(data=[...], layout={...}) to update a chart; chart.focus(),
    chart.pin(), and chart.remove() control placement and removal.
    """

    def __init__(self, state: UIState | None = None) -> None:
        """Keep protocol state local to one execution session."""
        self.reset_output()
        self._widgets: dict[str, Figure] = {}
        self._sequence = 0
        self._closed = False
        self.state = state if state is not None else UIState()

    def plot(
        self,
        data: Sequence[Trace | dict[str, object]] | None = None,
        *,
        layout: Layout | dict[str, object] | None = None,
        surface_id: str | None = None,
        **attributes: object,
    ) -> Figure:
        """Publish trace data with ui.plot(data=[...], layout={...}) and return a Figure handle.

        The data argument accepts a list of trace dictionaries or trace models,
        not a Figure instance. Supported types: scatter, area, bar, pie, choropleth.
        This is not Plotly: inspect help(Choropleth), help(Bar), etc. before using
        unfamiliar fields; model_json_schema() lists exact fields and constraints.
        Layout supports only title, xaxis, and showlegend; see help(Layout).

        Maps require exactly one choropleth trace with locations and values
        (not z). Prefer built-in geography="world" for countries or a country
        code such as "FRA" for its administrative regions. Use geography_info()
        to discover supported locations. Missing values are None.

        Pass surface_id to replace a named surface; otherwise a new surface is created.
        The returned chart.surface_id identifies the published surface;
        chart.id is the internal component ID, not a chart reference.
        """
        if layout is not None:
            attributes["layout"] = layout
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
        if self._widgets.get(widget.surface_id) is not widget:
            raise RuntimeError("This widget was removed, replaced, or reset")

    def update(self, widget: Figure, attributes: dict[str, object]) -> None:
        """Validate a complete candidate before publishing an in-place update."""
        self._active(widget)
        if {"id", "component"} & attributes.keys():
            raise ValueError("Widget identity and type cannot be changed by update()")
        candidate = Figure.model_validate({**widget.model_dump(mode="json", exclude=set(attributes)), **attributes})
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
