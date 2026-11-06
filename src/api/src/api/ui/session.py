import json
from typing import Literal
from pydantic import TypeAdapter
from api.ui.state import UIState
from api.ui.types import Action, Identifier
from api.ui.plot.map import MapView, MapLayer
from collections.abc import Sequence
from api.ui.plot.figure import Trace, Figure
from api.ui.plot.layout import Layout
from api.ui.plot.geojson import FeatureCollection

identifier_adapter = TypeAdapter(Identifier)
action_adapter = TypeAdapter(Action)


def _encoded_json_size(value: object) -> int:
    return len(json.dumps(value, allow_nan=False))


class UI:
    """Publish charts from persistent Python using a custom API, not Plotly.

    Use scatter(), area(), bar(), pie(), or choropleth() for a single series;
    use map() for interactive maps with building polygons or point markers.
    Read help(ui.bar), help(ui.choropleth), etc. for arguments and examples.
    Use plot() for multiple series or advanced trace models and layout settings.
    This is not Plotly; only the documented fields are supported.

    Example bar chart:
        chart = ui.bar(["A", "B"], [10, 20], title="Counts")

    Example country map with built-in boundaries (no GeoJSON download needed):
        chart = ui.choropleth(["FRA", "DEU"], [10, 20])
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

    def scatter(
        self,
        x: list[str | float | int],
        y: list[float | int | None],
        *,
        mode: Literal["lines", "markers", "lines+markers"] = "lines",
        name: str = "",
        title: str = "",
        surface_id: str | None = None,
    ) -> Figure:
        """Draw a line or scatter chart.

        Args:
            x: 1 to 200 coordinates, all numeric or all strings, in display order.
            y: Equally many finite numbers; None indicates missing data.
            mode: Draw lines, markers, or both.
            name: Series label in the legend.
            title: Chart title.
            surface_id: Replace this named surface, or create a new one if omitted.

        Returns:
            A chart handle with surface_id, update(), focus(), pin(), and remove().

        Example:
            chart = ui.scatter([2024, 2025], [9.0, 9.1], mode="lines+markers")

        """
        return self.plot(
            [{"type": "scatter", "x": x, "y": y, "mode": mode, "name": name}], layout={"title": {"text": title}}, surface_id=surface_id
        )

    def area(
        self,
        x: list[str | float | int],
        y: list[float | int | None],
        *,
        curve: Literal["linear", "monotone"] = "linear",
        name: str = "",
        title: str = "",
        surface_id: str | None = None,
    ) -> Figure:
        """Draw a series filled to zero.

        Args:
            x: 1 to 200 coordinates, all numeric or all strings, in display order.
            y: Equally many finite numbers; None indicates missing data.
            curve: Straight segments or a smooth monotone curve.
            name: Series label in the legend.
            title: Chart title.
            surface_id: Replace this named surface, or create a new one if omitted.

        Returns:
            A chart handle with surface_id, update(), focus(), pin(), and remove().

        Example:
            chart = ui.area([2024, 2025], [10, 20], title="Growth")

        """
        return self.plot(
            [{"type": "area", "x": x, "y": y, "curve": curve, "name": name}], layout={"title": {"text": title}}, surface_id=surface_id
        )

    def bar(
        self, x: list[str | float | int], y: list[float | int | None], *, name: str = "", title: str = "", surface_id: str | None = None
    ) -> Figure:
        """Compare values using bars.

        Args:
            x: 1 to 200 coordinates, all numeric or all strings, in display order.
            y: Equally many finite numbers; None indicates missing data.
            name: Series label in the legend.
            title: Chart title.
            surface_id: Replace this named surface, or create a new one if omitted.

        Returns:
            A chart handle with surface_id, update(), focus(), pin(), and remove().

        Example:
            chart = ui.bar(["A", "B"], [10, 20], title="Counts")

        """
        return self.plot([{"type": "bar", "x": x, "y": y, "name": name}], layout={"title": {"text": title}}, surface_id=surface_id)

    def pie(
        self, labels: list[str], values: list[float | int | None], *, name: str = "", title: str = "", surface_id: str | None = None
    ) -> Figure:
        """Show nonnegative shares as pie slices.

        Args:
            labels: 1 to 200 slice labels.
            values: Equally many finite nonnegative numbers; None omits a slice.
            name: Series name.
            title: Chart title.
            surface_id: Replace this named surface, or create a new one if omitted.

        Returns:
            A chart handle with surface_id, update(), focus(), pin(), and remove().

        Example:
            chart = ui.pie(["A", "B"], [10, 20], title="Shares")

        """
        return self.plot(
            [{"type": "pie", "labels": labels, "values": values, "name": name}], layout={"title": {"text": title}}, surface_id=surface_id
        )

    def choropleth(
        self,
        locations: list[str],
        values: list[float | int | None],
        *,
        geography: str = "world",
        geojson: FeatureCollection | dict[str, object] | None = None,
        name: str = "",
        title: str = "",
        surface_id: str | None = None,
    ) -> Figure:
        """Color geographic boundaries by value.

        Args:
            locations: 1 to 200 unique supported location codes.
            values: Equally many finite numbers; None indicates missing data.
            geography: "world" for countries, or an uppercase ISO alpha-3 country
                code for its administrative regions. Use geography_info() to
                discover codes. First use downloads and caches boundaries.
            geojson: Optional custom WGS84 Polygon/MultiPolygon FeatureCollection,
                overriding geography. Unique string feature IDs must match
                locations. Limited to 200 features and 4,000 coordinate positions.
            name: Series label describing the values and their units.
            title: Chart title.
            surface_id: Replace this named surface, or create a new one if omitted.

        Returns:
            A chart handle with surface_id, update(), focus(), pin(), and remove().

        Example:
            chart = ui.choropleth(["CHE", "FRA"], [9.09, 68.72],
                                 name="Population (millions)")

        Map zoom, custom colors, and label placement are not supported.

        """
        return self.plot(
            [{"type": "choropleth", "locations": locations, "values": values, "geography": geography, "geojson": geojson, "name": name}],
            layout={"title": {"text": title}},
            surface_id=surface_id,
        )

    def map(
        self,
        layers: Sequence[MapLayer | dict[str, object]] = (),
        *,
        view: MapView | dict[str, object] | None = None,
        title: str = "",
        surface_id: str | None = None,
    ) -> Figure:
        """Publish an interactive MapLibre map with geographic data layers.

        Args:
            layers: Up to eight layers with id, data (GeoJSON FeatureCollection),
                optional name, color (#RRGGBB), and popup_fields (property names).
                Features require unique string IDs, WGS84 Point/Polygon/
                MultiPolygon geometry, and at most 12 scalar properties. Pass only
                properties intended for browser display. Across all layers, at
                most 200 features and 4,000 positions may be sent. Empty data is
                valid; filter real workspace data in Python before publishing.
            view: Exactly one of {"country": "CHE"}, {"bounds": [[west, south],
                [east, north]]}, or {"center": [longitude, latitude], "zoom": 12}.
                Omit for a world view. Country bounds and OpenStreetMap basemap
                tiles require internet access. Place search and renovation data
                are not supplied by the map.
            title: Map title.
            surface_id: Replace this named surface, or create a new one if omitted.

        Returns:
            A figure handle. Use chart.set_view(country="CHE") for navigation,
            chart.update(data=[{"type": "map", "layers": [...]}]) to replace data
            without moving the camera, and focus(), pin(), or remove() as usual.

        Example:
            chart = ui.map(view={"country": "CHE"}, title="Switzerland")

        """
        return self.plot([{"type": "map", "layers": list(layers), "view": view}], layout={"title": {"text": title}}, surface_id=surface_id)

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
        not a Figure instance. Supported types: scatter, area, bar, pie, choropleth, map.
        Prefer bar(), scatter(), area(), pie(), or choropleth() for single series.
        This is not Plotly. Trace models provide model_json_schema() for advanced
        field inspection. Layout supports only title, xaxis, and showlegend:
        {"title": {"text": "Counts"}, "xaxis": {"title": {"text": "Year"}},
         "showlegend": True}.

        Statistical maps require exactly one choropleth trace with locations and values
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
