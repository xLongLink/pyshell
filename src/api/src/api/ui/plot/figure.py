from typing import TYPE_CHECKING, Literal, Annotated
from pydantic import Field, PrivateAttr, model_validator
from api.ui.model import UIModel
from api.ui.types import ImmutableList
from api.ui.plot.bar import Bar
from api.ui.plot.map import Map, MapView
from api.ui.plot.pie import Pie
from api.ui.plot.area import Area
from api.ui.plot.layout import Layout
from api.ui.plot.scatter import Scatter
from api.ui.plot.choropleth import Choropleth

if TYPE_CHECKING:
    from api.ui.session import UI

Trace = Annotated[Scatter | Area | Bar | Pie | Choropleth | Map, Field(discriminator="type")]


class Figure(UIModel):
    """Represent a chart in the custom UI schema, not the Plotly API.

    Publish trace dictionaries or models with ui.plot(data=[...], layout={...}).
    Cartesian charts accept at most 200 total points and 20 traces; pie and map
    charts require exactly one trace. Only fields in the models are supported.
    The returned handle exposes surface_id for references and update(), focus(),
    pin(), and remove(). Its id is always the internal component ID "root".
    """

    id: Literal["root"] = "root"
    component: Literal["Figure"] = "Figure"
    data: Annotated[ImmutableList[Trace], Field(min_length=1, max_length=20)]
    layout: Layout = Field(default_factory=Layout)

    # Publication metadata is private and excluded from the wire model.
    _owner: "UI | None" = PrivateAttr(default=None)
    _surface_id: str | None = PrivateAttr(default=None)

    @property
    def surface_id(self) -> str:
        """The stable ID assigned when the figure was published."""
        if self._surface_id is None:
            raise RuntimeError("Publish the widget before using its handle")
        return self._surface_id

    def _publisher(self) -> "UI":
        if self._owner is None:
            raise RuntimeError("Publish the widget before using its handle")
        return self._owner

    def update(self, **attributes: object) -> None:
        """Publish changes without replacing this chart's surface_id.

        Args:
            **attributes: data replaces the entire list of trace dictionaries or
                models; layout replaces the entire layout. Omitted top-level
                fields are preserved. Identity fields cannot be changed.
                Layout supports title={"text": "..."},
                xaxis={"title": {"text": "..."}}, and showlegend=True or False.

        Example:
            chart.update(data=[{"type": "bar", "x": ["A"], "y": [20]}],
                         layout={"title": {"text": "Updated"}})

        Invalid changes raise an error without modifying the published chart.

        """
        self._publisher().update(self, attributes)

    def focus(self) -> None:
        """Move this figure into the center, which holds up to four widgets."""
        self._publisher().action(self, "focus")

    def set_view(self, **attributes: object) -> None:
        """Navigate an interactive map without replacing its layers.

        Pass country="CHE", bounds=[[west, south], [east, north]], or
        center=[longitude, latitude] with optional zoom=12. Coordinates use WGS84.
        Each call navigates even when repeating a previous destination after a
        manual pan. Country navigation uses downloaded built-in world boundaries.
        """
        trace = self.data[0]
        if not isinstance(trace, Map):
            raise ValueError("set_view() requires an interactive map")
        revision = trace.view.revision + 1 if trace.view is not None else 1
        view = MapView.model_validate({**attributes, "revision": revision})
        self.update(data=[{"type": trace.type, "layers": list(trace.layers), "view": view}])

    def pin(self) -> None:
        """Move this figure into the sidebar."""
        self._publisher().action(self, "pin")

    def remove(self) -> None:
        """Remove this figure and invalidate its handle."""
        self._publisher().remove_widget(self)

    @model_validator(mode="after")
    def traces(self) -> "Figure":
        """Require compatible traces on one coordinate system."""
        if any(isinstance(trace, (Pie, Choropleth, Map)) for trace in self.data):
            if len(self.data) != 1:
                if any(isinstance(trace, Map) for trace in self.data):
                    raise ValueError("An interactive map requires exactly one map trace")
                if any(isinstance(trace, Choropleth) for trace in self.data):
                    raise ValueError("A map figure requires exactly one choropleth trace")
                raise ValueError("A pie figure requires exactly one pie trace")

        # Cartesian traces share one numeric or categorical coordinate system.
        else:
            if len({isinstance(value, str) for trace in self.data for value in trace.x}) != 1:
                raise ValueError("All x coordinates must be numeric or all categorical")
            count = sum(len(trace.x) for trace in self.data)

            # The point budget applies to the entire figure, not each trace.
            if count > 200:
                raise ValueError("A figure accepts at most 200 total points")
        return self
