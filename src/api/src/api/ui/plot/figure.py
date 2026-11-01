from typing import TYPE_CHECKING, Literal, Annotated
from pydantic import Field, PrivateAttr, model_validator
from api.ui.model import UIModel
from api.ui.types import ImmutableList
from api.ui.plot.bar import Bar
from api.ui.plot.pie import Pie
from api.ui.plot.layout import Layout
from api.ui.plot.scatter import Scatter
from api.ui.plot.choropleth import Choropleth

if TYPE_CHECKING:
    from api.ui.session import UI

Trace = Annotated[Scatter | Bar | Pie | Choropleth, Field(discriminator="type")]


class Figure(UIModel):
    """Publish a bounded Plotly-style figure on one surface."""

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
        """Publish validated changes without replacing the figure's identity."""
        self._publisher().update(self, attributes)

    def focus(self) -> None:
        """Move this figure into the center, which holds up to four widgets."""
        self._publisher().action(self, "focus")

    def pin(self) -> None:
        """Move this figure into the sidebar."""
        self._publisher().action(self, "pin")

    def remove(self) -> None:
        """Remove this figure and invalidate its handle."""
        self._publisher().remove_widget(self)

    @model_validator(mode="after")
    def traces(self) -> "Figure":
        """Require compatible traces on one coordinate system."""
        standalone = next((trace for trace in self.data if isinstance(trace, (Pie, Choropleth))), None)
        if standalone is not None:
            if len(self.data) != 1:
                if any(isinstance(trace, Choropleth) for trace in self.data):
                    raise ValueError("A map figure requires exactly one choropleth trace")
                raise ValueError("A pie figure requires exactly one pie trace")
            count = len(standalone.values)

        # Cartesian traces share one numeric or categorical coordinate system.
        else:
            if len({isinstance(value, str) for trace in self.data for value in trace.x}) != 1:
                raise ValueError("All x coordinates must be numeric or all categorical")
            count = sum(len(trace.x) for trace in self.data)

        # The point budget applies to the entire figure, not each trace.
        if count > 200:
            raise ValueError("A figure accepts at most 200 total points")
        return self
