from typing import Literal, ClassVar, Annotated
from pydantic import Field, model_validator
from src.ui.widget import Widget
from src.ui.plot.bar import Bar
from src.ui.plot.pie import Pie
from src.ui.plot.layout import Layout
from src.ui.plot.scatter import Scatter

Trace = Annotated[Scatter | Bar | Pie, Field(discriminator="type")]


class Figure(Widget):
    """Publish a bounded Plotly-style figure on one surface."""

    arguments: ClassVar[tuple[str, ...]] = ("data",)
    component: Literal["Figure"] = "Figure"
    data: Annotated[list[Trace], Field(min_length=1, max_length=20)]
    layout: Layout = Field(default_factory=Layout)

    @model_validator(mode="after")
    def traces(self) -> "Figure":
        """Require compatible traces on one coordinate system."""
        if any(isinstance(trace, Pie) for trace in self.data):
            if len(self.data) != 1:
                raise ValueError("A pie figure requires exactly one pie trace")
            count = len(self.data[0].values)

        # Cartesian traces share one numeric or categorical coordinate system.
        else:
            coordinates = [value for trace in self.data for value in trace.x]
            if len({isinstance(value, str) for value in coordinates}) != 1:
                raise ValueError("All x coordinates must be numeric or all categorical")
            count = len(coordinates)

        # The point budget applies to the entire figure, not each trace.
        if count > 200:
            raise ValueError("A figure accepts at most 200 total points")
        return self
