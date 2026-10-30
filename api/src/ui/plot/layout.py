from pydantic import Field
from src.ui.types import TextValue
from src.ui.plot.common import PlotModel


class Title(PlotModel):
    """Describe a figure or axis title."""

    text: TextValue = ""


class Axis(PlotModel):
    """Configure an axis title."""

    title: Title = Field(default_factory=Title)


class Layout(PlotModel):
    """Configure figure-wide presentation."""

    # Each figure owns its title and axis defaults.
    title: Title = Field(default_factory=Title)
    xaxis: Axis = Field(default_factory=Axis)
    showlegend: bool = True
