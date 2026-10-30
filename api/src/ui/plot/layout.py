from pydantic import Field
from src.ui.model import UIModel
from src.ui.types import TextValue


class Title(UIModel):
    """Describe a figure or axis title."""

    text: TextValue = ""


class Axis(UIModel):
    """Configure an axis title."""

    title: Title = Field(default_factory=Title)


class Layout(UIModel):
    """Configure figure-wide presentation."""

    # Each figure owns its title and axis defaults.
    title: Title = Field(default_factory=Title)
    xaxis: Axis = Field(default_factory=Axis)
    showlegend: bool = True
