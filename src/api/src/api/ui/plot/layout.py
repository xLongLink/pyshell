from pydantic import Field
from api.ui.model import UIModel
from api.ui.types import TextValue


class Title(UIModel):
    """Describe a figure or axis title."""

    text: TextValue = ""


class Axis(UIModel):
    """Configure an axis title."""

    title: Title = Field(default_factory=Title)


class Layout(UIModel):
    """Configure title, xaxis title, and legend visibility; no other fields.

    Example: ui.plot(data=[...], layout={"title": {"text": "Population"},
                                       "showlegend": False})
    For Cartesian charts, xaxis={"title": {"text": "Year"}} sets the x-axis title.
    Plotly geo, height, margin, annotations, and other layout fields are unsupported.
    Print Layout.model_json_schema() for exact field types and constraints.
    """

    # Each figure owns its title and axis defaults.
    title: Title = Field(default_factory=Title)
    xaxis: Axis = Field(default_factory=Axis)
    showlegend: bool = True
