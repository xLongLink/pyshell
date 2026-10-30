from typing import Literal, Annotated
from pydantic import Field
from src.ui.types import Number
from src.ui.chart.common import Chart, Point


class Slice(Point):
    """Represent a nonnegative share, or a missing value."""

    value: Annotated[Number, Field(ge=0)] | None


class PieChart(Chart):
    """Compare nonnegative shares of a total."""

    component: Literal["PieChart"] = "PieChart"
    points: Annotated[list[Slice], Field(min_length=1, max_length=200)]
