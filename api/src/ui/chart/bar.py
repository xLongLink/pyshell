from typing import Literal
from src.ui.chart.common import Chart


class BarChart(Chart):
    """Compare labeled values using bars."""

    component: Literal["BarChart"] = "BarChart"
