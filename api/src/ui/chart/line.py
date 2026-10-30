from typing import Literal
from src.ui.chart.common import Chart


class LineChart(Chart):
    """Show an ordered series while preserving gaps for missing values."""

    component: Literal["LineChart"] = "LineChart"
