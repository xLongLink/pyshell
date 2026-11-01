from typing import Literal
from api.ui.plot.common import CartesianTrace


class Scatter(CartesianTrace):
    """Draw lines, markers, or both for an ordered coordinate series."""

    type: Literal["scatter"] = "scatter"
    mode: Literal["lines", "markers", "lines+markers"] = "lines"
