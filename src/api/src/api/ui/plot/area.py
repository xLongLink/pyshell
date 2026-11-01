from typing import Literal
from api.ui.plot.common import CartesianTrace


class Area(CartesianTrace):
    """Fill an ordered coordinate series to zero with an optional smooth curve."""

    type: Literal["area"] = "area"
    curve: Literal["linear", "monotone"] = "linear"
