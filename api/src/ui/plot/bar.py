from typing import Literal
from src.ui.plot.common import CartesianTrace


class Bar(CartesianTrace):
    """Compare values on shared Cartesian axes."""

    type: Literal["bar"] = "bar"
