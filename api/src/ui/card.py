from typing import Literal
from src.ui.common import Layout


class Card(Layout):
    """Group related components in a card."""

    component: Literal["Card"] = "Card"
