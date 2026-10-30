from typing import Literal
from src.ui.common import Layout


class Row(Layout):
    """Arrange child components horizontally."""

    component: Literal["Row"] = "Row"
