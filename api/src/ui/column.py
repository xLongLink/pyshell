from typing import Literal
from src.ui.common import Layout


class Column(Layout):
    """Arrange child components vertically."""

    component: Literal["Column"] = "Column"
