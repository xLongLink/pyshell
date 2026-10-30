from typing import Literal, ClassVar
from src.ui.types import TextValue
from src.ui.widget import Widget


class Metric(Widget):
    """Display a labeled, formatted value."""

    arguments: ClassVar[tuple[str, ...]] = ("label", "value")
    component: Literal["Metric"] = "Metric"
    label: TextValue
    value: TextValue
    detail: TextValue = ""
