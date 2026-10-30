from typing import Literal, ClassVar
from src.ui.types import TextValue
from src.ui.widget import Widget


class Text(Widget):
    """Display plain text without interpreting HTML or Markdown."""

    arguments: ClassVar[tuple[str, ...]] = ("text",)
    component: Literal["Text"] = "Text"
    text: TextValue
    title: TextValue = ""
