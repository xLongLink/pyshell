from typing import ClassVar, Annotated
from pydantic import Field
from src.ui.widget import Widget
from a2ui.core.schema import Child


class Layout(Widget):
    """Arrange explicitly referenced child components."""

    arguments: ClassVar[tuple[str, ...]] = ("children",)
    children: Annotated[list[Child], Field(min_length=1, max_length=50)]
