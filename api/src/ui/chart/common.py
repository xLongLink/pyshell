from typing import ClassVar, Annotated
from pydantic import Field, BaseModel, ConfigDict
from src.ui.types import Number, TextValue
from src.ui.widget import Widget


class Point(BaseModel):
    """Preserve missing chart values as gaps."""

    model_config = ConfigDict(extra="forbid", strict=True)
    label: TextValue
    value: Number | None


class Chart(Widget):
    """Share the title and labeled numeric series across chart types."""

    arguments: ClassVar[tuple[str, ...]] = ("points",)
    title: TextValue = ""
    points: Annotated[list[Point], Field(min_length=1, max_length=200)]
