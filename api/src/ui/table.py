from typing import Literal, ClassVar, Annotated
from pydantic import Field, model_validator
from src.ui.types import Cell, TextValue
from src.ui.widget import Widget


class Table(Widget):
    """Display bounded rectangular data with named columns."""

    arguments: ClassVar[tuple[str, ...]] = ("columns", "rows")
    component: Literal["Table"] = "Table"
    title: TextValue = ""
    columns: Annotated[list[TextValue], Field(min_length=1, max_length=20)]
    rows: Annotated[list[Annotated[list[Cell], Field(max_length=20)]], Field(max_length=500)]

    @model_validator(mode="after")
    def rectangular(self) -> "Table":
        """Reject rows that do not match the table's columns."""
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("Each table row must match the number of columns")
        return self
