from typing import Literal, Annotated
from pydantic import Field, model_validator
from api.ui.model import UIModel
from api.ui.types import Number, TextValue, ImmutableList


class Pie(UIModel):
    """Show nonnegative shares, omitting missing values from slices."""

    type: Literal["pie"] = "pie"
    name: TextValue = ""
    labels: Annotated[ImmutableList[TextValue], Field(min_length=1, max_length=200)]
    values: Annotated[ImmutableList[Annotated[Number, Field(ge=0)] | None], Field(min_length=1, max_length=200)]

    @model_validator(mode="after")
    def coordinates(self) -> "Pie":
        """Require a label for each value."""
        if len(self.labels) != len(self.values):
            raise ValueError("labels and values must have equal lengths")
        return self
