from typing import Annotated
from pydantic import Field, model_validator
from api.ui.model import UIModel
from api.ui.types import Number, TextValue, ImmutableList


class CartesianTrace(UIModel):
    """Keep each trace's coordinates paired and ordered."""

    name: TextValue = ""
    x: Annotated[ImmutableList[Number | TextValue], Field(min_length=1, max_length=200)]
    y: Annotated[ImmutableList[Number | None], Field(min_length=1, max_length=200)]

    @model_validator(mode="after")
    def coordinates(self) -> "CartesianTrace":
        """Require matching coordinate lengths."""
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have equal lengths")
        return self
