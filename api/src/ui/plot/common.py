from typing import Annotated
from pydantic import Field, BaseModel, ConfigDict, model_validator
from src.ui.types import Number, TextValue


class PlotModel(BaseModel):
    """Validate the supported figure specification."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CartesianTrace(PlotModel):
    """Keep each trace's coordinates paired and ordered."""

    name: TextValue = ""
    x: Annotated[list[Number | TextValue], Field(min_length=1, max_length=200)]
    y: Annotated[list[Number | None], Field(min_length=1, max_length=200)]

    @model_validator(mode="after")
    def coordinates(self) -> "CartesianTrace":
        """Require matching coordinate lengths."""
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have equal lengths")
        return self
