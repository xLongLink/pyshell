from typing import Literal, Annotated
from pydantic import Field, model_validator
from src.ui.types import Number, TextValue
from src.ui.plot.common import PlotModel
from src.ui.plot.geography import geography_info


class Choropleth(PlotModel):
    """Color provider-backed geographic boundaries by statistical value."""

    type: Literal["choropleth"] = "choropleth"
    name: TextValue = ""
    geography: Annotated[TextValue, Field(pattern=r"^(world|[A-Z]{3})$")] = "world"
    level: Literal["country", "region"] = "country"
    locations: Annotated[list[TextValue], Field(min_length=1, max_length=200)]
    values: Annotated[list[Number | None], Field(min_length=1, max_length=200)]

    @model_validator(mode="after")
    def locations_match(self) -> "Choropleth":
        """Require unique, supported locations with paired values."""
        if self.level != ("country" if self.geography == "world" else "region"):
            raise ValueError("Unsupported level for this geography")
        if len(self.locations) != len(self.values):
            raise ValueError("locations and values must have equal lengths")
        if len(set(self.locations)) != len(self.locations):
            raise ValueError("locations must be unique")
        try:
            geography = geography_info(self.geography)
        except OSError as error:
            raise ValueError("Map data is unavailable; first use requires internet access") from error
        if any(location not in geography["locations"] for location in self.locations):
            raise ValueError("Unknown location for this geography")
        return self
