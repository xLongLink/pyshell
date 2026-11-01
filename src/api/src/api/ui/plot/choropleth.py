from typing import Literal, Annotated
from pydantic import Field, model_validator
from api.ui.model import UIModel
from api.ui.types import Number, TextValue, ImmutableList
from api.ui.plot.geojson import FeatureCollection
from api.ui.plot.geography import geography_info


class Choropleth(UIModel):
    """Color built-in or inline boundaries using paired locations and values.

    Example:
        ui.plot(data=[{"type": "choropleth", "geography": "world",
                       "locations": ["FRA", "DEU"], "values": [10, None]}],
                layout={"title": {"text": "Country values"}})

    Prefer built-in boundaries: geography="world" selects countries; an uppercase
    three-letter country code such as "FRA" selects its administrative regions.
    geography_info("world") or geography_info("FRA") returns supported location
    codes and names. First use requires internet access; boundaries are cached.

    Supply 1 to 200 unique locations with equally many numeric values; None means
    missing data. A map figure accepts exactly one trace. This is not Plotly:
    z, featureidkey, customdata, colorscale, marker, colorbar, showscale, and
    hovertemplate are unsupported. Layout does not support geo or map zoom fields.

    Only use geojson for custom boundaries. It overrides geography and requires
    Polygon/MultiPolygon features with unique string IDs matching locations,
    closed WGS84 longitude/latitude rings, at most 200 features, and at most
    4,000 positions total. Built-in boundaries do not use this inline limit.
    Print Choropleth.model_json_schema() for exact field types and constraints.

    """

    type: Literal["choropleth"] = "choropleth"
    name: TextValue = ""
    geography: Annotated[TextValue, Field(pattern=r"^(world|[A-Z]{3})$")] = "world"
    geojson: FeatureCollection | None = Field(
        default=None,
        description="Inline WGS84 polygon boundaries; overrides geography. Feature IDs must match locations. Simplify to 4,000 positions.",
    )
    locations: Annotated[ImmutableList[TextValue], Field(min_length=1, max_length=200)]
    values: Annotated[ImmutableList[Number | None], Field(min_length=1, max_length=200)]

    @model_validator(mode="after")
    def locations_match(self) -> "Choropleth":
        """Require unique, supported locations with paired values."""
        if len(self.locations) != len(self.values):
            raise ValueError("locations and values must have equal lengths")
        if len(set(self.locations)) != len(self.locations):
            raise ValueError("locations must be unique")
        if self.geojson is not None:
            identifiers = {feature.id for feature in self.geojson.features}
            if any(location not in identifiers for location in self.locations):
                raise ValueError("Unknown location for the supplied GeoJSON")
            return self
        try:
            geography = geography_info(self.geography)
        except OSError as error:
            raise ValueError("Map data is unavailable; first use requires internet access") from error
        if geography["level"] != ("country" if self.geography == "world" else "region"):
            raise ValueError("Unsupported level for this geography")
        if any(location not in geography["locations"] for location in self.locations):
            raise ValueError("Unknown location for this geography")
        return self
