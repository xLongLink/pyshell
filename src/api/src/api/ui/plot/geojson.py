from typing import Literal, Annotated
from pydantic import Field, ConfigDict, AfterValidator, model_validator
from api.ui.model import UIModel
from api.ui.types import Number, TextValue, ImmutableList


def _position(value: tuple[float, ...]) -> tuple[float, ...]:
    if not (-180 <= value[0] <= 180 and -90 <= value[1] <= 90):
        raise ValueError("GeoJSON positions must be longitude/latitude coordinates")
    return value


Position = Annotated[ImmutableList[Number], Field(min_length=2, max_length=2), AfterValidator(_position)]


def _ring(value: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    if value[0] != value[-1]:
        raise ValueError("GeoJSON polygon rings must be closed")
    return value


Ring = Annotated[ImmutableList[Position], Field(min_length=4, max_length=4000), AfterValidator(_ring)]
PolygonCoordinates = Annotated[ImmutableList[Ring], Field(min_length=1, max_length=200)]


class Polygon(UIModel):
    """Describe a polygon using WGS84 longitude/latitude rings."""

    type: Literal["Polygon"] = "Polygon"
    coordinates: PolygonCoordinates


class MultiPolygon(UIModel):
    """Describe a collection of polygon coordinate arrays."""

    type: Literal["MultiPolygon"] = "MultiPolygon"
    coordinates: Annotated[ImmutableList[PolygonCoordinates], Field(min_length=1, max_length=200)]


class FeatureProperties(UIModel):
    """Keep a feature's display name from its source properties."""

    model_config = ConfigDict(extra="ignore")
    name: TextValue = ""


class Feature(UIModel):
    """Pair polygon boundaries with a unique string identifier."""

    model_config = ConfigDict(extra="ignore")
    type: Literal["Feature"] = "Feature"
    id: Annotated[TextValue, Field(min_length=1)]
    properties: FeatureProperties = Field(default_factory=FeatureProperties)
    geometry: Annotated[Polygon | MultiPolygon, Field(discriminator="type")]


class FeatureCollection(UIModel):
    """Supply inline GeoJSON with at most 200 features and 4,000 positions."""

    model_config = ConfigDict(extra="ignore")
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: Annotated[ImmutableList[Feature], Field(min_length=1, max_length=200)]

    @model_validator(mode="after")
    def boundaries(self) -> "FeatureCollection":
        """Require distinct feature IDs and bounded coordinate complexity."""
        if len({feature.id for feature in self.features}) != len(self.features):
            raise ValueError("GeoJSON feature IDs must be unique")
        count = 0
        for feature in self.features:
            geometry = feature.geometry
            polygons = (geometry.coordinates,) if isinstance(geometry, Polygon) else geometry.coordinates
            count += sum(len(ring) for polygon in polygons for ring in polygon)
        if count > 4000:
            raise ValueError("Inline GeoJSON accepts at most 4,000 positions; simplify the boundaries first")
        return self
