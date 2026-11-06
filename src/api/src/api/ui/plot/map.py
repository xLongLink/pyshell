from types import MappingProxyType
from typing import Literal, Annotated
from pydantic import Field, field_validator, model_validator, field_serializer
from api.ui.model import UIModel
from api.ui.types import Number, TextValue, Identifier, ImmutableList
from collections.abc import Mapping
from api.ui.plot.geojson import Polygon, Position, MultiPolygon


class Point(UIModel):
    """Locate a feature using WGS84 longitude and latitude."""

    type: Literal["Point"] = "Point"
    coordinates: Position


class MapFeature(UIModel):
    """Pair geographic geometry with browser-visible scalar properties."""

    type: Literal["Feature"] = "Feature"
    id: Annotated[TextValue, Field(min_length=1)]
    geometry: Annotated[Point | Polygon | MultiPolygon, Field(discriminator="type")]
    properties: Annotated[Mapping[TextValue, TextValue | Number | bool | None], Field(max_length=12)] = Field(
        default_factory=dict, validate_default=True
    )

    @field_validator("properties")
    @classmethod
    def freeze_properties(cls, value: Mapping) -> Mapping:
        """Keep published properties immutable like the rest of the UI model."""
        return MappingProxyType(dict(value))

    @field_serializer("properties")
    def serialize_properties(self, value: Mapping) -> dict:
        """Serialize the read-only properties as a JSON object."""
        return dict(value)


class MapData(UIModel):
    """Supply geographic features, including an empty query result."""

    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: Annotated[ImmutableList[MapFeature], Field(max_length=200)] = ()


class MapLayer(UIModel):
    """Style a named geographic dataset and choose its popup fields."""

    id: Identifier
    data: MapData
    name: TextValue = ""
    color: Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")] = "#2563eb"
    popup_fields: Annotated[ImmutableList[TextValue], Field(max_length=12)] = ()


class MapView(UIModel):
    """Request navigation by country, bounds, or center and zoom."""

    zoom: Annotated[Number, Field(ge=0, le=22)] | None = None
    bounds: Annotated[ImmutableList[Position], Field(min_length=2, max_length=2)] | None = None
    center: Position | None = None
    country: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | None = None
    revision: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def camera(self) -> "MapView":
        """Require a single unambiguous geographic destination."""
        if sum(value is not None for value in (self.bounds, self.center, self.country)) != 1:
            raise ValueError("Specify exactly one of bounds, center, or country")
        if self.zoom is not None and self.center is None:
            raise ValueError("zoom requires center")
        if self.bounds is not None and self.bounds[0][1] >= self.bounds[1][1]:
            raise ValueError("bounds must run from southwest to northeast")
        return self


class Map(UIModel):
    """Display interactive geographic data, independently of choropleth charts.

    layers accepts up to eight MapLayer objects or dictionaries. Each layer has
    id, data (a WGS84 GeoJSON FeatureCollection), optional name, hex color, and
    popup_fields (property names to display on click). Features need unique string
    IDs within their layer and Point, Polygon, or MultiPolygon geometry. Properties
    are limited to 12 scalar values; pass only data intended for browser display.
    Empty collections are supported. All layers together accept 200 features and
    4,000 positions, subject to the existing 256 KiB surface publication limit.

    view accepts exactly one of country (ISO alpha-3, e.g. CHE), bounds
    ([[west, south], [east, north]]), or center ([longitude, latitude], optional
    zoom). Country bounds require an internet connection and provider coverage.
    Omit view on a data update to preserve the current camera. Reuse the returned
    figure's set_view() for explicit navigation and update(data=[...]) for layers.
    MapLibre renders OpenStreetMap raster tiles; internet access is required.
    MapLibre does not supply renovation records or infer missing feature data.
    """

    type: Literal["map"] = "map"
    view: MapView | None = None
    layers: Annotated[ImmutableList[MapLayer], Field(max_length=8)] = ()

    @model_validator(mode="after")
    def budget(self) -> "Map":
        """Bound aggregate geometry and require stable, distinct identifiers."""
        if len({layer.id for layer in self.layers}) != len(self.layers):
            raise ValueError("Map layer IDs must be unique")
        if sum(len(layer.data.features) for layer in self.layers) > 200:
            raise ValueError("A map accepts at most 200 total features")
        positions = 0
        for layer in self.layers:
            if len({feature.id for feature in layer.data.features}) != len(layer.data.features):
                raise ValueError("Map feature IDs must be unique within each layer")
            for feature in layer.data.features:
                geometry = feature.geometry
                if isinstance(geometry, Point):
                    positions += 1
                else:
                    polygons = (geometry.coordinates,) if isinstance(geometry, Polygon) else geometry.coordinates
                    positions += sum(len(ring) for polygon in polygons for ring in polygon)
        if positions > 4000:
            raise ValueError("A map accepts at most 4,000 positions; simplify the geometry first")
        return self
