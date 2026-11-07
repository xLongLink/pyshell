import { expect, test } from "vitest";
import { figureProps } from "../lib/a2ui";
import { geojsonSchema } from "../lib/geojson";
import { mapTraceSchema } from "../lib/map";

function figure() {
    return {
        data: [
            {
                type: "choropleth",
                locations: ["CHE"],
                values: [42],
                geojson: {
                    type: "FeatureCollection",
                    features: [
                        {
                            type: "Feature",
                            id: "CHE",
                            properties: { name: "Switzerland" },
                            geometry: {
                                type: "Polygon",
                                coordinates: [
                                    [
                                        [7, 46],
                                        [8, 46],
                                        [8, 47],
                                        [7, 46],
                                    ],
                                ],
                            },
                        },
                    ],
                },
            },
        ],
    };
}

test("inline GeoJSON survives figure validation", () => {
    const input = figure();
    expect(figureProps.parse(input).data[0]).toMatchObject(input.data[0]);
});

test.each([
    "duplicate",
    "open",
    "short",
    "empty",
    "unknown",
    "longitude",
    "latitude",
    "nonfinite",
])("rejects invalid inline GeoJSON: %s", (caseName) => {
    const input = figure();
    const trace = input.data[0];
    const feature = trace.geojson.features[0];
    const ring = feature.geometry.coordinates[0];
    if (caseName === "duplicate")
        trace.geojson.features.push(structuredClone(feature));
    if (caseName === "open") ring[ring.length - 1] = [7, 47];
    if (caseName === "short") ring.pop();
    if (caseName === "empty") trace.geojson.features = [];
    if (caseName === "unknown") trace.locations = ["OTHER"];
    if (caseName === "longitude") ring[1] = [181, 46];
    if (caseName === "latitude") ring[1] = [7, 91];
    if (caseName === "nonfinite") ring[1] = [Infinity, 46];
    expect(figureProps.safeParse(input).success).toBe(false);
});

test("bounds total positions across features", () => {
    const input = figure();
    const feature = input.data[0].geojson.features[0];
    feature.geometry.coordinates = [
        Array.from({ length: 4000 }, () => [7, 46]),
    ];
    const second = structuredClone(feature);
    second.id = "FRA";
    second.geometry.coordinates = [
        [
            [7, 46],
            [8, 46],
            [8, 47],
            [7, 46],
        ],
    ];
    input.data[0].geojson.features.push(second);
    expect(figureProps.safeParse(input).success).toBe(false);
});

test.each(["Polygon", "MultiPolygon"])(
    "inline and map schemas validate %s geometry consistently",
    (type) => {
        const polygon = figure().data[0].geojson.features[0].geometry.coordinates;
        const geometry = {
            type,
            coordinates: type === "Polygon" ? polygon : [polygon],
        };
        for (const candidate of [geometry, { ...geometry, extra: true }]) {
            const feature = {
                type: "Feature",
                id: "CHE",
                geometry: candidate,
            };
            const data = { type: "FeatureCollection", features: [feature] };
            const valid = !("extra" in candidate);
            expect(geojsonSchema.safeParse(data).success).toBe(valid);
            expect(
                mapTraceSchema.safeParse({
                    type: "map",
                    layers: [{ id: "regions", data }],
                }).success,
            ).toBe(valid);
        }
    },
);

test.each([4000, 4001])(
    "polygon budgets count holes and closing positions: %i",
    (positions) => {
        const feature = {
            type: "Feature",
            id: "CHE",
            geometry: {
                type: "MultiPolygon",
                coordinates: [
                    [
                        Array.from({ length: positions - 4 }, () => [7, 46]),
                        Array.from({ length: 4 }, () => [7, 46]),
                    ],
                ],
            },
        };
        const data = { type: "FeatureCollection", features: [feature] };
        expect(geojsonSchema.safeParse(data).success).toBe(positions === 4000);
        expect(
            mapTraceSchema.safeParse({
                type: "map",
                layers: [{ id: "regions", data }],
            }).success,
        ).toBe(positions === 4000);
    },
);

test.each([3999, 4000])(
    "map position budgets include points across layers: %i polygon positions",
    (positions) => {
        const polygon = {
            type: "Feature",
            id: "region",
            geometry: {
                type: "Polygon",
                coordinates: [Array.from({ length: positions }, () => [7, 46])],
            },
        };
        const point = {
            type: "Feature",
            id: "point",
            geometry: { type: "Point", coordinates: [7, 46] },
        };
        expect(
            mapTraceSchema.safeParse({
                type: "map",
                layers: [polygon, point].map((feature, index) => ({
                    id: `layer_${index}`,
                    data: { type: "FeatureCollection", features: [feature] },
                })),
            }).success,
        ).toBe(positions === 3999);
    },
);
