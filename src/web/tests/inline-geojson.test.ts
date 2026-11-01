import { expect, test } from "vitest";
import { figureProps } from "../lib/a2ui";

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
