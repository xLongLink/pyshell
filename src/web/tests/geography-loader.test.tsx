import { afterEach, expect, test, vi } from "vitest";
import { loadGeography, type Geography } from "../components/plot/geography";

const boundaries: Geography = {
    type: "FeatureCollection",
    geography: "CHE",
    name: "Switzerland",
    level: "region",
    features: [
        {
            type: "Feature",
            id: "CH-ZH",
            properties: { name: "Zurich" },
            geometry: {
                type: "Polygon",
                coordinates: [
                    [
                        [8, 47],
                        [8, 48],
                        [9, 48],
                        [9, 47],
                        [8, 47],
                    ],
                ],
            },
        },
    ],
};

afterEach(() => vi.unstubAllGlobals());

test.each([
    { ...boundaries, geography: "USA" },
    { ...boundaries, features: [] },
    {
        ...boundaries,
        features: [
            {
                ...boundaries.features[0],
                geometry: { type: "Point", coordinates: [8, 47] },
            },
        ],
    },
    {
        ...boundaries,
        features: [
            {
                ...boundaries.features[0],
                geometry: { type: "Polygon", coordinates: [[["8", 47]]] },
            },
        ],
    },
])("rejects invalid geometry or geography: %#", async (data) => {
    vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValueOnce(new Response(JSON.stringify(data))),
    );
    await expect(loadGeography("CHE")).rejects.toBeTruthy();
});

test("rejects unsuccessful HTTP responses", async () => {
    vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(new Response("{}", { status: 502 })),
    );
    await expect(loadGeography("CHE")).rejects.toBeTruthy();
});

test("loads valid boundaries", async () => {
    vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(new Response(JSON.stringify(boundaries))),
    );
    await expect(loadGeography("CHE")).resolves.toEqual(boundaries);
});
