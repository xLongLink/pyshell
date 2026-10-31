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

test.each([
    ["HTTP", 502],
    ["validation", 200],
] as const)(
    "failed %s boundary requests can be retried",
    async (_failure, status) => {
        const fetch = vi
            .fn()
            .mockResolvedValueOnce(new Response("{}", { status }));
        fetch.mockImplementation(() =>
            Promise.resolve(new Response(JSON.stringify(boundaries))),
        );
        vi.stubGlobal("fetch", fetch);

        await expect(loadGeography("CHE")).rejects.toBeTruthy();
        await expect(loadGeography("CHE")).resolves.toEqual(boundaries);
        expect(fetch).toHaveBeenCalledTimes(2);
    },
);
