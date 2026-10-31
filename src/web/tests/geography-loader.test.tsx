import { afterEach, expect, test, vi } from "vitest";
import type { Geography } from "../components/plot/geography";

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
    ...[
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
    ].map(
        (data) =>
            [
                "invalid geometry or geography",
                async () => new Response(JSON.stringify(data)),
            ] as const,
    ),
] as const)("rejects %s boundary responses", async (_failure, firstFetch) => {
    // Arrange
    vi.resetModules();
    vi.stubGlobal("fetch", vi.fn().mockImplementationOnce(firstFetch));
    const { loadGeography } = await import("../components/plot/geography");
    // Act
    // Assert
    await expect(loadGeography("CHE")).rejects.toBeTruthy();
});

test.each([
    ["HTTP", 502],
    ["validation", 200],
] as const)(
    "failed %s boundary requests can be retried",
    async (_failure, status) => {
        vi.resetModules();
        const fetch = vi
            .fn()
            .mockResolvedValueOnce(new Response("{}", { status }));
        fetch.mockImplementation(() =>
            Promise.resolve(new Response(JSON.stringify(boundaries))),
        );
        vi.stubGlobal("fetch", fetch);
        const { loadGeography } = await import("../components/plot/geography");

        await expect(loadGeography("CHE")).rejects.toBeTruthy();
        await expect(loadGeography("CHE")).resolves.toEqual(boundaries);
        expect(fetch).toHaveBeenCalledTimes(2);
    },
);

test("successful geography requests are shared only until settlement", async () => {
    vi.resetModules();
    const fetch = vi
        .fn()
        .mockImplementation(() =>
            Promise.resolve(new Response(JSON.stringify(boundaries))),
        );
    vi.stubGlobal("fetch", fetch);
    const { loadGeography } = await import("../components/plot/geography");

    const first = loadGeography("CHE");
    expect(loadGeography("CHE")).toBe(first);
    expect(fetch).toHaveBeenCalledTimes(1);
    const original = await first;
    const second = loadGeography("CHE");
    expect(second).not.toBe(first);
    expect(loadGeography("CHE")).toBe(second);
    const reloaded = await second;
    expect(reloaded).toEqual(original);
    expect(reloaded).not.toBe(original);
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(fetch).toHaveBeenNthCalledWith(1, "/geographies/CHE");
    expect(fetch).toHaveBeenNthCalledWith(2, "/geographies/CHE");
});
