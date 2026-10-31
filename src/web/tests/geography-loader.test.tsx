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
    ["http", async () => new Response("unavailable", { status: 502 })],
    ["malformed", async () => new Response("{}")],
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
] as const)(
    "failed %s boundary responses can be retried with concurrent requests deduplicated",
    async (_failure, firstFetch) => {
        // Arrange
        vi.resetModules();
        const fetch = vi.fn().mockImplementationOnce(firstFetch);
        fetch.mockImplementation(() =>
            Promise.resolve(new Response(JSON.stringify(boundaries))),
        );
        vi.stubGlobal("fetch", fetch);
        const { loadGeography } = await import("../components/plot/geography");
        // Act
        // Assert
        await expect(loadGeography("CHE")).rejects.toBeTruthy();
        const retry = loadGeography("CHE");
        expect(loadGeography("CHE")).toBe(retry);
        await expect(retry).resolves.toEqual(boundaries);
        expect(fetch).toHaveBeenCalledTimes(2);
        expect(fetch).toHaveBeenNthCalledWith(1, "/geographies/CHE");
        expect(fetch).toHaveBeenNthCalledWith(2, "/geographies/CHE");
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
});

test("rendering does not retain malformed attribution metadata", async () => {
    vi.resetModules();
    vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
            new Response(
                JSON.stringify({
                    ...boundaries,
                    source: { source: 123, url: "invalid", license: false },
                }),
            ),
        ),
    );
    const { loadGeography } = await import("../components/plot/geography");

    const loaded = await loadGeography("CHE");
    expect(loaded).toEqual(boundaries);
    expect(loaded).not.toHaveProperty("source");
});
