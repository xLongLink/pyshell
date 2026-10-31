import { afterEach, expect, test, vi } from "vitest";
import type { Geography } from "../components/plot/geography";

const boundaries: Geography = {
    type: "FeatureCollection",
    geography: "CHE",
    name: "Switzerland",
    level: "region",
    source: {
        source: "Test",
        url: "https://example.com",
        license: "Test",
        licenseUrl: "https://example.com/license",
    },
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
] as const)(
    "failed %s boundary responses can be retried and cached after recovery",
    async (_failure, firstFetch) => {
        // Arrange
        vi.resetModules();
        const fetch = vi.fn().mockImplementationOnce(firstFetch);
        fetch.mockResolvedValueOnce(new Response(JSON.stringify(boundaries)));
        vi.stubGlobal("fetch", fetch);
        const { loadGeography } = await import("../components/plot/geography");
        // Act
        // Assert
        await expect(loadGeography("CHE")).rejects.toBeTruthy();
        const retry = loadGeography("CHE");
        await expect(retry).resolves.toEqual(boundaries);
        expect(loadGeography("CHE")).toBe(retry);
        expect(fetch).toHaveBeenCalledTimes(2);
        expect(fetch).toHaveBeenNthCalledWith(1, "/geographies/CHE");
        expect(fetch).toHaveBeenNthCalledWith(2, "/geographies/CHE");
    },
);
