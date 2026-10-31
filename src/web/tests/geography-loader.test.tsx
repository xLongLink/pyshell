import { afterEach, expect, test, vi } from "vitest";

afterEach(() => vi.unstubAllGlobals());

test.each([
    ["http", async () => new Response("unavailable", { status: 502 })],
    ["malformed", async () => new Response("{}")],
] as const)(
    "failed %s boundary responses can be retried",
    async (_failure, firstFetch) => {
        // Arrange
        vi.resetModules();
        const fetch = vi.fn().mockImplementationOnce(firstFetch);
        fetch.mockRejectedValueOnce(new Error("retry reached server"));
        vi.stubGlobal("fetch", fetch);
        const { loadGeography } = await import("../components/plot/geography");
        // Act
        // Assert
        await expect(loadGeography("CHE")).rejects.toBeTruthy();
        await expect(loadGeography("CHE")).rejects.toThrow(
            "retry reached server",
        );
        expect(fetch).toHaveBeenNthCalledWith(1, "/geographies/CHE");
        expect(fetch).toHaveBeenNthCalledWith(2, "/geographies/CHE");
    },
);
