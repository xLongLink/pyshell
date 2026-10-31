import { afterEach, expect, test, vi } from "vitest";

afterEach(() => vi.unstubAllGlobals());

test.each(["http", "network", "malformed"])(
    "failed %s boundary responses can be retried",
    async (failure) => {
        vi.resetModules();
        const fetch = vi.fn();
        if (failure === "network")
            fetch.mockRejectedValueOnce(new Error("offline"));
        else
            fetch.mockResolvedValueOnce(
                failure === "http"
                    ? new Response("unavailable", { status: 502 })
                    : new Response("{}"),
            );
        fetch.mockRejectedValueOnce(new Error("retry reached server"));
        vi.stubGlobal("fetch", fetch);
        const { loadGeography } = await import("../components/plot/geography");
        await expect(loadGeography("CHE")).rejects.toBeTruthy();
        await expect(loadGeography("CHE")).rejects.toThrow(
            "retry reached server",
        );
        expect(fetch).toHaveBeenNthCalledWith(1, "/geographies/CHE");
        expect(fetch).toHaveBeenNthCalledWith(2, "/geographies/CHE");
    },
);
