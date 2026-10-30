import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { dataset } from "./geography-fixtures";

const geography = dataset("CHE");

beforeEach(() => vi.resetModules());
afterEach(() => vi.unstubAllGlobals());

test("boundary requests are same-origin, shared while pending, and cached after success", async () => {
    const fetch = vi.fn(async () => new Response(JSON.stringify(geography)));
    vi.stubGlobal("fetch", fetch);
    const { loadGeography } = await import("../components/plot/geography");
    const [first, second] = await Promise.all([
        loadGeography("CHE"),
        loadGeography("CHE"),
    ]);
    expect(first.name).toBe("Switzerland");
    expect(second.features).toEqual(first.features);
    expect((await loadGeography("CHE")).features).toEqual(first.features);
    expect(fetch).toHaveBeenCalledExactlyOnceWith("/geographies/CHE");
});

test.each(["http", "network", "malformed", "wrong-country"])(
    "failed %s responses are not cached and can be retried",
    async (failure) => {
        const fetch = vi.fn();
        if (failure === "network")
            fetch.mockRejectedValueOnce(new Error("offline"));
        else
            fetch.mockResolvedValueOnce(
                failure === "http"
                    ? new Response("unavailable", { status: 502 })
                    : new Response(
                          JSON.stringify(
                              failure === "malformed"
                                  ? {}
                                  : { ...geography, geography: "USA" },
                          ),
                      ),
            );
        fetch.mockResolvedValueOnce(new Response(JSON.stringify(geography)));
        vi.stubGlobal("fetch", fetch);
        const { loadGeography } = await import("../components/plot/geography");
        await expect(loadGeography("CHE")).rejects.toBeTruthy();
        expect((await loadGeography("CHE")).name).toBe("Switzerland");
        expect(fetch).toHaveBeenCalledTimes(2);
    },
);
