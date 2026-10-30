import { afterEach, beforeEach, expect, test, vi } from "vitest";
import {
    cleanup,
    fireEvent,
    render,
    screen,
    waitFor,
} from "@testing-library/react";
import { geoArea, geoPath } from "d3-geo";
import { Figure } from "../components/plot/figure";
import { Results } from "../components/results";
import { a2uiBatchSchema, catalogId, figureProps } from "../lib/a2ui";
import * as geographyLoader from "../components/plot/geography";
import { dataset } from "./geography-fixtures";
const world = dataset("world");

beforeEach(() => {
    vi.stubGlobal(
        "fetch",
        vi.fn(async (url: string) => {
            const geography = url.split("/").at(-1)!;
            return new Response(JSON.stringify(dataset(geography)), {
                headers: { "content-type": "application/geo+json" },
            });
        }),
    );
});

afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
});

function figure(values: (number | null)[], showlegend = true) {
    return figureProps.parse({
        data: [
            { type: "choropleth", locations: ["FRA", "DEU", "ESP"], values },
        ],
        layout: { title: { text: "Population" }, showlegend },
    });
}

test("world map requests only its backend boundaries and exposes hover and focus details", async () => {
    render(<Figure {...figure([42, null, 0])} />);
    const france = await screen.findByRole("img", { name: "France: 42" });
    const germany = screen.getByRole("img", { name: "Germany: No data" });
    const spain = screen.getByRole("img", { name: "Spain: 0" });
    expect(screen.getAllByRole("img")).toHaveLength(world.features.length);
    const map = screen.getByRole("group", { name: "World statistical map" });
    expect(map.tagName).toBe("svg");
    expect(map.getAttribute("viewBox")).toBe("0 0 800 450");
    expect(map.hasAttribute("width")).toBe(false);
    expect(map.hasAttribute("height")).toBe(false);
    expect(france.getAttribute("d")).toMatch(/^M/);
    expect(france.querySelector("title")?.textContent).toBe("France: 42");
    expect(france.getAttribute("vector-effect")).toBe("non-scaling-stroke");
    expect(france.getAttribute("fill")).not.toBe(spain.getAttribute("fill"));
    expect(germany.getAttribute("fill")).toBe("var(--muted)");
    expect(spain.getAttribute("fill")).not.toBe("var(--muted)");
    expect(screen.getByLabelText("Map legend")).toBeTruthy();
    expect(screen.queryByRole("link")).toBeNull();
    expect(france.getAttribute("tabindex")).toBe("0");
    fireEvent.mouseEnter(france);
    expect(screen.getByRole("status").textContent).toBe("France: 42");
    fireEvent.mouseLeave(france);
    fireEvent.focus(germany);
    expect(screen.getByRole("status").textContent).toBe("Germany: No data");
    fireEvent.blur(germany);
    expect(screen.getByRole("status").textContent).toContain("Hover or focus");
    expect(fetch).toHaveBeenCalledExactlyOnceWith("/geographies/world");
});

test.each([
    [0, 0, 0],
    [-10, 0, 10],
    [-Number.MAX_VALUE, 0, Number.MAX_VALUE],
    [Number.MIN_VALUE, Number.MIN_VALUE, Number.MIN_VALUE],
] as number[][])("map color scale handles %j", async (...values) => {
    render(<Figure {...figure(values as number[])} />);
    await screen.findByRole("group", { name: "World statistical map" });
    await waitFor(() =>
        expect(screen.getAllByRole("img").length).toBeGreaterThan(0),
    );
    for (const path of screen.getAllByRole("img")) {
        expect(path.getAttribute("fill")).not.toMatch(/NaN|Infinity/);
        expect(path.getAttribute("d")).not.toMatch(/NaN|Infinity/);
    }
    if (values[0] === values[2]) {
        expect(
            screen.getByRole("img", { name: /^France:/ }).getAttribute("fill"),
        ).toBe(
            screen.getByRole("img", { name: /^Spain:/ }).getAttribute("fill"),
        );
    }
});

test("all-missing maps retain neutral boundaries and can hide the legend", async () => {
    render(<Figure {...figure([null, null, null], false)} />);
    const path = await screen.findByRole("img", { name: "France: No data" });
    expect(path.getAttribute("fill")).toBe("var(--muted)");
    expect(screen.getByText("No values to display.")).toBeTruthy();
    expect(screen.queryByLabelText("Map legend")).toBeNull();
});

test("published map surfaces accept updates and geography changes without stale values", async () => {
    const initial = a2uiBatchSchema.parse([
        { version: "v0.9", createSurface: { surfaceId: "map", catalogId } },
        {
            version: "v0.9",
            updateComponents: {
                surfaceId: "map",
                components: [
                    {
                        id: "root",
                        component: "Figure",
                        ...figure([42, null, 0]),
                    },
                ],
            },
        },
    ]);
    const { rerender } = render(<Results messages={initial} />);
    const path = await screen.findByRole("img", { name: "France: 42" });
    fireEvent.focus(path);
    const update = a2uiBatchSchema.parse([
        {
            version: "v0.9",
            updateComponents: {
                surfaceId: "map",
                components: [
                    {
                        id: "root",
                        component: "Figure",
                        ...figure([99, 2, 1], false),
                    },
                ],
            },
        },
    ]);
    rerender(<Results messages={[...initial, ...update]} />);
    await screen.findByRole("img", { name: "France: 99" });
    expect(screen.getByRole("status").textContent).toBe("France: 99");
    expect(screen.queryByLabelText("Map legend")).toBeNull();
    const regional = a2uiBatchSchema.parse([
        {
            version: "v0.9",
            updateComponents: {
                surfaceId: "map",
                components: [
                    {
                        id: "root",
                        component: "Figure",
                        data: [
                            {
                                type: "choropleth",
                                geography: "FRA",
                                level: "region",
                                locations: ["FR-BRE"],
                                values: [48],
                            },
                        ],
                    },
                ],
            },
        },
    ]);
    rerender(<Results messages={[...initial, ...update, ...regional]} />);
    await screen.findByRole("img", { name: "Bretagne: 48" });
    expect(screen.queryByRole("img", { name: "France: 99" })).toBeNull();
    expect(screen.getAllByRole("article")).toHaveLength(1);
});

test.each(["world", "FRA", "USA", "FJI"])(
    "%s provider polygons are normalized to d3 winding and finite fitted paths",
    async (geography) => {
        const boundaries = await geographyLoader.loadGeography(geography);
        expect(
            Object.fromEntries(
                boundaries.features.map((feature) => [
                    feature.id,
                    feature.properties?.name,
                ]),
            ),
        ).toEqual(
            Object.fromEntries(
                dataset(geography).features.map((feature) => [
                    feature.id,
                    feature.properties.name,
                ]),
            ),
        );
        const projection = geographyLoader.mapProjection(
            boundaries,
            geography === "world",
        );
        const path = geoPath(projection);
        for (const feature of boundaries.features) {
            expect(geoArea(feature)).toBeLessThan(2 * Math.PI);
            expect(path(feature)).toMatch(/^M/);
            expect(path(feature)).not.toMatch(/NaN|Infinity/);
        }
    },
);

test("loading a country's boundaries can fail and be retried", async () => {
    const load = vi
        .spyOn(geographyLoader, "loadGeography")
        .mockRejectedValueOnce(new Error("Network failure"));
    render(<Figure {...figure([42, 2, 0])} />);
    const retry = await screen.findByRole("button", { name: "Retry" });
    expect(screen.getByRole("alert").textContent).toContain(
        "Unable to load map boundaries",
    );
    fireEvent.click(retry);
    await screen.findByRole("img", { name: "France: 42" });
    expect(load).toHaveBeenCalledTimes(2);
});

test("a slow boundary load cannot replace a newly selected geography", async () => {
    let finishWorld!: (data: geographyLoader.Geography) => void;
    const pending = new Promise<geographyLoader.Geography>((resolve) => {
        finishWorld = resolve;
    });
    vi.spyOn(geographyLoader, "loadGeography").mockReturnValueOnce(pending);
    const { rerender } = render(<Figure {...figure([42, 2, 0])} />);
    await screen.findByText("Loading map…");
    rerender(
        <Figure
            {...figureProps.parse({
                data: [
                    {
                        type: "choropleth",
                        geography: "FRA",
                        level: "region",
                        locations: ["FR-IDF"],
                        values: [75],
                    },
                ],
            })}
        />,
    );
    await screen.findByRole("img", { name: "Île-de-France: 75" });
    finishWorld(world);
    await waitFor(() =>
        expect(screen.queryByRole("img", { name: "France: 42" })).toBeNull(),
    );
    expect(screen.getAllByRole("img")).toHaveLength(3);
    expect(screen.getByRole("img", { name: "Corse: No data" })).toBeTruthy();
});

test("location membership is checked against the backend dataset before rendering", async () => {
    render(
        <Figure
            {...figureProps.parse({
                data: [
                    {
                        type: "choropleth",
                        geography: "CHE",
                        level: "region",
                        locations: ["__proto__"],
                        values: [1],
                    },
                ],
            })}
        />,
    );
    await screen.findByRole("alert");
    expect(screen.getByRole("alert").textContent).toContain(
        "Map locations do not match",
    );
    expect(screen.queryByRole("img")).toBeNull();
});

test("dateline-spanning country maps fit their divisions together", async () => {
    const boundaries = await geographyLoader.loadGeography("FJI");
    const projection = geographyLoader.mapProjection(boundaries, false);
    const bounds = geoPath(projection).bounds(boundaries);
    expect(bounds[1][0] - bounds[0][0]).toBeGreaterThan(400);
    expect(bounds[1][0] - bounds[0][0]).toBeLessThanOrEqual(777);
    expect(bounds[1][1] - bounds[0][1]).toBeGreaterThan(100);
});
