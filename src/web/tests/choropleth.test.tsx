import { afterEach, expect, test, vi } from "vitest";
import {
    act,
    cleanup,
    fireEvent,
    render,
    screen,
    waitFor,
} from "@testing-library/react";
import { Figure } from "../components/plot/figure";
import { figureProps } from "../lib/a2ui";
import * as geographyLoader from "../components/plot/geography";

const boundaries: geographyLoader.Geography = {
    type: "FeatureCollection",
    geography: "world",
    name: "World",
    level: "country",
    features: [
        {
            type: "Feature",
            id: "FRA",
            properties: { name: "France" },
            geometry: {
                type: "Polygon",
                coordinates: [
                    [
                        [0, 0],
                        [0, 1],
                        [1, 1],
                        [1, 0],
                        [0, 0],
                    ],
                ],
            },
        },
    ],
};

const props = figureProps.parse({
    data: [{ type: "choropleth", locations: ["FRA"], values: [42] }],
});

afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
});

test("a successful retry keeps its boundaries without additional loads", async () => {
    const load = vi
        .spyOn(geographyLoader, "loadGeography")
        .mockRejectedValueOnce(new Error("Network failure"))
        .mockResolvedValue(boundaries);
    const { rerender } = render(<Figure {...props} />);
    expect((await screen.findByRole("alert")).textContent).toContain(
        "Unable to load map boundaries",
    );
    fireEvent.click(await screen.findByRole("button", { name: "Retry" }));
    await screen.findByRole("img", { name: "France: 42" });
    rerender(
        <Figure {...props} layout={{ ...props.layout, showlegend: false }} />,
    );
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
    expect(screen.queryByLabelText("Map legend")).toBeNull();
    expect(load).toHaveBeenCalledTimes(2);
});

test("provider maps render the same boundary for either ring winding without mutating it", async () => {
    const reversed = structuredClone(boundaries);
    reversed.features[0].geometry.coordinates[0].reverse();
    const original = structuredClone(reversed);
    vi.spyOn(geographyLoader, "loadGeography")
        .mockResolvedValueOnce(boundaries)
        .mockResolvedValueOnce(reversed);
    const first = render(<Figure {...props} />);
    const path = (
        await screen.findByRole("img", { name: "France: 42" })
    ).getAttribute("d");
    expect(path).toBeTruthy();
    first.unmount();
    render(<Figure {...props} />);
    expect(
        (await screen.findByRole("img", { name: "France: 42" })).getAttribute(
            "d",
        ),
    ).toBe(path);
    expect(reversed).toEqual(original);
});

test("changing geography replaces the pending map and ignores its old load", async () => {
    let resolve!: (value: geographyLoader.Geography) => void;
    const pending = new Promise<geographyLoader.Geography>((yes) => {
        resolve = yes;
    });
    const regions: geographyLoader.Geography = {
        ...boundaries,
        geography: "FRA",
        name: "France",
        level: "region",
        features: [
            {
                ...boundaries.features[0],
                id: "FR-IDF",
                properties: { name: "Paris" },
            },
        ],
    };
    const load = vi
        .spyOn(geographyLoader, "loadGeography")
        .mockReturnValueOnce(pending)
        .mockResolvedValue(regions);
    const { rerender } = render(<Figure {...props} />);
    await waitFor(() => expect(load).toHaveBeenCalledWith("world"));
    rerender(
        <Figure
            {...figureProps.parse({
                data: [
                    {
                        type: "choropleth",
                        geography: "FRA",
                        locations: ["FR-IDF"],
                        values: [7],
                    },
                ],
            })}
        />,
    );
    await screen.findByRole("img", { name: "Paris: 7" });
    await act(async () => {
        resolve(boundaries);
        await pending;
    });
    expect(screen.getByRole("img", { name: "Paris: 7" })).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(load).toHaveBeenCalledTimes(2);
});

test("hover, focus and live values preserve paths while updating map details", async () => {
    const load = vi.spyOn(geographyLoader, "loadGeography").mockResolvedValue({
        ...boundaries,
        features: [
            ...boundaries.features,
            {
                ...boundaries.features[0],
                id: "ESP",
                properties: { name: "Spain" },
            },
        ],
    });
    const figure = (values: (number | null)[]) =>
        figureProps.parse({
            data: [
                {
                    type: "choropleth",
                    locations: ["FRA", "ESP"],
                    values,
                },
            ],
        });
    const { rerender } = render(<Figure {...figure([42, 0])} />);
    const france = await screen.findByRole("img", { name: "France: 42" });
    const spain = screen.getByRole("img", { name: "Spain: 0" });
    const paths = [france, spain].map((region) => region.getAttribute("d"));
    expect(paths.every((path) => !!path)).toBe(true);
    const originalFill = france.getAttribute("fill");

    fireEvent.mouseEnter(france);
    expect(screen.getByRole("status").textContent).toBe("France: 42");
    fireEvent.mouseLeave(france);
    expect(screen.getByRole("status").textContent).toBe(
        "Hover or focus a region for details.",
    );
    fireEvent.focus(spain);
    expect(screen.getByRole("status").textContent).toBe("Spain: 0");
    fireEvent.blur(spain);
    fireEvent.focus(france);
    expect(screen.getByRole("status").textContent).toBe("France: 42");
    expect([france, spain].map((region) => region.getAttribute("d"))).toEqual(
        paths,
    );

    rerender(<Figure {...figure([10, 20])} />);
    expect(screen.getByRole("img", { name: "France: 10" })).toBe(france);
    expect(screen.getByRole("img", { name: "Spain: 20" })).toBe(spain);
    expect(france.querySelector("title")?.textContent).toBe("France: 10");
    expect(screen.getByRole("status").textContent).toBe("France: 10");
    expect(france.getAttribute("fill")).not.toBe(originalFill);
    expect(screen.getByLabelText("Map legend").textContent).toBe("1020No data");

    rerender(<Figure {...figure([null, 20])} />);
    expect(screen.getByRole("img", { name: "France: No data" })).toBe(france);
    expect(france.getAttribute("fill")).toBe("var(--muted)");
    expect(screen.getByRole("status").textContent).toBe("France: No data");
    expect([france, spain].map((region) => region.getAttribute("d"))).toEqual(
        paths,
    );
    expect(load).toHaveBeenCalledTimes(1);
});

test.each([
    { geography: "world", level: "region" as const, location: "FRA" },
    { geography: "FRA", level: "country" as const, location: "FRA" },
    { geography: "world", level: "country" as const, location: "UNKNOWN" },
])("map rejects mismatched boundaries: %j", async (dataset) => {
    vi.spyOn(geographyLoader, "loadGeography").mockResolvedValue({
        ...boundaries,
        geography: dataset.geography,
        level: dataset.level,
    });
    render(
        <Figure
            {...figureProps.parse({
                data: [
                    {
                        type: "choropleth",
                        geography: dataset.geography,
                        locations: [dataset.location],
                        values: [1],
                    },
                ],
            })}
        />,
    );
    expect((await screen.findByRole("alert")).textContent).toBe(
        "Map locations do not match the selected geography.",
    );
    expect(screen.queryByRole("img")).toBeNull();
});

test("inline boundaries render and update without fetching geography or mutating coordinates", async () => {
    const load = vi.spyOn(geographyLoader, "loadGeography");
    const geojson = {
        type: "FeatureCollection",
        features: boundaries.features,
    };
    const original = JSON.stringify(geojson);
    const figure = (value: number, name = "France") =>
        figureProps.parse({
            data: [
                {
                    type: "choropleth",
                    geojson: {
                        ...geojson,
                        features: [
                            { ...boundaries.features[0], properties: { name } },
                        ],
                    },
                    locations: ["FRA"],
                    values: [value],
                },
            ],
        });
    const { rerender } = render(<Figure {...figure(42)} />);
    const france = await screen.findByRole("img", { name: "France: 42" });
    expect(france.getAttribute("d")).toBeTruthy();
    fireEvent.focus(france);
    expect(screen.getByRole("status").textContent).toBe("France: 42");
    rerender(<Figure {...figure(84, "Updated France")} />);
    expect(
        screen.getByRole("img", { name: "Updated France: 84" }),
    ).toBeTruthy();
    expect(JSON.stringify(geojson)).toBe(original);
    expect(load).not.toHaveBeenCalled();
});

test("inline multipolygons use feature IDs when names are absent", async () => {
    const load = vi.spyOn(geographyLoader, "loadGeography");
    render(
        <Figure
            {...figureProps.parse({
                data: [
                    {
                        type: "choropleth",
                        locations: ["FRA"],
                        values: [7],
                        geojson: {
                            type: "FeatureCollection",
                            features: [
                                {
                                    ...boundaries.features[0],
                                    properties: {},
                                    geometry: {
                                        type: "MultiPolygon",
                                        coordinates: [
                                            boundaries.features[0].geometry
                                                .coordinates,
                                        ],
                                    },
                                },
                            ],
                        },
                    },
                ],
            })}
        />,
    );
    fireEvent.focus(await screen.findByRole("img", { name: "FRA: 7" }));
    expect(screen.getByRole("status").textContent).toBe("FRA: 7");
    expect(load).not.toHaveBeenCalled();
});

test("inline maps render only supplied features and replace their shapes on update", async () => {
    const load = vi.spyOn(geographyLoader, "loadGeography");
    const makeFigure = (ids: string[], coordinates: number[][][]) =>
        figureProps.parse({
            data: [
                {
                    type: "choropleth",
                    locations: ids,
                    values: ids.map(() => 42),
                    geojson: {
                        type: "FeatureCollection",
                        features: ids.map((id) => ({
                            type: "Feature",
                            id,
                            properties: { name: id },
                            geometry: { type: "Polygon", coordinates },
                        })),
                    },
                },
            ],
        });
    const { rerender } = render(
        <Figure
            {...makeFigure(
                ["CHE", "FRA", "ITA"],
                [
                    [
                        [0, 0],
                        [0, 1],
                        [1, 1],
                        [1, 0],
                        [0, 0],
                    ],
                ],
            )}
        />,
    );
    const switzerland = await screen.findByRole("img", { name: "CHE: 42" });
    const originalPath = switzerland.getAttribute("d");
    expect(screen.getAllByRole("img")).toHaveLength(3);
    rerender(
        <Figure
            {...makeFigure(
                ["CHE"],
                [
                    [
                        [0, 0],
                        [0, 1],
                        [1, 0],
                        [0, 0],
                    ],
                ],
            )}
        />,
    );
    expect(screen.getAllByRole("img")).toHaveLength(1);
    expect(
        screen.getByRole("img", { name: "CHE: 42" }).getAttribute("d"),
    ).not.toBe(originalPath);
    expect(load).not.toHaveBeenCalled();
});
