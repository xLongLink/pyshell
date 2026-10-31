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
