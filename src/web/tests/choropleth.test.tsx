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
    source: {
        source: "Test",
        url: "https://example.com",
        license: "Test",
        licenseUrl: "https://example.com/license",
    },
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

test("failed map loads show an error and allow retry", async () => {
    const load = vi
        .spyOn(geographyLoader, "loadGeography")
        .mockRejectedValue(new Error("Network failure"));
    render(
        <Figure
            {...figureProps.parse({
                data: [
                    {
                        type: "choropleth",
                        locations: ["FRA"],
                        values: [42],
                    },
                ],
            })}
        />,
    );
    fireEvent.click(await screen.findByRole("button", { name: "Retry" }));
    await screen.findByRole("button", { name: "Retry" });
    expect(screen.getByRole("alert").textContent).toContain(
        "Unable to load map boundaries",
    );
    expect(load).toHaveBeenCalledTimes(2);
});

test("a successful retry keeps its boundaries without additional loads", async () => {
    const load = vi
        .spyOn(geographyLoader, "loadGeography")
        .mockRejectedValueOnce(new Error("Network failure"))
        .mockResolvedValue(boundaries);
    const { rerender } = render(<Figure {...props} />);
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
                        level: "region",
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
