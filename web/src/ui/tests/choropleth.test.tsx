import { afterEach, expect, test, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { Figure } from "../components/plot/figure";
import { figureProps } from "../lib/a2ui";
import * as geographyLoader from "../components/plot/geography";

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
