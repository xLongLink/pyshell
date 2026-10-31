import { afterEach, expect, test } from "vitest";
import {
    cleanup,
    fireEvent,
    render,
    screen,
    waitFor,
    within,
} from "@testing-library/react";
import { Results } from "../components/results";
import { figureProps } from "../lib/a2ui";
import type { Surface } from "../lib/workspace";

afterEach(cleanup);

function figure(
    data: unknown[],
    layout?: unknown,
    surfaceId = "sales",
): Surface[] {
    return [
        {
            id: surfaceId,
            pinned: false,
            props: figureProps.parse({ data, layout }),
        },
    ];
}

function bar(title: string, surfaceId = "sales") {
    return figure(
        [{ type: "bar", x: ["A"], y: [42] }],
        { title: { text: title } },
        surfaceId,
    );
}

test("appending another surface preserves an existing chart and its keyboard focus", async () => {
    const surfaces = bar("First", "first");
    const { rerender } = render(<Results surfaces={surfaces} />);
    const chart = await screen.findByRole("application");
    chart.focus();
    rerender(<Results surfaces={[...surfaces, ...bar("Second", "second")]} />);
    await waitFor(() => expect(screen.getByText("Second")).toBeTruthy());
    expect(document.activeElement).toBe(chart);
    expect(
        screen
            .getByRole("article", { name: "Widget first" })
            .querySelector('[role="application"]'),
    ).toBe(chart);
});

test("replacement snapshots replace surfaces even when their count is unchanged", async () => {
    const { rerender } = render(<Results surfaces={bar("Old", "old")} />);
    await waitFor(() => expect(screen.getByText("Old")).toBeTruthy());
    rerender(<Results surfaces={bar("New", "new")} />);
    await waitFor(() => expect(screen.getByText("New")).toBeTruthy());
    expect(screen.queryByText("Old")).toBeNull();
    expect(screen.getAllByRole("article")).toHaveLength(1);
});

test("chart titles remain inert even when they contain HTML", async () => {
    const text = '<img src=x onerror="alert(1)">';
    const { container } = render(<Results surfaces={bar(text)} />);
    await waitFor(() => expect(screen.getByText(text)).toBeTruthy());
    expect(container.querySelector("img")).toBeNull();
});

test("pie rendering preserves missing slice positions and colors", async () => {
    const { container } = render(
        <Results
            surfaces={figure(
                [
                    {
                        type: "pie",
                        labels: ["Apples", "Pears", "Oranges"],
                        values: [42, null, 84],
                    },
                ],
                { title: { text: "Revenue" } },
            )}
        />,
    );
    await waitFor(() => expect(screen.getByText("Revenue")).toBeTruthy());
    await waitFor(() =>
        expect(container.querySelector(".recharts-pie")).not.toBeNull(),
    );
    const chart = screen.getByRole("application");
    expect(chart.getAttribute("tabindex")).toBe("0");
    chart.focus();
    expect(document.activeElement).toBe(chart);
    const legend = within(
        container.querySelector<HTMLElement>(".recharts-legend-wrapper")!,
    );
    expect(legend.getByText("Apples")).toBeTruthy();
    expect(legend.getByText("Oranges")).toBeTruthy();
    expect(legend.queryByText("Pears")).toBeNull();
    expect(container.querySelectorAll(".recharts-pie-sector")).toHaveLength(2);
    expect(
        [...container.querySelectorAll(".recharts-pie-sector path")].map(
            (slice) => slice.getAttribute("fill"),
        ),
    ).toEqual(["var(--chart-1)", "var(--chart-3)"]);
});

test("pie legends and tooltips resolve slice labels as inert text", async () => {
    const label = '</style><img src=x onerror="alert(1)">';
    const { container } = render(
        <Results
            surfaces={figure([{ type: "pie", labels: [label], values: [42] }])}
        />,
    );
    await waitFor(() => expect(screen.getAllByText(label)).toHaveLength(1));
    expect(container.querySelector("img")).toBeNull();
    const slice = container.querySelector(".recharts-pie-sector path")!;
    expect(slice.getAttribute("fill")).toBe("var(--chart-1)");
    fireEvent.mouseEnter(container.querySelector(".recharts-pie-sector")!);
    await waitFor(() =>
        expect(
            container.querySelector(".recharts-tooltip-wrapper")?.textContent,
        ).toBe(`${label}42`),
    );
    expect(
        container.querySelector<HTMLElement>(
            ".recharts-tooltip-wrapper [style]",
        )?.style.backgroundColor,
    ).toBe(slice.getAttribute("fill"));
    expect(container.querySelector("img")).toBeNull();
});

test.each(["bar", "scatter"] as const)(
    "%s legends and keyboard tooltips preserve user labels and default series names",
    async (type) => {
        const name = '<img src=x onerror="alert(1)">__proto__';
        const { container } = render(
            <Results
                surfaces={figure([
                    { type, name, x: ["A"], y: [1234.5] },
                    { type, x: ["B"], y: [10] },
                ])}
            />,
        );
        await screen.findByText(name);
        expect(screen.getByText("Series 2")).toBeTruthy();
        fireEvent.focus(screen.getByRole("application"));
        await waitFor(() =>
            expect(
                container.querySelector(".recharts-tooltip-wrapper")
                    ?.textContent,
            ).toBe(`A${name}${(1234.5).toLocaleString()}`),
        );
        expect(
            container.querySelector<HTMLElement>(
                ".recharts-tooltip-wrapper [style]",
            )?.style.backgroundColor,
        ).toBe("var(--chart-1)");
        fireEvent.keyDown(screen.getByRole("application"), {
            key: "ArrowRight",
        });
        await waitFor(() =>
            expect(
                container.querySelector(".recharts-tooltip-wrapper")
                    ?.textContent,
            ).toBe("BSeries 210"),
        );
        expect(container.querySelector("img")).toBeNull();
    },
);

test("charts keep independent labels and colors when another chart changes", async () => {
    const pie = (labels: string[], surfaceId = "sales") =>
        figure(
            [{ type: "pie", labels, values: labels.map(() => 42) }],
            undefined,
            surfaceId,
        );
    const second = pie(["Other", "Shared"], "second");
    const { rerender } = render(
        <Results surfaces={[...pie(["Shared"]), ...second]} />,
    );
    const widget = await screen.findByRole("article", {
        name: "Widget second",
    });
    await waitFor(() =>
        expect(within(widget).getByText("Shared")).toBeTruthy(),
    );
    const slice = widget.querySelectorAll(".recharts-pie-sector path")[1];
    expect(slice.getAttribute("fill")).toBe("var(--chart-2)");
    fireEvent.mouseEnter(widget.querySelectorAll(".recharts-pie-sector")[1]);
    await waitFor(() =>
        expect(
            widget.querySelector(".recharts-tooltip-wrapper")?.textContent,
        ).toBe("Shared42"),
    );
    rerender(<Results surfaces={[...pie(["Changed", "Shared"]), ...second]} />);
    await screen.findByText("Changed");
    const updated = screen.getByRole("article", { name: "Widget second" });
    const updatedSlice = updated.querySelectorAll(
        ".recharts-pie-sector path",
    )[1];
    expect(updatedSlice.getAttribute("fill")).toBe("var(--chart-2)");
    fireEvent.mouseEnter(updated.querySelectorAll(".recharts-pie-sector")[1]);
    await waitFor(() =>
        expect(
            updated.querySelector(".recharts-tooltip-wrapper")?.textContent,
        ).toBe("Shared42"),
    );
    expect(
        updated.querySelector<HTMLElement>(".recharts-tooltip-wrapper [style]")
            ?.style.backgroundColor,
    ).toBe(updatedSlice.getAttribute("fill"));
});

test("explicit figure updates replace traces in one widget and retain sidebar placement", async () => {
    const surfaces = figure(
        [{ type: "scatter", mode: "lines", x: [0, 1], y: [1, 2] }],
        { title: { text: "Before" } },
    ).map((surface) => ({ ...surface, pinned: true }));
    const { container, rerender } = render(<Results surfaces={surfaces} />);
    expect(screen.queryByRole("alert")).toBeNull();
    await waitFor(() =>
        expect(container.querySelector(".recharts-line-curve")).not.toBeNull(),
    );
    const chart = screen.getByRole("application");
    chart.focus();
    const updated = figure(
        [{ type: "scatter", mode: "markers", x: [2, 7, 9], y: [3, 5, 4] }],
        { title: { text: "After" } },
    ).map((surface) => ({ ...surface, pinned: true }));
    rerender(<Results surfaces={updated} />);
    await waitFor(() =>
        expect(container.querySelectorAll(".recharts-line-dot")).toHaveLength(
            3,
        ),
    );
    expect(screen.queryByText("Before")).toBeNull();
    expect(screen.getByText("After")).toBeTruthy();
    expect(screen.getAllByRole("article")).toHaveLength(1);
    expect(screen.getByRole("application")).toBe(chart);
    expect(document.activeElement).toBe(chart);
    expect(
        screen.getByRole("complementary").contains(screen.getByRole("article")),
    ).toBe(true);
    expect(
        container
            .querySelector(".recharts-line-curve")
            ?.getAttribute("stroke-width"),
    ).toBe("0");
    rerender(<Results surfaces={[]} />);
    await waitFor(() => expect(screen.queryByText("After")).toBeNull());
    rerender(<Results surfaces={surfaces} />);
    await waitFor(() => expect(screen.getByText("Before")).toBeTruthy());
    expect(screen.getAllByRole("article")).toHaveLength(1);
    rerender(<Results surfaces={[]} />);
    await waitFor(() => expect(screen.queryByLabelText("Results")).toBeNull());
});

test("backend placement moves widgets between center and sidebar without controls or visible IDs", () => {
    const surfaces = ["first", "second"].flatMap((id) =>
        figure([{ type: "bar", x: ["A"], y: [null] }], undefined, id),
    );
    const { rerender } = render(<Results surfaces={surfaces} />);
    const second = screen.getByRole("article", { name: "Widget second" });
    expect(
        screen.getByRole("region", { name: "Center widgets" }).contains(second),
    ).toBe(true);
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.queryByText("second")).toBeNull();
    rerender(
        <Results
            surfaces={surfaces.map((surface) => ({
                ...surface,
                pinned: surface.id === "second",
            }))}
        />,
    );
    expect(
        screen
            .getByRole("complementary", { name: "Side widgets" })
            .contains(screen.getByRole("article", { name: "Widget second" })),
    ).toBe(true);
    rerender(<Results surfaces={surfaces} />);
    expect(screen.queryByRole("complementary")).toBeNull();
    expect(screen.getAllByRole("article")[0].getAttribute("aria-label")).toBe(
        "Widget first",
    );
});

test("backend-pinned overflow stays visible on the side", () => {
    const surfaces = [1, 2, 3, 4, 5].flatMap((index) =>
        figure(
            [{ type: "bar", x: ["A"], y: [null] }],
            undefined,
            `chart_${index}`,
        ),
    );
    render(
        <Results
            surfaces={surfaces.map((surface, index) => ({
                ...surface,
                pinned: index === 0,
            }))}
        />,
    );
    expect(
        screen
            .getByRole("region", { name: "Center widgets" })
            .querySelectorAll("article"),
    ).toHaveLength(4);
    expect(
        screen.getByRole("complementary").querySelectorAll("article"),
    ).toHaveLength(1);
    expect(
        screen
            .getByRole("complementary")
            .contains(screen.getByRole("article", { name: "Widget chart_1" })),
    ).toBe(true);
});

test("the sidebar renders every backend-pinned widget in surface order", () => {
    const surfaces = [1, 2, 3, 4, 5, 6].flatMap((index) =>
        figure(
            [{ type: "bar", x: ["A"], y: [null] }],
            undefined,
            `chart_${index}`,
        ),
    );
    render(
        <Results
            surfaces={surfaces.map((surface) => ({ ...surface, pinned: true }))}
        />,
    );
    expect(
        [...screen.getByRole("complementary").querySelectorAll("article")].map(
            (article) => article.getAttribute("aria-label"),
        ),
    ).toEqual([
        "Widget chart_1",
        "Widget chart_2",
        "Widget chart_3",
        "Widget chart_4",
        "Widget chart_5",
        "Widget chart_6",
    ]);
    expect(
        screen
            .getByRole("region", { name: "Center widgets" })
            .querySelectorAll("article"),
    ).toHaveLength(0);
});
