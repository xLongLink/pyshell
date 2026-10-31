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
    return [{ id: surfaceId, props: figureProps.parse({ data, layout }) }];
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
    const { rerender } = render(
        <Results
            surfaces={surfaces}
            placements={{ first: { pinned: false } }}
        />,
    );
    const chart = await screen.findByRole("application");
    chart.focus();
    rerender(
        <Results
            surfaces={[...surfaces, ...bar("Second", "second")]}
            placements={{ first: { pinned: false }, second: { pinned: false } }}
        />,
    );
    await waitFor(() => expect(screen.getByText("Second")).toBeTruthy());
    expect(document.activeElement).toBe(chart);
    expect(
        screen
            .getByRole("article", { name: "Widget first" })
            .querySelector('[role="application"]'),
    ).toBe(chart);
});

test("replacement snapshots replace surfaces even when their count is unchanged", async () => {
    const { rerender } = render(
        <Results
            surfaces={bar("Old", "old")}
            placements={{ old: { pinned: false } }}
        />,
    );
    await waitFor(() => expect(screen.getByText("Old")).toBeTruthy());
    rerender(
        <Results
            surfaces={bar("New", "new")}
            placements={{ new: { pinned: false } }}
        />,
    );
    await waitFor(() => expect(screen.getByText("New")).toBeTruthy());
    expect(screen.queryByText("Old")).toBeNull();
    expect(screen.getAllByRole("article")).toHaveLength(1);
});

test("replacement, removal, and reset do not duplicate result surfaces", async () => {
    const surfaces = bar("Before");
    const { rerender } = render(
        <Results
            surfaces={surfaces}
            placements={{ sales: { pinned: false } }}
        />,
    );
    await waitFor(() => expect(screen.getByText("Before")).toBeTruthy());
    rerender(
        <Results
            surfaces={bar("After")}
            placements={{ sales: { pinned: false } }}
        />,
    );
    await waitFor(() => expect(screen.getByText("After")).toBeTruthy());
    expect(screen.queryByText("Before")).toBeNull();
    rerender(<Results surfaces={[]} placements={{}} />);
    await waitFor(() => expect(screen.queryByText("After")).toBeNull());
    rerender(
        <Results
            surfaces={surfaces}
            placements={{ sales: { pinned: false } }}
        />,
    );
    await waitFor(() => expect(screen.getByText("Before")).toBeTruthy());
    rerender(<Results surfaces={[]} placements={{}} />);
    await waitFor(() => expect(screen.queryByLabelText("Results")).toBeNull());
});

test("chart titles remain inert even when they contain HTML", async () => {
    const text = '<img src=x onerror="alert(1)">';
    const { container } = render(
        <Results
            surfaces={bar(text)}
            placements={{ sales: { pinned: false } }}
        />,
    );
    await waitFor(() => expect(screen.getByText(text)).toBeTruthy());
    expect(container.querySelector("img")).toBeNull();
});

test("pie rendering preserves missing slice positions and colors", async () => {
    const { container } = render(
        <Results
            placements={{ sales: { pinned: false } }}
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
            placements={{ sales: { pinned: false } }}
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
                placements={{ sales: { pinned: false } }}
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
        <Results
            surfaces={[...pie(["Shared"]), ...second]}
            placements={{ sales: { pinned: false }, second: { pinned: false } }}
        />,
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
    rerender(
        <Results
            surfaces={[...pie(["Changed", "Shared"]), ...second]}
            placements={{ sales: { pinned: false }, second: { pinned: false } }}
        />,
    );
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

test("a pie with zero or missing values reports an empty plot", async () => {
    render(
        <Results
            placements={{ sales: { pinned: false } }}
            surfaces={figure([
                {
                    type: "pie",
                    labels: ["Zero", "Unknown"],
                    values: [0, null],
                },
            ])}
        />,
    );
    await waitFor(() =>
        expect(screen.getByText("No positive values to display.")).toBeTruthy(),
    );
});

test("independent lines and markers share numeric axes without adding gaps", async () => {
    const { container } = render(
        <Results
            placements={{ sales: { pinned: false } }}
            surfaces={figure(
                [
                    {
                        type: "scatter",
                        name: "Predicted",
                        mode: "lines",
                        x: [0, 5, 10],
                        y: [1, 4, 8],
                    },
                    {
                        type: "scatter",
                        name: "Observed",
                        mode: "markers",
                        x: [2, 7],
                        y: [3, 5],
                    },
                ],
                { xaxis: { title: { text: "Time" } } },
            )}
        />,
    );
    await waitFor(() =>
        expect(container.querySelectorAll(".recharts-line-dot")).toHaveLength(
            2,
        ),
    );
    const lines = container.querySelectorAll(".recharts-line");
    const curve = lines[0].querySelector(".recharts-line-curve");
    expect(curve?.getAttribute("d")).not.toMatch(/[CQ]/);
    expect(lines[0].querySelector(".recharts-line-dot")).toBeNull();
    expect(curve?.getAttribute("d")?.match(/M/g)).toHaveLength(1);
    expect(
        lines[1]
            .querySelector(".recharts-line-curve")
            ?.getAttribute("stroke-width"),
    ).toBe("0");
    const dots = container.querySelectorAll(".recharts-line-dot");
    const coordinates = curve!
        .getAttribute("d")!
        .match(/-?\d+(?:\.\d+)?/g)!
        .map(Number);
    const start = coordinates[0];
    const end = coordinates.at(-2)!;
    expect(Number(dots[0].getAttribute("cx"))).toBeCloseTo(
        start + (end - start) * 0.2,
        1,
    );
    expect(Number(dots[1].getAttribute("cx"))).toBeCloseTo(
        start + (end - start) * 0.7,
        1,
    );
    expect(screen.getByText("Predicted")).toBeTruthy();
    expect(screen.getByText("Observed")).toBeTruthy();
    expect(screen.getByText("Time")).toBeTruthy();
    const application = screen.getByRole("application");
    fireEvent.focus(application);
    fireEvent.keyDown(application, { key: "ArrowRight" });
    fireEvent.keyDown(application, { key: "ArrowRight" });
    fireEvent.keyDown(application, { key: "ArrowRight" });
    await waitFor(() =>
        expect(
            container.querySelector(".recharts-tooltip-wrapper")?.textContent,
        ).toBe("2Observed3"),
    );
    expect(
        container.querySelector<HTMLElement>(
            ".recharts-tooltip-wrapper [style]",
        )?.style.backgroundColor,
    ).toBe("var(--chart-2)");
});

test("lines+markers preserves input order, repeated x values, and missing gaps", async () => {
    const { container } = render(
        <Results
            placements={{ sales: { pinned: false } }}
            surfaces={figure(
                [
                    {
                        type: "scatter",
                        name: "Hidden legend",
                        mode: "lines+markers",
                        x: [2, 0, 2, 3, 4, 5],
                        y: [3, 1, 4, null, 5, 2],
                    },
                ],
                { showlegend: false },
            )}
        />,
    );
    await waitFor(() =>
        expect(container.querySelectorAll(".recharts-line-dot")).toHaveLength(
            5,
        ),
    );
    const dots = container.querySelectorAll(".recharts-line-dot");
    expect(Number(dots[0].getAttribute("cx"))).toBeGreaterThan(
        Number(dots[1].getAttribute("cx")),
    );
    expect(Number(dots[0].getAttribute("cx"))).toBeCloseTo(
        Number(dots[2].getAttribute("cx")),
        1,
    );
    expect(Number(dots[0].getAttribute("cy"))).not.toBe(
        Number(dots[2].getAttribute("cy")),
    );
    expect(
        container
            .querySelector(".recharts-line-curve")
            ?.getAttribute("d")
            ?.match(/M/g),
    ).toHaveLength(2);
    expect(screen.queryByText("Hidden legend")).toBeNull();
});

test("bar and scatter traces render on shared categorical axes", async () => {
    const { container } = render(
        <Results
            placements={{ sales: { pinned: false } }}
            surfaces={figure([
                {
                    type: "bar",
                    name: "Actual",
                    x: ["A", "B"],
                    y: [4, -2],
                },
                {
                    type: "bar",
                    name: "Previous",
                    x: ["B", "C"],
                    y: [3, 5],
                },
                {
                    type: "scatter",
                    name: "Target",
                    mode: "lines+markers",
                    x: ["A", "C"],
                    y: [6, 7],
                },
            ])}
        />,
    );
    await waitFor(() =>
        expect(
            container.querySelectorAll(".recharts-bar-rectangle"),
        ).toHaveLength(4),
    );
    expect(container.querySelectorAll(".recharts-line-dot")).toHaveLength(2);
    const [actual, previous] = Array.from(
        container.querySelectorAll(".recharts-bar"),
        (series) =>
            Array.from(
                series.querySelectorAll(".recharts-bar-rectangle path"),
                (bar) =>
                    Number(bar.getAttribute("x")) +
                    Number(bar.getAttribute("width")) / 2,
            ),
    );
    expect(actual).toHaveLength(2);
    expect(previous).toHaveLength(2);
    const [targetA, targetC] = Array.from(
        container.querySelectorAll(".recharts-line-dot"),
        (dot) => Number(dot.getAttribute("cx")),
    );
    const groupOffset = (previous[0] - actual[1]) / 2;
    expect(groupOffset).toBeGreaterThan(0);
    expect(targetA).toBeCloseTo(actual[0] + groupOffset, 0);
    expect(targetC).toBeCloseTo(previous[1] - groupOffset, 0);
    expect((actual[1] + previous[0]) / 2).toBeCloseTo(
        (targetA + targetC) / 2,
        0,
    );
    expect(screen.getByText("Actual")).toBeTruthy();
    expect(screen.getByText("Previous")).toBeTruthy();
    expect(screen.getByText("Target")).toBeTruthy();
});

test("all-missing mixed Cartesian traces report an empty plot", async () => {
    render(
        <Results
            placements={{ sales: { pinned: false } }}
            surfaces={figure([
                { type: "bar", x: [0, 1], y: [null, null] },
                { type: "scatter", x: [0], y: [null] },
            ])}
        />,
    );
    await waitFor(() =>
        expect(screen.getByText("No values to display.")).toBeTruthy(),
    );
    expect(screen.queryByRole("application")).toBeNull();
});

test("numeric bars keep their spacing when overlaid with a line", async () => {
    const { container } = render(
        <Results
            placements={{ sales: { pinned: false } }}
            surfaces={figure([
                { type: "bar", x: [0, 2, 10], y: [2, -3, 4] },
                {
                    type: "scatter",
                    mode: "lines",
                    x: [1, 9],
                    y: [1, 5],
                },
            ])}
        />,
    );
    await waitFor(() =>
        expect(
            container.querySelectorAll(".recharts-bar-rectangle path"),
        ).toHaveLength(3),
    );
    const bars = [
        ...container.querySelectorAll(".recharts-bar-rectangle path"),
    ];
    const xs = bars.map((bar) => Number(bar.getAttribute("x")));
    expect((xs[1] - xs[0]) / (xs[2] - xs[0])).toBeCloseTo(0.2);
    expect(
        bars.every((bar) => Math.abs(Number(bar.getAttribute("height"))) > 0),
    ).toBe(true);
    expect(container.querySelector(".recharts-line-curve")).not.toBeNull();
});

test("explicit figure updates replace traces in one widget and retain sidebar placement", async () => {
    const surfaces = figure(
        [{ type: "scatter", mode: "lines", x: [0, 1], y: [1, 2] }],
        { title: { text: "Before" } },
    );
    const placements = { sales: { pinned: true } };
    const { container, rerender } = render(
        <Results surfaces={surfaces} placements={placements} />,
    );
    expect(screen.queryByRole("alert")).toBeNull();
    await waitFor(() =>
        expect(container.querySelector(".recharts-line-curve")).not.toBeNull(),
    );
    const chart = screen.getByRole("application");
    chart.focus();
    const updated = figure(
        [{ type: "scatter", mode: "markers", x: [2, 7, 9], y: [3, 5, 4] }],
        { title: { text: "After" } },
    );
    rerender(<Results surfaces={updated} placements={placements} />);
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
});

test("backend placement moves widgets between center and sidebar without controls or visible IDs", () => {
    const surfaces = [...bar("", "first"), ...bar("", "second")];
    const { rerender } = render(
        <Results
            surfaces={surfaces}
            placements={{ first: { pinned: false }, second: { pinned: false } }}
        />,
    );
    const second = screen.getByRole("article", { name: "Widget second" });
    expect(
        screen.getByRole("region", { name: "Center widgets" }).contains(second),
    ).toBe(true);
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.queryByText("second")).toBeNull();
    rerender(
        <Results
            surfaces={surfaces}
            placements={{
                first: { pinned: false },
                second: { pinned: true },
            }}
        />,
    );
    expect(
        screen
            .getByRole("complementary", { name: "Side widgets" })
            .contains(screen.getByRole("article", { name: "Widget second" })),
    ).toBe(true);
    rerender(
        <Results
            surfaces={surfaces}
            placements={{
                first: { pinned: false },
                second: { pinned: false },
            }}
        />,
    );
    expect(screen.queryByRole("complementary")).toBeNull();
    expect(screen.getAllByRole("article")[0].getAttribute("aria-label")).toBe(
        "Widget first",
    );
});

test("backend-pinned overflow stays visible on the side", () => {
    const surfaces = [1, 2, 3, 4, 5].flatMap((index) =>
        bar("", `chart_${index}`),
    );
    const placements = Object.fromEntries(
        surfaces.map(({ id }, index) => [id, { pinned: index === 0 }]),
    );
    render(<Results surfaces={surfaces} placements={placements} />);
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
        bar("", `chart_${index}`),
    );
    const placements = Object.fromEntries(
        surfaces.map(({ id }) => [id, { pinned: true }]),
    );
    render(<Results surfaces={surfaces} placements={placements} />);
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
