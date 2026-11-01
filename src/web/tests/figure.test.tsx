import { afterEach, expect, test } from "vitest";
import {
    cleanup,
    fireEvent,
    render,
    screen,
    waitFor,
    within,
} from "@testing-library/react";
import { Figure } from "../components/plot/figure";
import { figureProps } from "../lib/a2ui";

afterEach(cleanup);

test("area charts fill to zero with unique gradients and preserve missing values", async () => {
    const props = figureProps.parse({
        data: [
            {
                type: "area",
                name: "Revenue",
                curve: "monotone",
                x: [1, 2, 3, 4, 5],
                y: [2, 4, null, -1, 3],
            },
        ],
    });
    const { container } = render(
        <>
            <Figure {...props} />
            <Figure {...props} />
        </>,
    );
    await waitFor(() =>
        expect(container.querySelectorAll(".recharts-area-area")).toHaveLength(
            2,
        ),
    );
    const gradients = [...container.querySelectorAll("linearGradient")];
    expect(gradients).toHaveLength(2);
    expect(new Set(gradients.map((gradient) => gradient.id)).size).toBe(2);
    const fills = [...container.querySelectorAll(".recharts-area-area")];
    fills.forEach((fill, index) => {
        expect(fill.getAttribute("fill")).toBe(`url(#${gradients[index].id})`);
        expect((fill.getAttribute("d")?.match(/M/g) ?? []).length).toBe(2);
    });
    expect(container.querySelectorAll(".recharts-line")).toHaveLength(0);
});

test("area schema defaults to linear and rejects unsupported curves and mismatched coordinates", () => {
    const trace = { type: "area", x: [1, 2], y: [2, 4] };
    expect(figureProps.parse({ data: [trace] }).data[0]).toMatchObject({
        curve: "linear",
    });
    expect(
        figureProps.safeParse({ data: [{ ...trace, curve: "invalid" }] })
            .success,
    ).toBe(false);
    expect(
        figureProps.safeParse({ data: [{ ...trace, y: [2] }] }).success,
    ).toBe(false);
});

test("pie rendering preserves missing slice positions and colors", async () => {
    const props = figureProps.parse({
        data: [
            {
                type: "pie",
                labels: ["Apples", "Pears", "Oranges"],
                values: [42, null, 84],
            },
        ],
        layout: { title: { text: "Revenue" } },
    });
    const { container } = render(<Figure {...props} />);
    await screen.findByText("Revenue");
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

test("pie titles, legends and tooltips render HTML as inert text", async () => {
    const title = '<img src=x onerror="alert(1)">';
    const label = '</style><img src=x onerror="alert(1)">';
    const props = figureProps.parse({
        data: [{ type: "pie", labels: [label], values: [42] }],
        layout: { title: { text: title } },
    });
    const { container } = render(<Figure {...props} />);
    expect(screen.getByText(title)).toBeTruthy();
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
        const props = figureProps.parse({
            data: [
                { type, name, x: ["A"], y: [1234.5] },
                { type, x: ["B"], y: [10] },
            ],
        });
        const { container } = render(<Figure {...props} />);
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

test.each([
    {
        name: "zero or missing pie values",
        data: [{ type: "pie", labels: ["Zero", "Unknown"], values: [0, null] }],
        message: "No positive values to display.",
    },
    {
        name: "all-missing mixed Cartesian traces",
        data: [
            { type: "bar", x: [0, 1], y: [null, null] },
            { type: "scatter", x: [0], y: [null] },
        ],
        message: "No values to display.",
    },
])("$name report an empty plot", async ({ data, message }) => {
    const props = figureProps.parse({ data });
    render(<Figure {...props} />);
    await screen.findByText(message);
    if (data[0].type !== "pie")
        expect(screen.queryByRole("application")).toBeNull();
});

test("independent lines and markers share numeric axes without adding gaps", async () => {
    const props = figureProps.parse({
        data: [
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
        layout: { xaxis: { title: { text: "Time" } } },
    });
    const { container } = render(<Figure {...props} />);
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

test("bar and scatter traces render on shared categorical axes", async () => {
    const props = figureProps.parse({
        data: [
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
        ],
    });
    const { container } = render(<Figure {...props} />);
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
    expect(Math.abs(targetA - actual[0] - groupOffset)).toBeLessThan(1);
    expect(Math.abs(targetC - previous[1] + groupOffset)).toBeLessThan(1);
    expect(
        Math.abs((actual[1] + previous[0] - targetA - targetC) / 2),
    ).toBeLessThan(1);
    expect(screen.getByText("Actual")).toBeTruthy();
    expect(screen.getByText("Previous")).toBeTruthy();
    expect(screen.getByText("Target")).toBeTruthy();
});

test("numeric bars keep their spacing when overlaid with a line", async () => {
    const props = figureProps.parse({
        data: [
            { type: "bar", x: [0, 2, 10], y: [2, -3, 4] },
            {
                type: "scatter",
                mode: "lines",
                x: [1, 9],
                y: [1, 5],
            },
        ],
    });
    const { container } = render(<Figure {...props} />);
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

test.each([
    ["categorical duplicates", ["A", "A", "B"]],
    ["numeric duplicates", [0, 0, 10]],
    ["irregular numeric duplicates", [0, 0, 1, 100]],
    ["single-position duplicates", [2, 2, 2]],
    ["categorical control", ["A", "B", "C"]],
    ["irregular numeric control", [0, 2, 10]],
] as const)(
    "bars preserve every observation with nonzero width: %s",
    async (_, x) => {
        const props = figureProps.parse({
            data: [{ type: "bar", x, y: [2, 5, 3, 4].slice(0, x.length) }],
        });
        const { container } = render(<Figure {...props} />);
        await waitFor(() =>
            expect(
                container.querySelectorAll(".recharts-bar-rectangle path"),
            ).toHaveLength(x.length),
        );
        const bars = [
            ...container.querySelectorAll(".recharts-bar-rectangle path"),
        ];
        const widths = bars.map((bar) => Number(bar.getAttribute("width")));
        expect(widths.every((width) => width > 0 && width <= 48)).toBe(true);
        const heights = bars.map((bar) => Number(bar.getAttribute("height")));
        expect(heights[0] / heights[1]).toBeCloseTo(2 / 5);
        expect(heights[2] / heights[1]).toBeCloseTo(3 / 5);
        if (x[0] === x[1]) {
            expect(bars[0].getAttribute("x")).toBe(bars[1].getAttribute("x"));
        }
        if (x[0] === x[1] && x[1] !== x[2]) {
            expect(widths[0]).toBeLessThan(
                Number(bars[2].getAttribute("x")) -
                    Number(bars[0].getAttribute("x")),
            );
            const application = screen.getByRole("application");
            fireEvent.focus(application);
            fireEvent.keyDown(application, { key: "ArrowRight" });
            await waitFor(() =>
                expect(
                    container.querySelector(".recharts-tooltip-wrapper")
                        ?.textContent,
                ).toBe(`${x[2]}Series 13`),
            );
        }
    },
);

test("mixed tooltips visit bar-only and line-only positions", async () => {
    const props = figureProps.parse({
        data: [
            { type: "bar", name: "Actual", x: ["A", "B"], y: [2, 5] },
            {
                type: "scatter",
                name: "Target",
                mode: "lines+markers",
                x: ["A", "C"],
                y: [3, 7],
            },
        ],
    });
    const { container } = render(<Figure {...props} />);
    const application = await screen.findByRole("application");
    fireEvent.focus(application);
    const tooltip = () =>
        container.querySelector(".recharts-tooltip-wrapper")?.textContent;
    await waitFor(() => expect(tooltip()).toBe("AActual2Target3"));
    fireEvent.keyDown(application, { key: "ArrowRight" });
    await waitFor(() => expect(tooltip()).toBe("BActual5"));
    fireEvent.keyDown(application, { key: "ArrowRight" });
    await waitFor(() => expect(tooltip()).toBe("CTarget7"));
    fireEvent.keyDown(application, { key: "ArrowLeft" });
    await waitFor(() => expect(tooltip()).toBe("BActual5"));
});

test.each(["mixed", "duplicates"] as const)(
    "dense grouped %s stay within distinct x spacing",
    async (kind) => {
        const x = Array.from({ length: 10 }, (_, index) =>
            kind === "duplicates" ? Math.max(0, index - 1) : index,
        );
        const data: unknown[] = Array.from(
            { length: kind === "mixed" ? 19 : 20 },
            () => ({ type: "bar", x, y: x.map(() => 1) }),
        );
        if (kind === "mixed") {
            data.push({ type: "scatter", x: [0, 9], y: [1, 2] });
        }
        const props = figureProps.parse({ data });
        const { container } = render(<Figure {...props} />);
        await waitFor(() =>
            expect(
                container.querySelectorAll(".recharts-bar-rectangle path"),
            ).toHaveLength(kind === "mixed" ? 190 : 200),
        );
        const series = [...container.querySelectorAll(".recharts-bar")].map(
            (series) => [
                ...series.querySelectorAll(".recharts-bar-rectangle path"),
            ],
        );
        const groups = x.flatMap((value, point) => {
            if (point > 0 && value === x[point - 1]) return [];
            const bars = series.map((bars) => bars[point]);
            return [
                {
                    start: Math.min(
                        ...bars.map((bar) => Number(bar.getAttribute("x"))),
                    ),
                    end: Math.max(
                        ...bars.map(
                            (bar) =>
                                Number(bar.getAttribute("x")) +
                                Number(bar.getAttribute("width")),
                        ),
                    ),
                },
            ];
        });
        for (let index = 1; index < groups.length; index++) {
            expect(groups[index - 1].end).toBeLessThan(groups[index].start);
        }
        const step = groups[1].start - groups[0].start;
        for (const group of groups) {
            expect(group.end - group.start).toBeCloseTo(step * 0.8, 3);
        }
        for (const bars of series) {
            expect(
                bars.every((bar) => {
                    const width = Number(bar.getAttribute("width"));
                    return Number.isFinite(width) && width > 0;
                }),
            ).toBe(true);
        }
    },
);

test("mixed numeric interaction preserves ordered duplicate line points, gaps, and irregular spacing", async () => {
    const props = figureProps.parse({
        data: [
            { type: "bar", name: "Actual", x: [0, 2], y: [2, 5] },
            {
                type: "scatter",
                name: "Target",
                mode: "lines+markers",
                x: [10, 0, 10, 5, 7, 9],
                y: [3, 1, 4, null, 5, 2],
            },
        ],
    });
    const { container } = render(<Figure {...props} />);
    await waitFor(() =>
        expect(container.querySelectorAll(".recharts-line-dot")).toHaveLength(
            5,
        ),
    );
    const dots = [...container.querySelectorAll(".recharts-line-dot")];
    const xs = dots.map((dot) => Number(dot.getAttribute("cx")));
    expect(xs[0]).toBe(xs[2]);
    expect(xs[0]).toBeGreaterThan(xs[1]);
    expect(dots[0].getAttribute("cy")).not.toBe(dots[2].getAttribute("cy"));
    expect(
        container
            .querySelector(".recharts-line-curve")
            ?.getAttribute("d")
            ?.match(/M/g),
    ).toHaveLength(2);
    expect((xs[3] - xs[1]) / (xs[0] - xs[1])).toBeCloseTo(0.7);
    const application = screen.getByRole("application");
    fireEvent.focus(application);
    const tooltip = () =>
        container.querySelector(".recharts-tooltip-wrapper")?.textContent;
    await waitFor(() => expect(tooltip()).toBe("0Actual2Target1"));
    fireEvent.keyDown(application, { key: "ArrowRight" });
    await waitFor(() => expect(tooltip()).toBe("2Actual5"));
    fireEvent.keyDown(application, { key: "ArrowRight" });
    fireEvent.keyDown(application, { key: "ArrowRight" });
    await waitFor(() => expect(tooltip()).toBe("7Target5"));
    fireEvent.keyDown(application, { key: "ArrowRight" });
    await waitFor(() => expect(tooltip()).toBe("9Target2"));
    fireEvent.keyDown(application, { key: "ArrowRight" });
    await waitFor(() => expect(tooltip()).toBe("10Target3"));
});

test("compact figures remove legend and axis-title space without changing the full layout", () => {
    const props = figureProps.parse({
        data: [{ type: "bar", name: "Population", x: [2000, 2050], y: [1, 2] }],
        layout: {
            title: { text: "Population trend" },
            xaxis: { title: { text: "Year" } },
            showlegend: true,
        },
    });
    const { container, rerender } = render(<Figure {...props} compact />);
    expect(screen.getByText("Population trend")).toBeTruthy();
    expect(screen.queryByText("Year")).toBeNull();
    expect(container.querySelector(".recharts-legend-wrapper")).toBeNull();

    rerender(<Figure {...props} />);
    expect(screen.getByText("Year")).toBeTruthy();
    expect(container.querySelector(".recharts-legend-wrapper")).not.toBeNull();
});

test("categorical coordinates keep first-seen order and repeated positions across traces", async () => {
    const props = figureProps.parse({
        data: [
            { type: "bar", x: ["Beta", "Alpha"], y: [1, 2] },
            {
                type: "scatter",
                mode: "lines+markers",
                x: ["__proto__", "Beta", "__proto__", "Alpha"],
                y: [3, 4, 5, 6],
            },
        ],
    });
    const { container } = render(<Figure {...props} />);
    await waitFor(() =>
        expect(container.querySelectorAll(".recharts-line-dot")).toHaveLength(
            4,
        ),
    );
    const positions = [...container.querySelectorAll(".recharts-line-dot")].map(
        (dot) => Number(dot.getAttribute("cx")),
    );
    expect(positions[0]).toBe(positions[2]);
    expect(positions[0]).toBeGreaterThan(positions[3]);
    expect(positions[3]).toBeGreaterThan(positions[1]);
});

test.each(["bar", "scatter"] as const)(
    "%s-first plots retain a later zero-valued series when the first is all missing",
    async (type) => {
        // Arrange
        const props = figureProps.parse({
            data: [
                { type, x: [0], y: [null] },
                {
                    type: type === "bar" ? "scatter" : "bar",
                    name: "Zero series",
                    x: [0],
                    y: [0],
                },
            ],
        });
        // Act
        const { container } = render(<Figure {...props} />);
        fireEvent.focus(await screen.findByRole("application"));

        // Assert
        await waitFor(() =>
            expect(
                container.querySelector(".recharts-tooltip-wrapper")
                    ?.textContent,
            ).toBe("0Zero series0"),
        );
        expect(screen.queryByText("No values to display.")).toBeNull();
    },
);
