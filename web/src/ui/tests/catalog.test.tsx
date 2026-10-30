import { afterEach, beforeEach, expect, test, vi } from "vitest";
import {
    cleanup,
    fireEvent,
    render,
    screen,
    waitFor,
    within,
} from "@testing-library/react";
import { Results } from "../components/results";
import { ChartContainer } from "../components/ui/chart";
import { a2uiBatchSchema, catalogId } from "../lib/a2ui";

function mockChartWidth(width: number) {
    vi.stubGlobal(
        "ResizeObserver",
        class {
            constructor(private callback: ResizeObserverCallback) {}
            observe(target: Element) {
                this.callback(
                    [
                        {
                            target,
                            contentRect: { width, height: 300 },
                        } as ResizeObserverEntry,
                    ],
                    this as unknown as ResizeObserver,
                );
            }
            unobserve() {}
            disconnect() {}
        },
    );
}

beforeEach(() => mockChartWidth(600));

afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
});

function surface(components: unknown[], surfaceId = "sales") {
    return a2uiBatchSchema.parse([
        { version: "v0.9", createSurface: { surfaceId, catalogId } },
        { version: "v0.9", updateComponents: { surfaceId, components } },
    ]);
}

function bar(title: string) {
    return {
        id: "root",
        component: "Figure",
        layout: { title: { text: title } },
        data: [{ type: "bar", x: ["A"], y: [42] }],
    };
}

test("appending another surface preserves an existing chart and its keyboard focus", async () => {
    const messages = surface([bar("First")], "first");
    const { rerender } = render(<Results messages={messages} />);
    const chart = await screen.findByRole("application");
    chart.focus();
    rerender(
        <Results
            messages={[...messages, ...surface([bar("Second")], "second")]}
        />,
    );
    await waitFor(() => expect(screen.getByText("Second")).toBeTruthy());
    expect(chart.isConnected).toBe(true);
    expect(document.activeElement).toBe(chart);
    expect(
        screen
            .getByRole("article", { name: "Widget first" })
            .querySelector('[role="application"]'),
    ).toBe(chart);
});

test("replacement snapshots rebuild surfaces even when their history has the same length", async () => {
    const { rerender } = render(
        <Results messages={surface([bar("Old")], "old")} />,
    );
    await waitFor(() => expect(screen.getByText("Old")).toBeTruthy());
    rerender(<Results messages={surface([bar("New")], "new")} />);
    await waitFor(() => expect(screen.getByText("New")).toBeTruthy());
    expect(screen.queryByText("Old")).toBeNull();
    expect(screen.getAllByRole("article")).toHaveLength(1);
});

test.each(["duplicate", "unknown update", "unknown delete", "capacity"])(
    "invalid appended lifecycle messages fail closed: %s",
    async (invalid) => {
        const messages = surface([bar("Valid")]);
        const { rerender } = render(<Results messages={messages} />);
        await waitFor(() => expect(screen.getByText("Valid")).toBeTruthy());
        const addition =
            invalid === "duplicate"
                ? [messages[0]]
                : invalid === "unknown update"
                  ? surface([bar("Unknown")], "unknown").slice(1)
                  : invalid === "unknown delete"
                    ? a2uiBatchSchema.parse([
                          {
                              version: "v0.9",
                              deleteSurface: { surfaceId: "unknown" },
                          },
                      ])
                    : Array.from({ length: 20 }, (_, index) =>
                          surface([bar(`Extra ${index}`)], `extra_${index}`),
                      ).flat();
        rerender(<Results messages={[...messages, ...addition]} />);
        await waitFor(() => expect(screen.getByRole("alert")).toBeTruthy());
        expect(screen.queryByRole("article")).toBeNull();
        rerender(<Results messages={surface([bar("Recovered")])} />);
        await waitFor(() => expect(screen.getByText("Recovered")).toBeTruthy());
        expect(screen.queryByRole("alert")).toBeNull();
    },
);

test.each(["bar", "pie"])(
    "%s charts use keyboard-only focus styling without disabling focus",
    async (type) => {
        render(
            <Results
                messages={surface([
                    {
                        id: "root",
                        component: "Figure",
                        data: [
                            type === "pie"
                                ? { type, labels: ["A", "B"], values: [42, 10] }
                                : { type, x: ["A", "B"], y: [42, 10] },
                        ],
                    },
                ])}
            />,
        );
        const chart = await screen.findByRole("application");
        const wrapper = chart.closest('[data-slot="chart"]');
        expect(wrapper?.classList.contains("focus-within:ring-2")).toBe(false);
        expect(wrapper?.classList.contains("has-[:focus-visible]:ring-2")).toBe(
            true,
        );
        expect(
            wrapper?.classList.contains("has-[:focus-visible]:ring-ring"),
        ).toBe(true);
        expect(chart.getAttribute("tabindex")).toBe("0");
        fireEvent.click(chart);
        chart.focus();
        expect(document.activeElement).toBe(chart);
    },
);

test.each([1, 2, 3, 4])(
    "%i center widgets use the viewport layout",
    async (count) => {
        const messages = Array.from({ length: count }, (_, index) =>
            surface([bar(`Chart ${index}`)], `chart_${index}`),
        ).flat();
        const { container } = render(<Results messages={messages} />);
        await waitFor(() =>
            expect(screen.getByText(`Chart ${count - 1}`)).toBeTruthy(),
        );
        const center = screen.getByRole("region", { name: "Center widgets" });
        expect(center.className).toContain("md:h-[calc(100dvh-4rem)]");
        expect(center.classList.contains("md:grid-cols-2")).toBe(count >= 3);
        expect(center.classList.contains("md:grid-rows-2")).toBe(count > 1);
        const widgets = center.querySelectorAll("article");
        expect(widgets).toHaveLength(count);
        expect(widgets[count - 1].classList.contains("md:col-span-2")).toBe(
            count === 3,
        );
        expect(
            container.querySelector("figure")?.classList.contains("h-full"),
        ).toBe(true);
        expect(
            container
                .querySelector('[data-slot="chart"]')
                ?.classList.contains("h-full"),
        ).toBe(true);
    },
);

test("replacement, removal, and reset do not duplicate result surfaces", async () => {
    const messages = surface([bar("Before")]);
    const { rerender } = render(<Results messages={messages} />);
    await waitFor(() => expect(screen.getByText("Before")).toBeTruthy());
    const replaced = a2uiBatchSchema.parse([
        ...messages,
        { version: "v0.9", deleteSurface: { surfaceId: "sales" } },
        ...surface([bar("After")]),
    ]);
    rerender(<Results messages={replaced} />);
    await waitFor(() => expect(screen.getByText("After")).toBeTruthy());
    expect(screen.queryByText("Before")).toBeNull();
    expect(screen.getAllByText("After")).toHaveLength(1);
    rerender(
        <Results
            messages={[
                ...replaced,
                { version: "v0.9", deleteSurface: { surfaceId: "sales" } },
            ]}
        />,
    );
    await waitFor(() => expect(screen.queryByText("After")).toBeNull());
    rerender(<Results messages={messages} />);
    await waitFor(() => expect(screen.getByText("Before")).toBeTruthy());
    rerender(<Results messages={[]} />);
    await waitFor(() => expect(screen.queryByLabelText("Results")).toBeNull());
});

test("chart titles remain inert even when they contain HTML", async () => {
    const text = '<img src=x onerror="alert(1)">';
    const { container } = render(<Results messages={surface([bar(text)])} />);
    await waitFor(() => expect(screen.getByText(text)).toBeTruthy());
    expect(container.querySelector("img")).toBeNull();
});

test.each(["bar", "scatter", "pie"] as const)(
    "%s renders its plot without a data table while preserving missing values",
    async (type) => {
        const { container } = render(
            <Results
                messages={surface([
                    {
                        id: "root",
                        component: "Figure",
                        layout: { title: { text: "Revenue" } },
                        data: [
                            type === "pie"
                                ? {
                                      type,
                                      labels: ["Apples", "Pears", "Oranges"],
                                      values: [42, null, 84],
                                  }
                                : {
                                      type,
                                      x: [
                                          "Apples",
                                          "Pears",
                                          "Oranges",
                                          "Bananas",
                                      ],
                                      y: [42, null, 84, 50],
                                  },
                        ],
                    },
                ])}
            />,
        );
        await waitFor(() => expect(screen.getByText("Revenue")).toBeTruthy());
        expect(screen.queryByText("View chart data")).toBeNull();
        expect(screen.queryByRole("table")).toBeNull();
        const plot = {
            bar: ".recharts-bar",
            scatter: ".recharts-line",
            pie: ".recharts-pie",
        }[type];
        await waitFor(() =>
            expect(container.querySelector(plot!)).not.toBeNull(),
        );
        expect(container.querySelector('[data-slot="chart"]')).not.toBeNull();
        expect(screen.getByRole("application").getAttribute("tabindex")).toBe(
            "0",
        );
        if (type === "scatter") {
            const path = container
                .querySelector(".recharts-line-curve")
                ?.getAttribute("d");
            expect(path?.match(/M/g)).toHaveLength(2);
        }
        if (type === "pie") {
            const legend = within(
                container.querySelector<HTMLElement>(
                    ".recharts-legend-wrapper",
                )!,
            );
            expect(legend.getByText("Apples")).toBeTruthy();
            expect(legend.getByText("Oranges")).toBeTruthy();
            expect(legend.queryByText("Pears")).toBeNull();
            expect(
                container.querySelectorAll(".recharts-pie-sector"),
            ).toHaveLength(2);
        }
    },
);

test.each(['</style><img src=x onerror="alert(1)">', "__proto__", "slice_1"])(
    "pie legends and tooltips resolve slice labels as inert text: %s",
    async (label) => {
        const { container } = render(
            <Results
                messages={surface([
                    {
                        id: "root",
                        component: "Figure",
                        data: [{ type: "pie", labels: [label], values: [42] }],
                    },
                ])}
            />,
        );
        await waitFor(() => expect(screen.getAllByText(label)).toHaveLength(1));
        expect(container.querySelector("img")).toBeNull();
        const chart = container.querySelector<HTMLElement>(
            '[data-slot="chart"]',
        )!;
        expect(chart.style.getPropertyValue("--color-slice_0")).not.toBe("");
        expect(chart.getAttribute("style")).not.toContain(label);
        expect(chart.querySelector("style")).toBeNull();
        fireEvent.mouseEnter(container.querySelector(".recharts-pie-sector")!);
        await waitFor(() =>
            expect(
                container.querySelector(".recharts-tooltip-wrapper")
                    ?.textContent,
            ).toBe(`${label}42`),
        );
        expect(
            container
                .querySelector<HTMLElement>(".recharts-tooltip-wrapper [style]")
                ?.style.getPropertyValue("--color-bg"),
        ).toBe("var(--color-slice_0)");
        expect(container.querySelector("img")).toBeNull();
    },
);

test.each(["bar", "scatter"] as const)(
    "%s legends and keyboard tooltips preserve user labels and default series names",
    async (type) => {
        const name = '<img src=x onerror="alert(1)">__proto__';
        const { container } = render(
            <Results
                messages={surface([
                    {
                        id: "root",
                        component: "Figure",
                        data: [
                            { type, name, x: ["A"], y: [42] },
                            { type, x: ["B"], y: [10] },
                        ],
                    },
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
            ).toBe(`A${name}42`),
        );
        expect(
            container
                .querySelector<HTMLElement>(".recharts-tooltip-wrapper [style]")
                ?.style.getPropertyValue("--color-bg"),
        ).toBe("var(--color-trace_0)");
        expect(container.querySelector("img")).toBeNull();
    },
);

test("chart colors stay scoped to their containers and preserve supplied styles", () => {
    const { container, rerender } = render(
        <>
            <ChartContainer
                config={{ series: { color: "red" } }}
                style={{ opacity: 0.5 }}
            >
                <svg />
            </ChartContainer>
            <ChartContainer config={{ series: { color: "blue" } }}>
                <svg />
            </ChartContainer>
        </>,
    );
    const charts = container.querySelectorAll<HTMLElement>(
        '[data-slot="chart"]',
    );
    expect(charts[0].style.getPropertyValue("--color-series")).toBe("red");
    expect(charts[1].style.getPropertyValue("--color-series")).toBe("blue");
    expect(charts[0].style.opacity).toBe("0.5");
    rerender(
        <ChartContainer config={{}}>
            <svg />
        </ChartContainer>,
    );
    expect(
        container
            .querySelector<HTMLElement>('[data-slot="chart"]')!
            .style.getPropertyValue("--color-series"),
    ).toBe("");
});

test("a pie with zero or missing values reports an empty plot", async () => {
    render(
        <Results
            messages={surface([
                {
                    id: "root",
                    component: "Figure",
                    data: [
                        {
                            type: "pie",
                            labels: ["Zero", "Unknown"],
                            values: [0, null],
                        },
                    ],
                },
            ])}
        />,
    );
    await waitFor(() =>
        expect(screen.getByText("No positive values to display.")).toBeTruthy(),
    );
    expect(screen.queryByRole("table")).toBeNull();
});

test("independent lines and markers share numeric axes without adding gaps", async () => {
    const { container } = render(
        <Results
            messages={surface([
                {
                    id: "root",
                    component: "Figure",
                    layout: {
                        xaxis: { title: { text: "Time" } },
                    },
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
                },
            ])}
        />,
    );
    await waitFor(() =>
        expect(container.querySelectorAll(".recharts-line-dot")).toHaveLength(
            2,
        ),
    );
    const lines = container.querySelectorAll(".recharts-line");
    const curve = lines[0].querySelector(".recharts-line-curve");
    expect(curve?.getAttribute("stroke-width")).toBe("2");
    expect(curve?.getAttribute("stroke-linecap")).toBe("round");
    expect(curve?.getAttribute("d")).not.toMatch(/[CQ]/);
    expect(lines[0].querySelector(".recharts-line-dot")).toBeNull();
    expect(curve?.getAttribute("d")?.match(/M/g)).toHaveLength(1);
    expect(
        lines[1]
            .querySelector(".recharts-line-curve")
            ?.getAttribute("stroke-width"),
    ).toBe("0");
    const dots = container.querySelectorAll(".recharts-line-dot");
    expect(dots).toHaveLength(2);
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
        ).toContain("Observed"),
    );
    expect(
        container.querySelector(".recharts-tooltip-wrapper")?.textContent,
    ).not.toContain("Predicted");
    expect(
        container.querySelector(".recharts-tooltip-wrapper")?.textContent,
    ).toBe("2Observed3");
    expect(
        container
            .querySelector<HTMLElement>(".recharts-tooltip-wrapper [style]")
            ?.style.getPropertyValue("--color-bg"),
    ).toBe("var(--color-trace_1)");
});

test("lines+markers preserves input order, repeated x values, and missing gaps", async () => {
    const { container } = render(
        <Results
            messages={surface([
                {
                    id: "root",
                    component: "Figure",
                    layout: { showlegend: false },
                    data: [
                        {
                            type: "scatter",
                            name: "Hidden legend",
                            mode: "lines+markers",
                            x: [2, 0, 2, 3, 4],
                            y: [3, 1, null, 5, 2],
                        },
                    ],
                },
            ])}
        />,
    );
    await waitFor(() =>
        expect(container.querySelectorAll(".recharts-line-dot")).toHaveLength(
            4,
        ),
    );
    const dots = container.querySelectorAll(".recharts-line-dot");
    expect(Number(dots[0].getAttribute("cx"))).toBeGreaterThan(
        Number(dots[1].getAttribute("cx")),
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
            messages={surface([
                {
                    id: "root",
                    component: "Figure",
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
    expect(screen.getByText("Actual")).toBeTruthy();
    expect(screen.getByText("Previous")).toBeTruthy();
    expect(screen.getByText("Target")).toBeTruthy();
});

test("all-missing Cartesian traces report an empty plot", async () => {
    render(
        <Results
            messages={surface([
                {
                    id: "root",
                    component: "Figure",
                    data: [{ type: "scatter", x: [0, 1], y: [null, null] }],
                },
            ])}
        />,
    );
    await waitFor(() =>
        expect(screen.getByText("No values to display.")).toBeTruthy(),
    );
});

test("numeric bars keep their spacing when overlaid with a line", async () => {
    const { container } = render(
        <Results
            messages={surface([
                {
                    id: "root",
                    component: "Figure",
                    data: [
                        { type: "bar", x: [0, 2, 10], y: [2, -3, 4] },
                        {
                            type: "scatter",
                            mode: "lines",
                            x: [1, 9],
                            y: [1, 5],
                        },
                    ],
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

test("explicit figure updates replace rendered traces without creating another widget", async () => {
    const messages = surface([
        {
            id: "root",
            component: "Figure",
            layout: { title: { text: "Before" } },
            data: [{ type: "scatter", mode: "lines", x: [0, 1], y: [1, 2] }],
        },
    ]);
    const { container, rerender } = render(<Results messages={messages} />);
    await waitFor(() =>
        expect(container.querySelector(".recharts-line-curve")).not.toBeNull(),
    );
    const update = a2uiBatchSchema.parse([
        {
            version: "v0.9",
            updateComponents: {
                surfaceId: "sales",
                components: [
                    {
                        id: "root",
                        component: "Figure",
                        layout: { title: { text: "After" } },
                        data: [
                            {
                                type: "scatter",
                                mode: "markers",
                                x: [2, 7, 9],
                                y: [3, 5, 4],
                            },
                        ],
                    },
                ],
            },
        },
    ]);
    rerender(<Results messages={[...messages, ...update]} />);
    await waitFor(() =>
        expect(container.querySelectorAll(".recharts-line-dot")).toHaveLength(
            3,
        ),
    );
    expect(screen.queryByText("Before")).toBeNull();
    expect(screen.getByText("After")).toBeTruthy();
    expect(screen.getAllByRole("article")).toHaveLength(1);
    expect(
        container
            .querySelector(".recharts-line-curve")
            ?.getAttribute("stroke-width"),
    ).toBe("0");
});

test("backend placement moves widgets between center and sidebar without controls or visible IDs", async () => {
    const messages = [
        ...surface([bar("First result")], "first"),
        ...surface([bar("Second result")], "second"),
    ];
    const { rerender } = render(<Results messages={messages} />);
    await waitFor(() => expect(screen.getByText("Second result")).toBeTruthy());
    const second = screen.getByRole("article", { name: "Widget second" });
    expect(
        screen.getByRole("region", { name: "Center widgets" }).contains(second),
    ).toBe(true);
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.queryByText("second")).toBeNull();
    rerender(
        <Results
            messages={messages}
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
            messages={messages}
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

test.each(["scatter", "bar", "pie"] as const)(
    "%s plots limit legends to two rows and hide them when pinned",
    async (type) => {
        const names = Array.from(
            { length: 20 },
            (_, index) => `Category ${index}`,
        );
        const messages = surface([
            {
                id: "root",
                component: "Figure",
                data:
                    type === "pie"
                        ? [{ type, labels: names, values: names.map(() => 1) }]
                        : names.map((name) => ({
                              type,
                              name,
                              x: [0, 1],
                              y: [1, 2],
                          })),
            },
        ]);
        const { rerender } = render(<Results messages={messages} />);
        const legendClass = "[&_.recharts-legend-wrapper]:hidden";
        await waitFor(() =>
            expect(
                screen
                    .getByRole("article")
                    .querySelector(".recharts-legend-wrapper")?.textContent,
            ).toContain("Category 19"),
        );
        expect(
            screen.getByRole("article").classList.contains(legendClass),
        ).toBe(false);
        const legend = screen
            .getByRole("article")
            .querySelector(".recharts-legend-wrapper > div")!;
        expect(legend.classList.contains("max-h-16")).toBe(true);
        expect(legend.classList.contains("overflow-hidden")).toBe(true);
        expect(legend.classList.contains("gap-y-2")).toBe(true);
        expect(legend.classList.contains("pt-4")).toBe(true);
        expect(legend.children).toHaveLength(20);
        for (const item of legend.children) {
            expect(item.classList.contains("h-5")).toBe(true);
            expect(
                item.querySelector("span")?.classList.contains("truncate"),
            ).toBe(true);
        }
        rerender(
            <Results
                messages={messages}
                placements={{ sales: { pinned: true } }}
            />,
        );
        const pinned = screen
            .getByRole("complementary")
            .querySelector("article")!;
        await waitFor(() =>
            expect(pinned.querySelector(".recharts-surface")).not.toBeNull(),
        );
        expect(pinned.classList.contains(legendClass)).toBe(true);
        rerender(
            <Results
                messages={messages}
                placements={{ sales: { pinned: false } }}
            />,
        );
        await waitFor(() =>
            expect(
                screen
                    .getByRole("article")
                    .querySelector(".recharts-legend-wrapper")?.textContent,
            ).toContain("Category 19"),
        );
        expect(
            screen.getByRole("article").classList.contains(legendClass),
        ).toBe(false);
    },
);

test.each(["scatter", "bar"] as const)(
    "pinned %s plots hide the bottom axis title but keep tick numbers",
    async (type) => {
        const messages = surface([
            {
                id: "root",
                component: "Figure",
                layout: { xaxis: { title: { text: "Submission month" } } },
                data: [{ type, x: [1, 2, 3], y: [10, 20, 30] }],
            },
        ]);
        const { container, rerender } = render(
            <Results
                messages={messages}
                placements={{ sales: { pinned: true } }}
            />,
        );
        const titleClass = "[&_.plot-xaxis-title]:hidden";
        await waitFor(() =>
            expect(
                container.querySelector(".plot-xaxis-title")?.textContent,
            ).toBe("Submission month"),
        );
        expect(screen.getByRole("article").classList.contains(titleClass)).toBe(
            true,
        );
        expect(
            container.querySelectorAll(
                ".recharts-xAxis-tick-labels .recharts-cartesian-axis-tick-value",
            ).length,
        ).toBeGreaterThan(0);
        rerender(
            <Results
                messages={messages}
                placements={{ sales: { pinned: false } }}
            />,
        );
        await waitFor(() =>
            expect(screen.getByText("Submission month")).toBeTruthy(),
        );
        expect(screen.getByRole("article").classList.contains(titleClass)).toBe(
            false,
        );
    },
);

test.each(["scatter", "bar"] as const)(
    "%s plots hide value axis numbers at every width and placement",
    async (type) => {
        mockChartWidth(320);
        const messages = surface([
            {
                id: "root",
                component: "Figure",
                data: [
                    {
                        type,
                        name: "Submissions",
                        x: ["Jan", "Feb"],
                        y: [10, 20],
                    },
                ],
            },
        ]);
        const { container, rerender } = render(
            <Results
                messages={messages}
                placements={{ sales: { pinned: true } }}
            />,
        );
        await waitFor(() =>
            expect(container.querySelector(".recharts-xAxis")).not.toBeNull(),
        );
        expect(container.querySelector(".recharts-yAxis")).toBeNull();
        expect(
            container.querySelector(".recharts-yAxis-tick-labels"),
        ).toBeNull();
        fireEvent.focus(screen.getByRole("application"));
        await waitFor(() =>
            expect(
                container.querySelector(".recharts-tooltip-wrapper")
                    ?.textContent,
            ).toContain("10"),
        );
        mockChartWidth(600);
        rerender(
            <Results
                messages={messages}
                placements={{ sales: { pinned: false } }}
            />,
        );
        await waitFor(() =>
            expect(container.querySelector(".recharts-xAxis")).not.toBeNull(),
        );
        expect(container.querySelector(".recharts-yAxis")).toBeNull();
        expect(
            container.querySelector(".recharts-yAxis-tick-labels"),
        ).toBeNull();
    },
);

test("an in-place widget update retains its sidebar placement", async () => {
    const messages = surface([bar("Original")]);
    const placements = { sales: { pinned: true } };
    const { rerender } = render(
        <Results messages={messages} placements={placements} />,
    );
    await waitFor(() => expect(screen.getByText("Original")).toBeTruthy());
    const updated = a2uiBatchSchema.parse([
        {
            version: "v0.9",
            updateComponents: {
                surfaceId: "sales",
                components: [bar("Updated")],
            },
        },
    ]);
    rerender(
        <Results
            messages={[...messages, ...updated]}
            placements={placements}
        />,
    );
    await waitFor(() => expect(screen.getByText("Updated")).toBeTruthy());
    expect(screen.queryByText("Original")).toBeNull();
    expect(screen.getAllByRole("article")).toHaveLength(1);
    expect(
        screen.getByRole("complementary").contains(screen.getByRole("article")),
    ).toBe(true);
});

test("the center holds at most four widgets and overflow stays visible on the side", async () => {
    const messages = [1, 2, 3, 4, 5].flatMap((index) =>
        surface([bar(`Chart ${index}`)], `chart_${index}`),
    );
    render(<Results messages={messages} />);
    await waitFor(() => expect(screen.getByText("Chart 5")).toBeTruthy());
    expect(
        screen
            .getByRole("region", { name: "Center widgets" })
            .querySelectorAll("article"),
    ).toHaveLength(4);
    expect(
        screen.getByRole("complementary").querySelectorAll("article"),
    ).toHaveLength(1);
    expect(screen.getByRole("complementary").textContent).toContain("Chart 1");
});
