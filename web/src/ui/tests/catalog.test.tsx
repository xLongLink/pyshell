import { afterEach, beforeEach, expect, test, vi } from "vitest";
import {
    cleanup,
    fireEvent,
    render,
    screen,
    waitFor,
} from "@testing-library/react";
import { Results } from "../components/results";
import { a2uiBatchSchema, catalogId } from "../lib/a2ui";

beforeEach(() => {
    vi.stubGlobal(
        "ResizeObserver",
        class {
            constructor(private callback: ResizeObserverCallback) {}
            observe(target: Element) {
                this.callback(
                    [
                        {
                            target,
                            contentRect: { width: 600, height: 300 },
                        } as ResizeObserverEntry,
                    ],
                    this as unknown as ResizeObserver,
                );
            }
            unobserve() {}
            disconnect() {}
        },
    );
});

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
            expect(screen.getAllByText("Apples")).toHaveLength(1);
            expect(screen.queryByText("Pears")).toBeNull();
        }
    },
);

test("pie labels are rendered as text, never as generated CSS keys", async () => {
    const label = '</style><img src=x onerror="alert(1)">';
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
    const style = container.querySelector(
        '[data-slot="chart"] style',
    )?.textContent;
    expect(style).toContain("--color-slice_0");
    expect(style).not.toContain(label);
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
                        yaxis: { title: { text: "Amount" } },
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
    expect(screen.getByText("Amount")).toBeTruthy();
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
                first: { pinned: false, focused: true },
                second: { pinned: true, focused: false },
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
                first: { pinned: false, focused: true },
                second: { pinned: false, focused: true },
            }}
        />,
    );
    expect(screen.queryByRole("complementary")).toBeNull();
    expect(screen.getAllByRole("article")[0].getAttribute("aria-label")).toBe(
        "Widget first",
    );
});

test("an in-place widget update retains its sidebar placement", async () => {
    const messages = surface([bar("Original")]);
    const placements = { sales: { pinned: true, focused: false } };
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
