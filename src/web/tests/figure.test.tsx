import { afterEach, expect, test } from "vitest";
import {
    cleanup,
    fireEvent,
    render,
    screen,
    waitFor,
} from "@testing-library/react";
import { Figure } from "../components/plot/figure";
import { figureProps } from "../lib/a2ui";

afterEach(cleanup);

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
    const props = figureProps.parse({
        data: [
            {
                type: "scatter",
                name: "Hidden legend",
                mode: "lines+markers",
                x: [2, 0, 2, 3, 4, 5],
                y: [3, 1, 4, null, 5, 2],
            },
        ],
        layout: { showlegend: false },
    });
    const { container } = render(<Figure {...props} />);
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
