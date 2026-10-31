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
