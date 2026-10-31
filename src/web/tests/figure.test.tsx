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
