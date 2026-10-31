import { afterEach, expect, test, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { Figure } from "../components/plot/figure";
import { Cartesian } from "../components/plot/cartesian";
import { figureProps } from "../lib/a2ui";

vi.mock("../components/plot/cartesian", () => ({
    Cartesian: vi.fn(() => <div>Cartesian plot</div>),
}));

afterEach(() => {
    cleanup();
    vi.clearAllMocks();
});

test.each(["bar", "scatter"] as const)(
    "%s-first dispatch passes the original array including later non-null traces",
    (type) => {
        const props = figureProps.parse({
            data: [
                { type, x: [0], y: [null] },
                { type: type === "bar" ? "scatter" : "bar", x: [0], y: [0] },
            ],
        });
        render(<Figure {...props} />);
        expect(screen.getByText("Cartesian plot")).toBeTruthy();
        expect(screen.queryByText("No values to display.")).toBeNull();
        expect(vi.mocked(Cartesian).mock.calls[0][0].data).toBe(props.data);
        expect(vi.mocked(Cartesian).mock.calls[0][0].layout).toBe(props.layout);
    },
);
