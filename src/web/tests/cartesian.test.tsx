import { afterEach, expect, test } from "vitest";
import {
    cleanup,
    fireEvent,
    render,
    screen,
    waitFor,
} from "@testing-library/react";
import { Cartesian } from "../components/plot/cartesian";
import type { Trace } from "../components/plot/common";
import { figureProps } from "../lib/a2ui";

afterEach(cleanup);

test.each(["bar", "scatter"] as const)(
    "%s plots display the empty state when every series is missing",
    (type) => {
        const props = figureProps.parse({
            data: [
                { type, x: [0], y: [null] },
                { type, x: [0], y: [null] },
            ],
        });
        const data = props.data as Extract<Trace, { type: "bar" | "scatter" }>[];

        render(<Cartesian data={data} layout={props.layout} />);

        expect(screen.getByText("No values to display.").className).toBe(
            "py-8 text-sm text-muted-foreground",
        );
        expect(screen.queryByRole("application")).toBeNull();
    },
);

test.each(["bar", "scatter"] as const)(
    "%s plots retain a later zero-valued series when the first is all missing",
    async (type) => {
        const props = figureProps.parse({
            data: [
                { type, x: [0], y: [null] },
                { type, name: "Zero series", x: [0], y: [0] },
            ],
        });
        const data = props.data as Extract<Trace, { type: "bar" | "scatter" }>[];

        const { container } = render(
            <Cartesian data={data} layout={props.layout} />,
        );
        fireEvent.focus(await screen.findByRole("application"));

        await waitFor(() =>
            expect(
                container.querySelector(".recharts-tooltip-wrapper")
                    ?.textContent,
            ).toBe("0Zero series0"),
        );
        expect(screen.queryByText("No values to display.")).toBeNull();
    },
);
