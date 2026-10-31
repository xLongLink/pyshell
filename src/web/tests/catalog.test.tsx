import { afterEach, expect, test, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { Results } from "../components/results";
import { figureProps } from "../lib/a2ui";
import type { Surface } from "../lib/workspace";
import type { FigureProps } from "../components/plot/common";

vi.mock("../components/plot/figure", () => ({
    Figure: ({ layout, compact }: FigureProps & { compact?: boolean }) => (
        <div role="application" tabIndex={0} data-compact={compact}>
            {layout.title.text}
        </div>
    ),
}));

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
    await screen.findByText("Second");
    expect(document.activeElement).toBe(chart);
    expect(
        screen
            .getByRole("article", { name: "Widget first" })
            .querySelector('[role="application"]'),
    ).toBe(chart);
});

test("replacement snapshots replace surfaces even when their count is unchanged", async () => {
    const { rerender } = render(<Results surfaces={bar("Old", "old")} />);
    await screen.findByText("Old");
    rerender(<Results surfaces={bar("New", "new")} />);
    await screen.findByText("New");
    expect(screen.queryByText("Old")).toBeNull();
    expect(screen.getAllByRole("article")).toHaveLength(1);
});

test("backend placement moves widgets between center and sidebar without controls or visible IDs", () => {
    const surfaces = ["first", "second"].flatMap((id) =>
        figure([{ type: "bar", x: ["A"], y: [null] }], undefined, id),
    );
    const { rerender } = render(<Results surfaces={surfaces} />);
    const second = screen.getByRole("article", { name: "Widget second" });
    expect(
        second
            .querySelector('[role="application"]')
            ?.getAttribute("data-compact"),
    ).toBe("false");
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
    expect(
        screen
            .getByRole("article", { name: "Widget second" })
            .querySelector('[role="application"]')
            ?.getAttribute("data-compact"),
    ).toBe("true");
    rerender(<Results surfaces={surfaces} />);
    expect(screen.queryByRole("complementary")).toBeNull();
    expect(
        screen
            .getByRole("article", { name: "Widget second" })
            .querySelector('[role="application"]')
            ?.getAttribute("data-compact"),
    ).toBe("false");
    expect(screen.getAllByRole("article")[0].getAttribute("aria-label")).toBe(
        "Widget first",
    );
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
