import { afterEach, expect, test } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { Results } from "../components/results";
import { a2uiBatchSchema, catalogId } from "../lib/a2ui";

afterEach(cleanup);

test("a created surface stays empty until content arrives and updates retain its chart", () => {
    const created = a2uiBatchSchema.parse([
        { version: "v0.9", createSurface: { surfaceId: "sales", catalogId } },
    ]);
    const { rerender } = render(<Results messages={created} />);
    const article = screen.getByRole("article", { name: "Widget sales" });
    expect(article.children).toHaveLength(0);
    expect(screen.queryByRole("alert")).toBeNull();

    const update = (title: string) =>
        a2uiBatchSchema.parse([
            {
                version: "v0.9",
                updateComponents: {
                    surfaceId: "sales",
                    components: [
                        {
                            id: "root",
                            component: "Figure",
                            data: [{ type: "bar", x: ["A"], y: [42] }],
                            layout: { title: { text: title } },
                        },
                    ],
                },
            },
        ]);
    const messages = [...created, ...update("First")];
    rerender(<Results messages={messages} />);
    const chart = screen.getByRole("application");
    chart.focus();
    rerender(<Results messages={[...messages, ...update("Updated")]} />);
    expect(screen.getByText("Updated")).toBeTruthy();
    expect(screen.getByRole("application")).toBe(chart);
    expect(document.activeElement).toBe(chart);
});
