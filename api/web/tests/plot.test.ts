import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { test } from "vitest";
import {
    componentSchema,
    figureProps,
    catalogId,
    a2uiBatchSchema,
} from "../lib/a2ui.ts";

const cases = JSON.parse(
    readFileSync(
        resolve(import.meta.dirname, "../../tests/fixtures/figures.json"),
        "utf8",
    ),
) as {
    name: string;
    valid: boolean;
    browserValid?: boolean;
    figure: unknown;
}[];

const limitCases = JSON.parse(
    readFileSync(
        resolve(import.meta.dirname, "../../tests/fixtures/figure_limits.json"),
        "utf8",
    ),
) as { traces: number; points: number; valid: boolean }[];

for (const entry of cases) {
    test(`figure contract: ${entry.name}`, () => {
        const valid = entry.browserValid ?? entry.valid;
        assert.equal(
            componentSchema.safeParse({
                id: "root",
                component: "Figure",
                ...(entry.figure as object),
            }).success,
            valid,
        );
    });
}

test("figure defaults match the Python wire model", () => {
    assert.deepEqual(
        figureProps.parse({ data: [{ type: "scatter", x: [1], y: [2] }] }),
        {
            data: [
                { type: "scatter", name: "", mode: "lines", x: [1], y: [2] },
            ],
            layout: {
                title: { text: "" },
                xaxis: { title: { text: "" } },
                showlegend: true,
            },
        },
    );
});

test("nonfinite coordinates are rejected", () => {
    for (const value of [NaN, Infinity, -Infinity]) {
        for (const field of ["x", "y", "values"]) {
            const trace =
                field === "values"
                    ? {
                          ...{ type: "pie", labels: ["A"], values: [1] },
                          [field]: [value],
                      }
                    : {
                          ...{ type: "scatter", x: [1], y: [1] },
                          [field]: [value],
                      };
            assert.equal(
                figureProps.safeParse({ data: [trace] }).success,
                false,
            );
        }
    }
});

test("map defaults and nonfinite values match Python", () => {
    const trace = { type: "choropleth", locations: ["FRA"], values: [0] };
    assert.deepEqual(figureProps.parse({ data: [trace] }).data[0], {
        ...trace,
        name: "",
        geography: "world",
        level: "country",
    });
    for (const value of [NaN, Infinity, -Infinity]) {
        assert.equal(
            figureProps.safeParse({ data: [{ ...trace, values: [value] }] })
                .success,
            false,
        );
    }
});

test.each(limitCases)(
    "figure limits: $traces traces with $points points (valid: $valid)",
    ({ traces, points, valid }) => {
        const data = Array.from({ length: traces }, () => ({
            type: "scatter",
            x: Array.from({ length: points }, (_, i) => i),
            y: Array(points).fill(1),
        }));
        assert.equal(figureProps.safeParse({ data }).success, valid);
    },
);

test("only the new catalog and figure component are accepted", () => {
    assert.equal(catalogId, "urn:pyshell:catalog:v3");
    assert.equal(
        a2uiBatchSchema.safeParse([
            {
                version: "v0.9",
                createSurface: {
                    surfaceId: "sales",
                    catalogId: "urn:pyshell:catalog:v2",
                },
            },
        ]).success,
        false,
    );
    for (const component of ["BarChart", "LineChart", "PieChart"]) {
        assert.equal(
            componentSchema.safeParse({
                id: "root",
                component,
                points: [{ label: "A", value: 1 }],
            }).success,
            false,
        );
    }
});
