import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
    componentSchema,
    figureProps,
    catalogId,
    a2uiBatchSchema,
} from "../lib/a2ui.ts";

const cases = JSON.parse(
    readFileSync(
        new URL("../../../../api/tests/fixtures/figures.json", import.meta.url),
        "utf8",
    ),
) as {
    name: string;
    valid: boolean;
    browserValid?: boolean;
    figure: unknown;
}[];

for (const entry of cases) {
    test(`figure contract: ${entry.name}`, () => {
        const valid = entry.browserValid ?? entry.valid;
        assert.equal(figureProps.safeParse(entry.figure).success, valid);
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

test("figure point and trace limits match Python", () => {
    for (const [traces, points, valid] of [
        [1, 200, true],
        [2, 100, true],
        [2, 101, false],
        [20, 1, true],
        [21, 1, false],
        [1, 201, false],
    ] as const) {
        const data = Array.from({ length: traces }, () => ({
            type: "scatter",
            x: Array.from({ length: points }, (_, i) => i),
            y: Array(points).fill(1),
        }));
        assert.equal(figureProps.safeParse({ data }).success, valid);
    }
});

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
