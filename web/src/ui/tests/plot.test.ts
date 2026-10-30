import assert from "node:assert/strict";
import { test } from "vitest";
import { figureProps, catalogId, a2uiBatchSchema } from "../lib/a2ui.ts";

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

test("all nonfinite values are rejected", () => {
    for (const value of [NaN, Infinity, -Infinity]) {
        assert.equal(
            figureProps.safeParse({
                data: [{ type: "scatter", x: [value], y: [1] }],
            }).success,
            false,
        );
    }
});

test("other numeric fields reject nonfinite values", () => {
    for (const trace of [
        { type: "scatter", x: [1], y: [Infinity] },
        { type: "pie", labels: ["A"], values: [Infinity] },
    ]) {
        assert.equal(figureProps.safeParse({ data: [trace] }).success, false);
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
    assert.equal(
        figureProps.safeParse({ data: [{ ...trace, values: [Infinity] }] })
            .success,
        false,
    );
});

test.each([
    { traces: 20, points: 10, valid: true },
    { traces: 21, points: 1, valid: false },
    { traces: 1, points: 200, valid: true },
    { traces: 1, points: 201, valid: false },
    { traces: 2, points: 101, valid: false },
])(
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

test("only the current catalog is accepted", () => {
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
});
