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

test.each([{}, { title: {} }, { xaxis: {} }, { xaxis: { title: {} } }])(
    "partial layouts retain nested defaults: %j",
    (layout) => {
        assert.deepEqual(
            figureProps.parse({
                data: [{ type: "bar", x: [0], y: [null] }],
                layout,
            }).layout,
            {
                title: { text: "" },
                xaxis: { title: { text: "" } },
                showlegend: true,
            },
        );
    },
);

test("explicit defaults are not substituted for empty text, false, zero or null", () => {
    const figure = {
        data: [{ type: "bar", name: "", x: [0, 1], y: [0, null] }],
        layout: { title: { text: "" }, showlegend: false },
    };
    const parsed = figureProps.parse(figure);
    assert.equal(parsed.layout.showlegend, false);
    assert.deepEqual(parsed.data[0], figure.data[0]);
});

test.each([
    { extra: true },
    { data: [{ type: "bar", x: [0], y: [1], extra: true }] },
    { layout: { extra: true } },
    { layout: { title: { text: "", extra: true } } },
    { layout: { xaxis: { extra: true } } },
    { layout: { xaxis: { title: { text: "", extra: true } } } },
])("figure objects reject unknown keys at every level: %j", (fields) => {
    assert.equal(
        figureProps.safeParse({
            data: [{ type: "bar", x: [0], y: [1] }],
            ...fields,
        }).success,
        false,
    );
});

test("validated figures contain only one plot family", () => {
    const bar = { type: "bar", x: [0], y: [1] };
    const scatter = { type: "scatter", x: [0], y: [null] };
    const pie = { type: "pie", labels: ["A"], values: [1] };
    const map = { type: "choropleth", locations: ["FRA"], values: [1] };
    for (const data of [
        [bar, pie],
        [pie, bar],
        [scatter, map],
        [map, scatter],
        [pie, map],
        [map, pie],
        [pie, pie],
        [map, map],
    ])
        assert.equal(figureProps.safeParse({ data }).success, false);
    assert.equal(figureProps.safeParse({ data: [bar, scatter] }).success, true);
    assert.equal(figureProps.safeParse({ data: [scatter, bar] }).success, true);
});

test("batch complexity limits stop validation before message parsing", () => {
    let deep: unknown = null;
    for (let i = 0; i < 25; i++) deep = [deep];
    for (const payload of [
        deep,
        [Array(501).fill(null)],
        [Object.fromEntries(Array.from({ length: 17 }, (_, i) => [i, null]))],
        ["x".repeat(10001)],
        [Array.from({ length: 500 }, () => Array(41).fill(null))],
    ]) {
        const parsed = a2uiBatchSchema.safeParse(payload);
        assert.equal(parsed.success, false);
        if (!parsed.success)
            assert.deepEqual(
                parsed.error.issues.map(({ message }) => message),
                ["UI payload exceeds the display limits"],
            );
    }
});

test("snapshot batches reject histories and retain UTF-8 byte limits", () => {
    const create = {
        version: "v0.9",
        createSurface: { surfaceId: "sales", catalogId },
    };
    assert.equal(
        a2uiBatchSchema.safeParse(Array(100).fill(create)).success,
        false,
    );
    const large = {
        version: "v0.9",
        updateComponents: {
            surfaceId: "sales",
            components: [
                {
                    id: "root",
                    component: "Figure",
                    data: [
                        {
                            type: "bar",
                            name: "é".repeat(10000),
                            x: [0],
                            y: [1],
                        },
                    ],
                },
            ],
        },
    };
    assert.equal(a2uiBatchSchema.safeParse([create, large]).success, true);
    const oversized = Array(60).fill(large);
    const serialized = JSON.stringify(oversized);
    assert.ok(serialized.length < 1048576);
    assert.ok(new TextEncoder().encode(serialized).length > 1048576);
    for (const payload of [Array(101).fill(create), oversized]) {
        const parsed = a2uiBatchSchema.safeParse(payload);
        assert.equal(parsed.success, false);
        if (!parsed.success)
            assert.deepEqual(
                parsed.error.issues.map(({ message }) => message),
                ["UI batch exceeds the display limits"],
            );
    }
});

test.each([
    { type: "scatter", x: [Infinity], y: [1] },
    { type: "scatter", x: [1], y: [Infinity] },
    { type: "pie", labels: ["A"], values: [Infinity] },
])("numeric fields reject nonfinite values: %j", (trace) => {
    assert.equal(figureProps.safeParse({ data: [trace] }).success, false);
});

test("map defaults and nonfinite values match Python", () => {
    const trace = { type: "choropleth", locations: ["FRA"], values: [0] };
    assert.deepEqual(figureProps.parse({ data: [trace] }).data[0], {
        ...trace,
        name: "",
        geography: "world",
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

test.each(["v2", "v3"])("old catalog %s is rejected", (version) => {
    assert.equal(
        a2uiBatchSchema.safeParse([
            {
                version: "v0.9",
                createSurface: {
                    surfaceId: "sales",
                    catalogId: `urn:pyshell:catalog:${version}`,
                },
            },
            {
                version: "v0.9",
                updateComponents: {
                    surfaceId: "sales",
                    components: [
                        {
                            id: "root",
                            component: "Figure",
                            data: [{ type: "bar", x: [0], y: [1] }],
                        },
                    ],
                },
            },
        ]).success,
        false,
    );
});

test("the current catalog is v4 and is accepted", () => {
    assert.equal(catalogId, "urn:pyshell:catalog:v4");
    assert.equal(
        a2uiBatchSchema.safeParse([
            {
                version: "v0.9",
                createSurface: { surfaceId: "sales", catalogId },
            },
            {
                version: "v0.9",
                updateComponents: {
                    surfaceId: "sales",
                    components: [
                        {
                            id: "root",
                            component: "Figure",
                            data: [{ type: "bar", x: [0], y: [1] }],
                        },
                    ],
                },
            },
        ]).success,
        true,
    );
});

test.each([
    { geography: "world", level: "country" },
    { geography: "FRA", level: "region" },
])("legacy map level is rejected: %j", (fields) => {
    assert.equal(
        figureProps.safeParse({
            data: [
                {
                    type: "choropleth",
                    locations: ["FRA"],
                    values: [1],
                    ...fields,
                },
            ],
        }).success,
        false,
    );
});

test("regional maps need only geography, locations and values", () => {
    const trace = {
        type: "choropleth",
        geography: "FRA",
        locations: ["FR-IDF"],
        values: [7],
    };
    assert.deepEqual(figureProps.parse({ data: [trace] }).data[0], {
        ...trace,
        name: "",
    });
});
