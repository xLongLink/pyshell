import assert from "node:assert/strict";
import { test } from "vitest";
import { a2uiBatchSchema, catalogId } from "../lib/a2ui.ts";

test("surface IDs enforce identifier boundaries", () => {
    for (const [surfaceId, valid] of [
        ["", false],
        ["invalid/id", false],
        ["a".repeat(64), true],
        ["a".repeat(65), false],
        ["Sales_2026-01", true],
    ] as const) {
        assert.equal(
            a2uiBatchSchema.safeParse([
                {
                    version: "v0.9",
                    createSurface: { surfaceId, catalogId },
                },
            ]).success,
            valid,
        );
    }
});

test("removed widgets, invalid charts, and unknown catalog IDs are rejected", () => {
    for (const component of [
        { id: "root", component: "Image", url: "https://example.com" },
        {
            id: "root",
            component: "Figure",
            data: [{ type: "bar", x: ["A"], y: [1] }],
            html: "<script>",
        },
        {
            id: "other",
            component: "Figure",
            data: [{ type: "scatter", x: ["A"], y: [1] }],
        },
    ]) {
        assert.equal(
            a2uiBatchSchema.safeParse([
                {
                    version: "v0.9",
                    updateComponents: {
                        surfaceId: "sales",
                        components: [component],
                    },
                },
            ]).success,
            false,
        );
    }
    assert.equal(
        a2uiBatchSchema.safeParse([
            {
                version: "v0.9",
                createSurface: {
                    surfaceId: "sales",
                    catalogId: "https://untrusted.example/catalog",
                },
            },
        ]).success,
        false,
    );
});

test("oversized collections fail before reaching A2UI's schema processor", () => {
    assert.equal(
        a2uiBatchSchema.safeParse(new Array(100000).fill(null)).success,
        false,
    );
    assert.equal(
        a2uiBatchSchema.safeParse([
            {
                version: "v0.9",
                updateComponents: {
                    surfaceId: "sales",
                    components: [
                        {
                            id: "root",
                            component: "Figure",
                            layout: { title: { text: "x".repeat(10001) } },
                            data: [{ type: "bar", x: ["A"], y: [1] }],
                        },
                    ],
                },
            },
        ]).success,
        false,
    );
});

test("each surface accepts exactly one chart", () => {
    const chart = {
        id: "root",
        component: "Figure",
        data: [{ type: "bar", x: ["A"], y: [1] }],
    };
    assert.equal(
        a2uiBatchSchema.safeParse([
            {
                version: "v0.9",
                updateComponents: {
                    surfaceId: "chart",
                    components: [chart, chart],
                },
            },
        ]).success,
        false,
    );
});
