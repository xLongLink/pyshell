import assert from "node:assert/strict";
import { test } from "vitest";
import { messageSchema, placementsSchema } from "../lib/workspace.ts";
import { a2uiBatchSchema, catalogId } from "../lib/a2ui.ts";

test("snapshots use one pinned flag", () => {
    const widgets = { sales: { pinned: true } };
    const snapshot = messageSchema.parse({
        type: "snapshot",
        events: [],
        placements: widgets,
    });
    assert.deepEqual(snapshot.placements, widgets);
    assert.equal(
        placementsSchema.safeParse({
            sales: { pinned: false, focused: true },
        }).success,
        false,
    );
});

test("snapshots accept UI batches but reject live-only events", () => {
    const events = [
        {
            type: "a2ui",
            messages: [
                {
                    version: "v0.9",
                    createSurface: { surfaceId: "sales", catalogId },
                },
            ],
        },
    ];
    const snapshot = { type: "snapshot", events, placements: {} };
    assert.deepEqual(messageSchema.parse(snapshot), snapshot);
    assert.equal(
        messageSchema.safeParse({ ...snapshot, type: "done" }).success,
        false,
    );
    for (const event of [
        { type: "instruction", text: "Plot sales" },
        { type: "error", text: "Execution failed" },
        { type: "placement", widgets: { sales: { pinned: true } } },
        { type: "voice", name: "closed" },
        { type: "unknown" },
    ]) {
        assert.equal(
            messageSchema.safeParse({ ...snapshot, events: [...events, event] })
                .success,
            false,
        );
    }
});

test("surface IDs and placement keys enforce the same boundaries", () => {
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
        assert.equal(
            placementsSchema.safeParse({
                [surfaceId]: { pinned: false },
            }).success,
            valid,
        );
    }
});

test("unsupported widgets and invalid charts are rejected", () => {
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
