import assert from "node:assert/strict";
import { test } from "vitest";
import { messageSchema } from "../lib/workspace.ts";
import { a2uiBatchSchema, catalogId } from "../lib/a2ui.ts";

test("malformed events are rejected at the browser boundary", () => {
    for (const event of [
        { type: "code_delta", text: "print('unused')" },
        { type: "done" },
        { type: "code_start" },
        { type: "execution_start" },
        { type: "stdout_delta", text: "output" },
        { type: "stderr_delta", text: "warning" },
        { type: "result", exception: null },
    ])
        assert.equal(messageSchema.safeParse(event).success, false);
    assert.equal(
        messageSchema.safeParse({
            type: "snapshot",
            events: [{ type: "unknown" }],
        }).success,
        false,
    );
});

test("placement events and snapshots use one pinned flag", () => {
    const widgets = { sales: { pinned: true } };
    assert.deepEqual(messageSchema.parse({ type: "placement", widgets }), {
        type: "placement",
        widgets,
    });
    const snapshot = messageSchema.parse({
        type: "snapshot",
        events: [],
        placements: widgets,
    });
    assert.equal(snapshot.type, "snapshot");
    if (snapshot.type === "snapshot")
        assert.deepEqual(snapshot.placements, widgets);
    assert.equal(
        messageSchema.safeParse({
            type: "placement",
            widgets: { sales: { pinned: false, focused: true } },
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
    for (const event of [
        { type: "instruction", text: "Plot sales" },
        { type: "error", text: "Execution failed" },
        { type: "placement", widgets: { sales: { pinned: true } } },
        { type: "voice", name: "closed" },
    ]) {
        assert.equal(messageSchema.safeParse(event).success, true);
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
            messageSchema.safeParse({
                type: "placement",
                widgets: { [surfaceId]: { pinned: false } },
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
