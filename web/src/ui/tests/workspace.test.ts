import assert from "node:assert/strict";
import { test } from "node:test";
import { messageSchema } from "../lib/workspace.ts";
import { a2uiBatchSchema, catalogId } from "../lib/a2ui.ts";

test("malformed events are rejected at the browser boundary", () => {
    for (const event of [
        { type: "code_delta", text: "print('unused')" },
        { type: "done" },
    ])
        assert.equal(messageSchema.safeParse(event).success, false);
    assert.equal(
        messageSchema.safeParse({ type: "result", stdout: 42 }).success,
        false,
    );
    assert.equal(
        messageSchema.safeParse({
            type: "snapshot",
            events: [{ type: "unknown" }],
            busy: false,
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
        busy: false,
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

test("UI events are accepted at the browser boundary", () => {
    const event = messageSchema.parse({
        type: "a2ui",
        messages: [
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
                            data: [{ type: "bar", x: ["Revenue"], y: [42] }],
                        },
                    ],
                },
            },
        ],
    });
    assert.equal(event.type, "a2ui");
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

test("removed widgets, invalid charts, and unknown catalog IDs are rejected", () => {
    for (const component of [
        { id: "root", component: "Image", url: "https://example.com" },
        { id: "root", component: "Text", text: "Hello" },
        { id: "root", component: "Metric", label: "Revenue", value: "$42" },
        { id: "root", component: "Column", children: ["root"] },
        { id: "root", component: "Row", children: ["root"] },
        { id: "root", component: "Card", children: ["root"] },
        {
            id: "root",
            component: "Table",
            columns: ["Name"],
            rows: [["Cell"]],
        },
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

test("pie traces reject negative shares while Cartesian traces accept them", () => {
    for (const type of ["bar", "scatter", "pie"]) {
        const result = a2uiBatchSchema.safeParse([
            {
                version: "v0.9",
                updateComponents: {
                    surfaceId: "chart",
                    components: [
                        {
                            id: "root",
                            component: "Figure",
                            data: [
                                type === "pie"
                                    ? { type, labels: ["Loss"], values: [-42] }
                                    : { type, x: ["Loss"], y: [-42] },
                            ],
                        },
                    ],
                },
            },
        ]);
        assert.equal(result.success, type !== "pie");
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
