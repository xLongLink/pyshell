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
    assert.deepEqual(messageSchema.parse(snapshot), {
        surfaces: [{ id: "sales", props: undefined }],
        placements: {},
    });
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
                creation("sales"),
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
            creation("chart"),
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

function creation(surfaceId: string) {
    return { version: "v0.9", createSurface: { surfaceId, catalogId } };
}

function update(surfaceId: string) {
    return {
        version: "v0.9",
        updateComponents: {
            surfaceId,
            components: [
                {
                    id: "root",
                    component: "Figure",
                    data: [{ type: "bar", x: ["A"], y: [1] }],
                },
            ],
        },
    };
}

function snapshot(batches: unknown[][]) {
    return {
        type: "snapshot",
        events: batches.map((messages) => ({ type: "a2ui", messages })),
    };
}

test.each([
    { name: "empty batch", batches: [[]] },
    { name: "update without creation", batches: [[update("sales")]] },
    {
        name: "update before creation",
        batches: [[update("sales"), creation("sales")]],
    },
    {
        name: "mismatched update",
        batches: [[creation("sales"), update("other")]],
    },
    {
        name: "duplicate creation in batch",
        batches: [[creation("sales"), creation("sales")]],
    },
    {
        name: "duplicate creation across batches",
        batches: [[creation("sales")], [creation("sales")]],
    },
    {
        name: "second update",
        batches: [[creation("sales"), update("sales"), update("sales")]],
    },
    {
        name: "delete",
        batches: [
            [
                creation("sales"),
                { version: "v0.9", deleteSurface: { surfaceId: "sales" } },
            ],
        ],
    },
    {
        name: "cross-batch update",
        batches: [[creation("sales")], [update("sales")]],
    },
])(
    "snapshots reject $name without producing partial surfaces",
    ({ batches }) => {
        assert.equal(
            messageSchema.safeParse(snapshot([[creation("valid")], ...batches]))
                .success,
            false,
        );
    },
);

test("snapshots preserve creation order, empty surfaces, defaults and placements", () => {
    const input = {
        ...snapshot([
            [creation("second"), update("second")],
            [creation("first")],
        ]),
        placements: { first: { pinned: true }, second: { pinned: false } },
    };
    const parsed = messageSchema.parse(input);
    assert.deepEqual(
        parsed.surfaces.map(({ id }) => id),
        ["second", "first"],
    );
    assert.equal(parsed.surfaces[0].props?.layout.showlegend, true);
    assert.equal(parsed.surfaces[1].props, undefined);
    assert.deepEqual(parsed.placements, input.placements);
    assert.deepEqual(messageSchema.parse(snapshot([])), {
        surfaces: [],
        placements: {},
    });
});

test("snapshots accept at most 20 current surfaces", () => {
    const batches = Array.from({ length: 21 }, (_, i) => [
        creation(`surface_${i}`),
    ]);
    assert.equal(
        messageSchema.parse(snapshot(batches.slice(0, 20))).surfaces.length,
        20,
    );
    assert.equal(messageSchema.safeParse(snapshot(batches)).success, false);
});
