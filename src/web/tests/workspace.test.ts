import assert from "node:assert/strict";
import { test } from "vitest";
import { messageSchema, placementsSchema } from "../lib/workspace.ts";
import { a2uiBatchSchema } from "../lib/a2ui.ts";
import { creation, update } from "./workspace-fixtures.ts";

test("snapshots accept UI batches but reject live-only events", () => {
    const events = [
        {
            type: "a2ui",
            messages: [creation("sales"), update("sales")],
        },
    ];
    const snapshot = {
        type: "snapshot",
        events,
        placements: { sales: { pinned: false } },
    };
    assert.equal(messageSchema.parse(snapshot).surfaces[0].id, "sales");
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
            a2uiBatchSchema.safeParse([creation(surfaceId), update(surfaceId)])
                .success,
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

test.each([
    ["a", 10000, true],
    ["😀", 6000, true],
    ["😀", 10000, true],
    ["a", 10001, false],
    ["😀", 10001, false],
] as const)(
    "chart text limits count Unicode code points: %s × %i",
    (character, length, valid) => {
        const text = character.repeat(length);
        const content = {
            version: "v0.9",
            updateComponents: {
                surfaceId: "unicode",
                components: [
                    {
                        id: "root",
                        component: "Figure",
                        data: [{ type: "bar", name: text, x: [text], y: [1] }],
                        layout: { title: { text } },
                    },
                ],
            },
        };
        const parsed = messageSchema.safeParse(
            snapshot([
                [creation("valid"), update("valid")],
                [creation("unicode"), content],
            ]),
        );
        assert.equal(parsed.success, valid);
        if (parsed.success) {
            assert.equal(parsed.data.surfaces.length, 2);
            assert.equal(parsed.data.surfaces[1].props.layout.title.text, text);
        }
    },
);

function snapshot(batches: unknown[][]) {
    return {
        type: "snapshot",
        events: batches.map((messages) => ({ type: "a2ui", messages })),
        placements: Object.fromEntries(
            batches.flatMap((messages) => {
                const first = messages[0] as
                    ReturnType<typeof creation> | undefined;
                return first?.createSurface
                    ? [[first.createSurface.surfaceId, { pinned: true }]]
                    : [];
            }),
        ),
    };
}

test.each([
    { name: "creation without content", batches: [[creation("sales")]] },
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
        batches: [
            [creation("sales"), update("sales")],
            [creation("sales"), update("sales")],
        ],
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
])(
    "snapshots reject $name without producing partial surfaces",
    ({ batches }) => {
        assert.equal(
            messageSchema.safeParse(
                snapshot([[creation("valid"), update("valid")], ...batches]),
            ).success,
            false,
        );
    },
);

test("snapshots preserve creation order, complete surfaces, defaults and placements", () => {
    const input = {
        ...snapshot([
            [creation("second"), update("second")],
            [creation("first"), update("first")],
        ]),
        placements: { first: { pinned: true }, second: { pinned: false } },
    };
    const parsed = messageSchema.parse(input);
    assert.equal(parsed.surfaces[0].props.layout.showlegend, true);
    assert.equal(parsed.surfaces[1].props.data[0].type, "bar");
    assert.deepEqual(
        parsed.surfaces.map(({ id, pinned }) => ({ id, pinned })),
        [
            { id: "second", pinned: false },
            { id: "first", pinned: true },
        ],
    );
    assert.equal(Object.hasOwn(parsed, "placements"), false);
    assert.equal(
        placementsSchema.safeParse({
            sales: { pinned: false, focused: true },
        }).success,
        false,
    );
    assert.deepEqual(messageSchema.parse(snapshot([])), {
        surfaces: [],
    });
});

test("snapshots accept at most 20 current surfaces", () => {
    const batches = Array.from({ length: 21 }, (_, i) => [
        creation(`surface_${i}`),
        update(`surface_${i}`),
    ]);
    assert.equal(
        messageSchema.parse(snapshot(batches.slice(0, 20))).surfaces.length,
        20,
    );
    assert.equal(messageSchema.safeParse(snapshot(batches)).success, false);
});

test("snapshots require exactly one placement for each current surface", () => {
    const valid = snapshot([[creation("sales"), update("sales")]]);
    const { placements: _, ...missing } = valid;
    for (const input of [
        missing,
        {
            ...valid,
            placements: { sales: { pinned: false }, extra: { pinned: true } },
        },
        { ...valid, placements: { other: { pinned: true } } },
    ]) {
        assert.equal(messageSchema.safeParse(input).success, false);
    }
    const parsed = messageSchema.safeParse({ ...valid, placements: {} });
    assert.equal(parsed.success, false);
    if (!parsed.success)
        assert.match(parsed.error.message, /Placements must match/);
});

test("snapshots reject unpinned overflow rather than reassigning placements", () => {
    const input = snapshot(
        Array.from({ length: 5 }, (_, i) => [
            creation(`chart_${i}`),
            update(`chart_${i}`),
        ]),
    );
    input.placements = Object.fromEntries(
        Object.keys(input.placements).map((id) => [id, { pinned: false }]),
    );
    const parsed = messageSchema.safeParse(input);
    assert.equal(parsed.success, false);
    if (!parsed.success)
        assert.match(parsed.error.message, /At most four surfaces/);
    input.placements.chart_0.pinned = true;
    assert.equal(messageSchema.parse(input).surfaces.length, 5);
});
