import assert from "node:assert/strict";
import { test } from "vitest";
import { messageSchema } from "../lib/workspace.ts";

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

test("surface IDs and placement keys enforce the same boundaries", () => {
    for (const [surfaceId, valid] of [
        ["", false],
        ["invalid/id", false],
        ["a".repeat(64), true],
        ["a".repeat(65), false],
        ["Sales_2026-01", true],
    ] as const) {
        assert.equal(
            messageSchema.safeParse({
                type: "a2ui",
                messages: [
                    {
                        version: "v0.9",
                        createSurface: {
                            surfaceId,
                            catalogId: "urn:test:catalog:v1",
                        },
                    },
                ],
            }).success,
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
