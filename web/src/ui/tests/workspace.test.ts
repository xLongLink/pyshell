import assert from "node:assert/strict";
import { test } from "node:test";
import { applyEvent, messageSchema, type Entry } from "../lib/workspace.ts";
import { a2uiBatchSchema, catalogId } from "../lib/a2ui.ts";

test("a replayed transcript restores streamed code and execution results", () => {
    const snapshot = messageSchema.parse({
        type: "snapshot",
        busy: false,
        model: "test-model",
        events: [
            { type: "instruction", text: "Calculate" },
            { type: "code_start" },
            { type: "code_delta", text: "print(" },
            { type: "code_delta", text: "42)" },
            { type: "execution_start" },
            {
                type: "result",
                stdout: "42\n",
                stderr: "",
                exception: null,
                done: true,
            },
            { type: "done" },
        ],
    });
    assert.equal(snapshot.type, "snapshot");
    if (snapshot.type !== "snapshot") return;
    assert.deepEqual(snapshot.events.reduce(applyEvent, []), [
        { kind: "instruction", text: "Calculate" },
        { kind: "code", text: "print(42)" },
        { kind: "output", text: "42\n" },
        { kind: "status", text: "Task complete" },
    ]);
});

test("execution errors retain output and reset clears the transcript", () => {
    const entries: Entry[] = applyEvent([], {
        type: "result",
        stdout: "partial\n",
        stderr: "warning\n",
        exception: "ValueError",
        done: false,
    });
    assert.deepEqual(entries, [
        { kind: "output", text: "partial\n" },
        { kind: "error", text: "warning\n" },
        { kind: "error", text: "ValueError" },
    ]);
    assert.deepEqual(applyEvent(entries, { type: "reset" }), []);
});

test("malformed events are rejected at the browser boundary", () => {
    assert.equal(
        messageSchema.safeParse({ type: "result", stdout: 42 }).success,
        false,
    );
    assert.equal(
        messageSchema.safeParse({
            type: "snapshot",
            events: [{ type: "unknown" }],
            busy: false,
            model: "test",
        }).success,
        false,
    );
});

test("UI events are accepted without adding JSON to the execution log", () => {
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
    assert.notEqual(event.type, "snapshot");
    if (event.type === "snapshot") return;
    assert.deepEqual(applyEvent([], event), []);
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
