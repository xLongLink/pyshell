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
      { version: "v0.9", createSurface: { surfaceId: "sales", catalogId } },
      {
        version: "v0.9",
        updateComponents: {
          surfaceId: "sales",
          components: [
            { id: "root", component: "Metric", label: "Revenue", value: "$42" },
          ],
        },
      },
    ],
  });
  assert.notEqual(event.type, "snapshot");
  if (event.type === "snapshot") return;
  assert.deepEqual(applyEvent([], event), []);
});

test("unapproved components, catalog IDs, and cyclic layouts are rejected", () => {
  for (const component of [
    { id: "root", component: "Image", url: "https://example.com" },
    { id: "root", component: "Text", text: "Hello", html: "<script>" },
    { id: "root", component: "Column", children: ["root"] },
    { id: "root", component: "Column", children: ["missing"] },
    {
      id: "root",
      component: "Table",
      columns: ["Name"],
      rows: [["Extra", "Cell"]],
    },
  ]) {
    assert.equal(
      a2uiBatchSchema.safeParse([
        {
          version: "v0.9",
          updateComponents: { surfaceId: "sales", components: [component] },
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
            { id: "root", component: "Text", text: "x".repeat(10001) },
          ],
        },
      },
    ]).success,
    false,
  );
});

test("pie charts reject negative shares while bar and line charts accept them", () => {
  for (const component of ["BarChart", "LineChart", "PieChart"]) {
    const result = a2uiBatchSchema.safeParse([
      {
        version: "v0.9",
        updateComponents: {
          surfaceId: "chart",
          components: [
            { id: "root", component, points: [{ label: "Loss", value: -42 }] },
          ],
        },
      },
    ]);
    assert.equal(result.success, component !== "PieChart");
  }
});
