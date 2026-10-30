import assert from "node:assert/strict";
import { test } from "node:test";
import { applyEvent, messageSchema, type Entry } from "../src/lib/workspace.ts";

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
