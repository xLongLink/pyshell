import { z } from "zod";
import { a2uiBatchSchema } from "./a2ui.ts";

export const placementsSchema = z.record(
  z.string().regex(/^[A-Za-z0-9_-]{1,64}$/),
  z.object({ pinned: z.boolean(), focused: z.boolean() }).strict(),
);
export type Placements = z.infer<typeof placementsSchema>;

export const eventSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("a2ui"), messages: a2uiBatchSchema }),
  z.object({ type: z.literal("placement"), widgets: placementsSchema }),
  z.object({ type: z.literal("instruction"), text: z.string() }),
  z.object({ type: z.literal("code_start") }),
  z.object({ type: z.literal("code_delta"), text: z.string() }),
  z.object({ type: z.literal("execution_start") }),
  z.object({
    type: z.literal("result"),
    stdout: z.string(),
    stderr: z.string(),
    exception: z.string().nullable(),
    done: z.boolean(),
  }),
  z.object({ type: z.literal("done") }),
  z.object({ type: z.literal("error"), text: z.string() }),
  z.object({ type: z.literal("reset") }),
  z.object({ type: z.literal("state"), busy: z.boolean() }),
]);

export const messageSchema = z.union([
  eventSchema,
  z.object({
    type: z.literal("snapshot"),
    events: z.array(eventSchema),
    busy: z.boolean(),
    model: z.string(),
    placements: placementsSchema.default({}),
  }),
]);

export type Entry = {
  kind: "instruction" | "code" | "output" | "error" | "status";
  text: string;
};

export function applyEvent(
  entries: Entry[],
  event: z.infer<typeof eventSchema>,
): Entry[] {
  switch (event.type) {
    case "reset":
      return [];
    case "instruction":
      return [...entries, { kind: "instruction", text: event.text }];
    case "code_start":
      return [...entries, { kind: "code", text: "" }];
    case "code_delta": {
      const last = entries.at(-1);
      if (last?.kind !== "code") return entries;
      return [
        ...entries.slice(0, -1),
        { ...last, text: last.text + event.text },
      ];
    }
    case "result": {
      const next = [...entries];
      if (event.stdout) next.push({ kind: "output", text: event.stdout });
      if (event.stderr) next.push({ kind: "error", text: event.stderr });
      if (event.exception) next.push({ kind: "error", text: event.exception });
      return next;
    }
    case "error":
      return [...entries, { kind: "error", text: event.text }];
    case "done":
      return [...entries, { kind: "status", text: "Task complete" }];
    default:
      return entries;
  }
}
