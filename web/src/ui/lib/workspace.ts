import { z } from "zod";
import { a2uiBatchSchema } from "./a2ui.ts";

export const placementsSchema = z.record(
    z.string().regex(/^[A-Za-z0-9_-]{1,64}$/),
    z.object({ pinned: z.boolean() }).strict(),
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
