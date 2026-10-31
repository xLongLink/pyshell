import { z } from "zod";
import { a2uiBatchSchema, identifier } from "./a2ui.ts";

export const placementsSchema = z.record(
    identifier,
    z.strictObject({ pinned: z.boolean() }),
);
export type Placements = z.infer<typeof placementsSchema>;

export const messageSchema = z.object({
    type: z.literal("snapshot"),
    events: z
        .array(z.object({ type: z.literal("a2ui"), messages: a2uiBatchSchema }))
        .max(20),
    placements: placementsSchema.default({}),
});
